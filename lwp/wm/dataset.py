"""Episode dataset for world model training.

Loads episodes into memory for a given split and provides architecture-specific
batch sampling methods. All samplers return PyTorch tensors on CPU — the
training loop handles device transfer.

Supervision modes:
  - "labeled": full 15D state (kinematic + physics params). state_dim=15.
  - "blind": kinematic dims only (0:8). Physics dims sliced off, not
    zeroed — the network input is 8D. state_dim=8.

Prediction targets:
  - "absolute": predict s_{t+1} directly.
  - "delta": predict Δs = s_{t+1} - s_t (physics dims map to Δ ≈ 0 automatically).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class EpisodeDataset:
    """In-memory dataset of episodes for a single split.

    At init, loads all episodes for the requested split into memory as
    numpy arrays. Sampling methods convert to PyTorch tensors on the fly.

    Args:
        index_path: Path to the split index JSON (from build_episode_index).
        split: Which split to load ("train", "val", "test", "ood", "policy_holdout").
        supervision: "labeled" (full 15D state) or "blind" (kinematic 8D only).
        prediction_target: "absolute" (predict s_{t+1}) or "delta" (predict Δs).
    """

    def __init__(
        self,
        index_path: str | Path,
        split: str,
        supervision: str = "labeled",
        prediction_target: str = "absolute",
        max_episodes: int | None = None,
    ):
        self.split = split
        self.supervision = supervision
        self.prediction_target = prediction_target

        # Load split index and filter to requested split.
        index = json.loads(Path(index_path).read_text())
        paths = sorted(p for p, s in index.items() if s == split)

        # Optional truncation for smoke testing (--sample flag in CLI).
        if max_episodes is not None:
            paths = paths[:max_episodes]

        # Load all episodes into memory (numpy arrays per episode).
        # At 100K episodes × ~300 steps × 32 floats × 4 bytes ≈ 1GB — fits fine.
        self.states = []      # list of (T_i+1, D) float32 arrays (D=15 labeled, 8 blind)
        self.actions = []     # list of (T_i, 2) float32 arrays
        self.rewards = []     # list of (T_i,) float32 arrays
        self.lengths = []     # list of int (T_i = number of actions)

        skipped = 0
        for p in tqdm(paths, desc=f"Loading {split}", disable=len(paths) < 50):
            try:
                data = np.load(p, allow_pickle=False)
            except Exception as e:
                warnings.warn(f"Skipping corrupt file {p}: {e}")
                skipped += 1
                continue

            states = data["states"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            rewards = data["rewards"].astype(np.float32)

            # Shape validation: states must be (T+1, 15), actions (T, 2).
            if states.ndim != 2 or states.shape[1] != 15:
                warnings.warn(f"Skipping {p}: states shape {states.shape}, expected (T, 15)")
                skipped += 1
                continue
            if actions.ndim != 2 or actions.shape[1] != 2:
                warnings.warn(f"Skipping {p}: actions shape {actions.shape}, expected (T, 2)")
                skipped += 1
                continue

            # Blind mode: slice to kinematic dims only (0:8). The network
            # input is 8D — no dead weights on zeroed physics dims.
            if supervision == "blind":
                states = states[:, :8]

            self.states.append(states)
            self.actions.append(actions)
            self.rewards.append(rewards)
            self.lengths.append(len(actions))

        if skipped > 0:
            print(f"  Warning: skipped {skipped}/{skipped + len(self.states)} episodes")

        self.n_episodes = len(self.states)

    @property
    def state_dim(self) -> int:
        """Effective state dimension (15 for labeled, 8 for blind)."""
        if self.n_episodes == 0:
            return 15 if self.supervision != "blind" else 8
        return self.states[0].shape[1]

    def __len__(self) -> int:
        return self.n_episodes

    def _sample_episode_and_timestep(self, rng: np.random.Generator) -> tuple[int, int]:
        """Pick a random episode and a valid transition timestep within it.

        Samples episodes uniformly (not length-weighted). This means short
        episodes contribute the same expected number of samples as long ones,
        which prevents the model from over-fitting to dynamics patterns that
        happen to appear in longer episodes. If we sampled timesteps uniformly
        across all data, longer episodes would dominate the training signal.
        """
        ep_idx = rng.integers(self.n_episodes)
        t = rng.integers(self.lengths[ep_idx])
        return ep_idx, t

    def sample_transitions(
        self, batch_size: int, rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample random (s, a, s_next, r) transitions.

        For feedforward models (no context). Samples a random episode then a
        random timestep within that episode, independently for each batch item.

        Args:
            batch_size: Number of transitions to sample.
            rng: NumPy random generator. Default creates a new unseeded one.

        Returns:
            s: (B, 15), a: (B, 2), s_next: (B, 15), r: (B,)
            If prediction_target="delta", s_next is replaced by Δs = s_{t+1} - s_t.
        """
        if rng is None:
            rng = np.random.default_rng()

        s_batch, a_batch, snext_batch, r_batch = [], [], [], []
        for _ in range(batch_size):
            ep, t = self._sample_episode_and_timestep(rng)
            s_batch.append(self.states[ep][t])
            a_batch.append(self.actions[ep][t])
            snext_batch.append(self.states[ep][t + 1])
            r_batch.append(self.rewards[ep][t])

        s = torch.from_numpy(np.stack(s_batch))
        a = torch.from_numpy(np.stack(a_batch))
        s_next = torch.from_numpy(np.stack(snext_batch))
        r = torch.from_numpy(np.array(r_batch))

        # Delta prediction: target becomes Δs = s_{t+1} - s_t. Physics dims
        # are constant within an episode, so their delta is exactly 0 — the
        # model automatically learns to not predict physics changes without
        # needing per-dim loss weighting. Kinematic deltas are also typically
        # smaller and more stable than absolute states, which helps training.
        # See training-infrastructure.md "Prediction target" for rationale.
        if self.prediction_target == "delta":
            s_next = s_next - s

        return s, a, s_next, r

    def sample_with_context(
        self, batch_size: int, K: int, rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample transitions with K context transitions from the same episode.

        For context-conditioned models. Each sample includes a target transition
        and K other transitions from the same episode (excluding the target).
        The encoder processes the context to infer a physics-identification
        vector z.

        Context transitions are sampled from timesteps != target_t. If the
        episode has fewer than K+1 transitions, context may include duplicates
        (sampled with replacement from the non-target pool).

        Each context row is [s_t, a_t, s_{t+1}] concatenated (D+2+D dims,
        where D is state_dim: 8 for blind, 15 for labeled).

        Returns:
            s: (B, D), a: (B, 2), s_next: (B, D), r: (B,), context: (B, K, 2*D+2)
        """
        if rng is None:
            rng = np.random.default_rng()

        s_batch, a_batch, snext_batch, r_batch, ctx_batch = [], [], [], [], []
        for _ in range(batch_size):
            ep = rng.integers(self.n_episodes)
            T = self.lengths[ep]

            # Target timestep.
            target_t = rng.integers(T)

            # Context timesteps: K random transitions from same episode,
            # excluding the target timestep. If the target appeared in
            # context, the model could learn to copy s' from context instead
            # of learning the dynamics function — a trivial shortcut that
            # would produce perfect training loss but zero generalization.
            if T > 1:
                # Build pool of all timesteps except target_t.
                pool = np.concatenate([np.arange(target_t), np.arange(target_t + 1, T)])
                ctx_ts = rng.choice(pool, size=K, replace=True)
            else:
                # Edge case: episode has only 1 transition. Repeating it K
                # times is a graceful degradation — the mean-pool encoder
                # produces the same z regardless of repetition count (mean of
                # identical vectors = that vector). In practice episodes are
                # 200-600 steps, so this path almost never fires.
                ctx_ts = np.zeros(K, dtype=int)

            s_batch.append(self.states[ep][target_t])
            a_batch.append(self.actions[ep][target_t])
            snext_batch.append(self.states[ep][target_t + 1])
            r_batch.append(self.rewards[ep][target_t])

            # Build context: each row is [s_t, a_t, s_{t+1}] = 2*D+2 dims.
            # A complete (s, a, s') transition is the minimal unit that reveals
            # physics: observing how state changes under an action constrains
            # the dynamics parameters. The encoder needs all three — s alone
            # doesn't reveal dynamics, and (s, a) without s' gives no outcome.
            ctx_rows = []
            for ct in ctx_ts:
                row = np.concatenate([
                    self.states[ep][ct],       # D (state_dim)
                    self.actions[ep][ct],       # 2
                    self.states[ep][ct + 1],    # D (state_dim)
                ])
                ctx_rows.append(row)
            ctx_batch.append(np.stack(ctx_rows))

        s = torch.from_numpy(np.stack(s_batch))
        a = torch.from_numpy(np.stack(a_batch))
        s_next = torch.from_numpy(np.stack(snext_batch))
        r = torch.from_numpy(np.array(r_batch))
        context = torch.from_numpy(np.stack(ctx_batch))

        # Delta mode — see sample_transitions() for full rationale.
        if self.prediction_target == "delta":
            s_next = s_next - s

        return s, a, s_next, r, context

    def sample_sequences(
        self, batch_size: int, seq_len: int, rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample contiguous subsequences for recurrent models (GRU, RSSM).

        Picks random episodes with length >= seq_len, then random start positions.

        Returns:
            states: (B, seq_len+1, 15), actions: (B, seq_len, 2), rewards: (B, seq_len)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Filter to episodes long enough for the requested sequence length.
        # Recurrent models (GRU, RSSM) need contiguous subsequences to
        # propagate hidden state — unlike transition/context samplers which
        # draw independent timesteps and don't need temporal continuity.
        valid_eps = [i for i in range(self.n_episodes) if self.lengths[i] >= seq_len]
        if not valid_eps:
            raise ValueError(
                f"No episodes in '{self.split}' split have length >= {seq_len}. "
                f"Max episode length: {max(self.lengths) if self.lengths else 0}. "
                f"Use a shorter seq_len or add longer episodes."
            )

        s_batch, a_batch, r_batch = [], [], []
        for _ in range(batch_size):
            ep = rng.choice(valid_eps)
            start = rng.integers(self.lengths[ep] - seq_len + 1)
            s_batch.append(self.states[ep][start:start + seq_len + 1])
            a_batch.append(self.actions[ep][start:start + seq_len])
            r_batch.append(self.rewards[ep][start:start + seq_len])

        states = torch.from_numpy(np.stack(s_batch))
        actions = torch.from_numpy(np.stack(a_batch))
        rewards = torch.from_numpy(np.stack(r_batch))

        return states, actions, rewards

    def sample_sequences_with_context(
        self, batch_size: int, seq_len: int, K: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample contiguous subsequences with K context transitions per episode.

        For multi-step rollout training of context-conditioned models. Each
        sample is a contiguous subsequence plus K context transitions from
        the same episode. Context is drawn from timesteps outside the
        target subsequence to prevent the model from copying answers.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Length of each sequence (number of actions / transitions).
            K: Number of context transitions per sequence.
            rng: NumPy random generator.

        Returns:
            states:  (B, seq_len+1, D)    — contiguous state sequence
            actions: (B, seq_len, 2)      — contiguous action sequence
            context: (B, K, 2*D+2)        — context transitions [s, a, s']
            rewards: (B, seq_len)         — rewards (for completeness)
        """
        if rng is None:
            rng = np.random.default_rng()

        valid_eps = [i for i in range(self.n_episodes) if self.lengths[i] >= seq_len]
        if not valid_eps:
            raise ValueError(
                f"No episodes in '{self.split}' split have length >= {seq_len}. "
                f"Max episode length: {max(self.lengths) if self.lengths else 0}."
            )

        s_batch, a_batch, r_batch, ctx_batch = [], [], [], []
        for _ in range(batch_size):
            ep = rng.choice(valid_eps)
            T = self.lengths[ep]
            start = rng.integers(T - seq_len + 1)

            s_batch.append(self.states[ep][start:start + seq_len + 1])
            a_batch.append(self.actions[ep][start:start + seq_len])
            r_batch.append(self.rewards[ep][start:start + seq_len])

            # Context: K transitions from outside the target subsequence.
            # The target occupies timesteps [start, start+seq_len). Build
            # a pool of all other valid timesteps in the episode.
            target_set = set(range(start, start + seq_len))
            pool = np.array([t for t in range(T) if t not in target_set])
            if len(pool) > 0:
                ctx_ts = rng.choice(pool, size=K, replace=True)
            else:
                # Edge case: subsequence spans entire episode. Fall back to
                # sampling from the subsequence itself (with replacement).
                ctx_ts = rng.choice(np.arange(start, start + seq_len), size=K, replace=True)

            ctx_rows = []
            for ct in ctx_ts:
                row = np.concatenate([
                    self.states[ep][ct],       # 15
                    self.actions[ep][ct],       # 2
                    self.states[ep][ct + 1],    # 15
                ])
                ctx_rows.append(row)
            ctx_batch.append(np.stack(ctx_rows))

        states = torch.from_numpy(np.stack(s_batch))
        actions = torch.from_numpy(np.stack(a_batch))
        rewards = torch.from_numpy(np.stack(r_batch))
        context = torch.from_numpy(np.stack(ctx_batch))

        return states, actions, context, rewards


class WMIterableDataset(torch.utils.data.IterableDataset):
    """Wraps EpisodeDataset for use with PyTorch DataLoader.

    Yields individual samples in an infinite loop. DataLoader handles
    batching (batch_size) and parallel preparation (num_workers).
    Uses Pattern 1 from the DataLoader design research: infinite __iter__
    wrapping existing sampling methods.

    Why an IterableDataset instead of a map-style Dataset?
    EpisodeDataset samples transitions randomly (not by index) — episodes
    are drawn uniformly, then timesteps within each episode. This maps
    naturally to an infinite yield loop rather than __getitem__(idx).
    IterableDataset also avoids the overhead of a Sampler and lets
    DataLoader's num_workers prefetch batches in parallel worker processes.

    Modes match EpisodeDataset sampling methods:
      - "context": single-step → (s, a, target, r, ctx)
      - "transitions": single-step → (s, a, target, r)
      - "sequence_context": multi-step → (states, actions, ctx, rewards)
      - "sequences": multi-step → (states, actions, rewards)
    """

    def __init__(
        self,
        dataset: "EpisodeDataset",
        mode: str = "context",
        context_k: int | None = None,
        seq_len: int | None = None,
        seed: int | None = None,
    ):
        self.dataset = dataset
        self.mode = mode
        self.context_k = context_k
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        # Per-worker seeding to avoid duplicate batches across workers.
        # When num_workers > 0, each worker gets a copy of this IterableDataset
        # and calls __iter__ independently. Without distinct seeds, every worker
        # would produce identical sample sequences — wasting parallelism and
        # reducing effective batch diversity. Adding worker_id to the base seed
        # gives each worker a unique, reproducible stream.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and self.seed is not None:
            worker_seed = self.seed + worker_info.id
        elif self.seed is not None:
            worker_seed = self.seed
        else:
            worker_seed = None
        rng = np.random.default_rng(worker_seed)

        # Infinite yield loop: DataLoader pulls exactly batch_size samples
        # per batch, then stops (it doesn't drain the iterator). This means
        # the training loop controls epoch length via steps_per_epoch, not
        # dataset size — consistent with the existing random-sampling design.
        while True:
            if self.mode == "context":
                s, a, target, r, ctx = self.dataset.sample_with_context(
                    1, self.context_k, rng=rng)
                yield s[0], a[0], target[0], r[0], ctx[0]
            elif self.mode == "transitions":
                s, a, target, r = self.dataset.sample_transitions(1, rng=rng)
                yield s[0], a[0], target[0], r[0]
            elif self.mode == "sequence_context":
                states, actions, ctx, rewards = self.dataset.sample_sequences_with_context(
                    1, self.seq_len, self.context_k, rng=rng)
                yield states[0], actions[0], ctx[0], rewards[0]
            elif self.mode == "sequences":
                states, actions, rewards = self.dataset.sample_sequences(
                    1, self.seq_len, rng=rng)
                yield states[0], actions[0], rewards[0]
