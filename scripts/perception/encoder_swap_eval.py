#!/usr/bin/env python3
"""Encoder swap experiment: 2×2 eval with crossed encoder/MLP-head weights.

Tests whether the frozen→fine-tune improvement (84%→95%) comes from
the encoder or the MLP policy head by swapping components between agents.

Phase 1 (prepare): Load both agents, swap weight components, save 4
  checkpoint directories with proper config.json + model.zip + vec_normalize.pkl.
Phase 2 (eval): Call eval_agent.py on each checkpoint dir in parallel.

Each condition dir is a valid agent checkpoint — you can re-eval it,
run cross-config comparison, compute behavioral metrics, etc.

Usage:
    python scripts/perception/encoder_swap_eval.py \
        --frozen-dir .../frozen-lowlr/s42 \
        --finetune-dir .../finetune-lowlr/s42 \
        --episodes 100 \
        --output-dir /workspace/latent-world-physics/vsr-tmp/encoder-swap-s42

Output structure:
    output-dir/
      frozen_enc+frozen_mlp/        # each is a valid checkpoint dir
        config.json
        best/model.zip
        vec_normalize.pkl
        eval_episodes.csv           # from eval_agent.py
        eval_results.json
        plots/
      finetune_enc+finetune_mlp/
      frozen_enc+finetune_mlp/
      finetune_enc+frozen_mlp/
      encoder_swap_summary.json     # combined results
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from lwp.agents.model_loading import load_model
from lwp.agents.eval_utils import resolve_model_path


def split_policy_state(model) -> tuple[dict, dict]:
    """Split a loaded model's policy state_dict into encoder and MLP head."""
    state = model.policy.state_dict()
    prefix = "features_extractor."
    enc = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    mlp = {k: v for k, v in state.items() if not k.startswith(prefix)}
    return enc, mlp


def combine_state_dict(enc: dict, mlp: dict) -> dict:
    """Recombine encoder and MLP head into a full policy state_dict."""
    combined = {f"features_extractor.{k}": v for k, v in enc.items()}
    combined.update(mlp)
    return combined


def prepare_checkpoint_dir(
    cond_dir: Path,
    model,
    enc_state: dict,
    mlp_state: dict,
    source_ckpt_dir: str,
):
    """Create a valid checkpoint dir with swapped weights.

    Swaps weights into the model, saves via model.save(), copies
    config.json and vec_normalize.pkl from the source agent.
    """
    cond_dir.mkdir(parents=True, exist_ok=True)
    best_dir = cond_dir / "best"
    best_dir.mkdir(exist_ok=True)

    # Swap weights and save
    state = combine_state_dict(enc_state, mlp_state)
    model.policy.load_state_dict(state)
    model.save(str(best_dir / "model.zip"))

    # Copy config.json from source agent
    src_config = Path(source_ckpt_dir) / "config.json"
    if src_config.exists():
        shutil.copy2(str(src_config), str(cond_dir / "config.json"))

    # Copy vec_normalize.pkl from source agent
    src_vecnorm = Path(source_ckpt_dir) / "vec_normalize.pkl"
    if src_vecnorm.exists():
        shutil.copy2(str(src_vecnorm), str(cond_dir / "vec_normalize.pkl"))


def run_eval(cond_dir: str, episodes: int, seed: int, profiles: str | None = None) -> dict:
    """Call eval_agent.py as a subprocess. Returns parsed summary."""
    # Auto-detect profile from config.json if not explicitly specified
    if not profiles:
        config_path = Path(cond_dir) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            profiles = config.get("profile")

    cmd = [
        sys.executable, "scripts/agents/eval_agent.py",
        "--checkpoint-dir", cond_dir,
        "--model", "best/model.zip",
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--output-dir", cond_dir,
    ]
    if profiles:
        cmd.extend(["--profiles", profiles])
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    name = Path(cond_dir).name
    if result.returncode != 0:
        print(f"  FAILED: {name}\n{result.stderr[-500:]}")
        return {"name": name, "error": result.stderr[-500:], "elapsed": elapsed}

    # Read the results JSON that eval_agent.py wrote
    results_path = Path(cond_dir) / "eval_results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
            summary = data.get("overall", {})
    else:
        summary = {}

    return {"name": name, "summary": summary, "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(
        description="Encoder swap 2×2 experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--frozen-dir", required=True,
                        help="Frozen agent seed dir (e.g. .../frozen-lowlr/s42)")
    parser.add_argument("--finetune-dir", required=True,
                        help="Fine-tune agent seed dir (e.g. .../finetune-lowlr/s42)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Eval episodes per condition (default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Eval env seed (default: 0)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for all condition checkpoints + results")
    parser.add_argument("--profiles", default=None,
                        help="Eval profiles (e.g. 'gym-default' or 'easy,medium'). "
                             "Default: use whatever eval_agent.py defaults to.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Max parallel eval_agent.py processes (default: 4)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip checkpoint preparation (reuse existing dirs)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: Prepare checkpoint dirs ===
    condition_names = [
        "frozen_enc+frozen_mlp",
        "finetune_enc+finetune_mlp",
        "frozen_enc+finetune_mlp",
        "finetune_enc+frozen_mlp",
    ]

    if not args.skip_prepare:
        print("=== Phase 1: Preparing checkpoint directories")

        print("  Loading frozen agent...")
        model_frozen, config_frozen = load_model(args.frozen_dir, device="cpu")
        print("  Loading fine-tuned agent...")
        model_finetune, config_finetune = load_model(args.finetune_dir, device="cpu")

        frozen_enc, frozen_mlp = split_policy_state(model_frozen)
        finetune_enc, finetune_mlp = split_policy_state(model_finetune)

        enc_params = sum(v.numel() for v in frozen_enc.values())
        mlp_params = sum(v.numel() for v in frozen_mlp.values())
        print(f"  Encoder: {len(frozen_enc)} tensors ({enc_params:,} params)")
        print(f"  MLP head: {len(frozen_mlp)} tensors ({mlp_params:,} params)")

        # Use the frozen model object for saving all conditions
        # (model.save() saves the full SB3 state including optimizer, etc.)
        model = model_frozen

        # (name, enc, mlp, source_ckpt_dir for config/vecnorm)
        conditions = [
            ("frozen_enc+frozen_mlp",     frozen_enc,   frozen_mlp,   args.frozen_dir),
            ("finetune_enc+finetune_mlp", finetune_enc, finetune_mlp, args.finetune_dir),
            ("frozen_enc+finetune_mlp",   frozen_enc,   finetune_mlp, args.finetune_dir),
            ("finetune_enc+frozen_mlp",   finetune_enc, frozen_mlp,   args.frozen_dir),
        ]

        for name, enc, mlp, src_dir in conditions:
            cond_dir = output_dir / name
            print(f"  Preparing {name}...")
            prepare_checkpoint_dir(cond_dir, model, enc, mlp, src_dir)
            print(f"    Saved to {cond_dir}")

        del model, model_frozen, model_finetune
        print("  Preparation complete.\n")
    else:
        print("=== Skipping preparation (--skip-prepare)")

    # === Phase 2: Run eval_agent.py on each condition ===
    n_workers = min(args.n_workers, len(condition_names))
    print(f"=== Phase 2: Evaluating {len(condition_names)} conditions "
          f"({n_workers} workers, {args.episodes} episodes each)")

    t0 = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for name in condition_names:
            cond_dir = str(output_dir / name)
            future = executor.submit(run_eval, cond_dir, args.episodes, args.seed, args.profiles)
            futures[future] = name
            print(f"  Dispatched: {name}")

        for future in as_completed(futures):
            r = future.result()
            name = r["name"]
            if "error" in r:
                print(f"  FAILED: {name} ({r['elapsed']:.0f}s)")
                results[name] = {"error": r["error"]}
            else:
                summary = r["summary"]
                landed = summary.get("landed_pct", 0)
                crashed = summary.get("crashed_pct", 0)
                reward = summary.get("mean_reward", 0)
                results[name] = {
                    "landed_pct": landed,
                    "crashed_pct": crashed,
                    "mean_reward": reward,
                }
                print(f"  Done: {name}  Landed: {landed:.1f}%  "
                      f"Crashed: {crashed:.1f}%  Reward: {reward:.1f}  ({r['elapsed']:.0f}s)")

    total_elapsed = time.time() - t0
    print(f"\nAll conditions complete in {total_elapsed:.0f}s")

    # === Summary ===
    print("\n=== Summary")
    print(f"  {'Condition':40s}  {'Landed%':>8s}  {'Reward':>8s}")
    print("  " + "-" * 60)
    for name in condition_names:
        if name in results and "error" not in results[name]:
            r = results[name]
            print(f"  {name:40s}  {r['landed_pct']:7.1f}%  {r['mean_reward']:8.1f}")
        else:
            print(f"  {name:40s}  {'FAILED':>8s}  {'':>8s}")

    # Interpretation
    if all(name in results and "error" not in results[name] for name in condition_names):
        print("\n=== Interpretation")
        base_finetune = results["finetune_enc+finetune_mlp"]["landed_pct"]
        cross_ft_mlp = results["frozen_enc+finetune_mlp"]["landed_pct"]
        cross_fr_mlp = results["finetune_enc+frozen_mlp"]["landed_pct"]
        print(f"  If MLP head drives improvement: frozen_enc+finetune_mlp ≈ {base_finetune:.1f}% (got {cross_ft_mlp:.1f}%)")
        print(f"  If encoder drives improvement:  finetune_enc+frozen_mlp ≈ {base_finetune:.1f}% (got {cross_fr_mlp:.1f}%)")

    # Save combined summary
    summary_file = output_dir / "encoder_swap_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {summary_file}")
    print(f"  Per-condition data in {output_dir}/<condition>/")


if __name__ == "__main__":
    main()
