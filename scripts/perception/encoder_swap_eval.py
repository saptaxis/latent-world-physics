#!/usr/bin/env python3
"""Encoder swap experiment: 2×2 eval with crossed encoder/MLP-head weights.

Tests whether the frozen→fine-tune improvement (84%→95%) comes from
the encoder or the MLP policy head by swapping components between agents.

Uses load_model() from lwp.agents.model_loading which handles the
compat shim, VecNormalize, and VecFrameStack automatically.

Usage:
    python scripts/perception/encoder_swap_eval.py \
        --frozen-dir .../frozen-lowlr/s42 \
        --finetune-dir .../finetune-lowlr/s42 \
        --episodes 10 \
        --output-dir ~/vsr-tmp/encoder-swap-s42
"""
import argparse
import json
from pathlib import Path

from lwp.agents.model_loading import load_model
from lwp.agents.eval_utils import evaluate_agent, make_env_factory, resolve_model_path, resolve_vec_normalize_path


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


def main():
    parser = argparse.ArgumentParser(description="Encoder swap 2×2 experiment")
    parser.add_argument("--frozen-dir", required=True,
                        help="Frozen agent seed dir (e.g. .../frozen-lowlr/s42)")
    parser.add_argument("--finetune-dir", required=True,
                        help="Fine-tune agent seed dir (e.g. .../finetune-lowlr/s42)")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Eval episodes per condition (default: 100)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for results JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load both models (compat shim handles old pickle references)
    print("Loading frozen agent...")
    model_frozen, config_frozen = load_model(args.frozen_dir, device="cpu")
    print("Loading fine-tuned agent...")
    model_finetune, config_finetune = load_model(args.finetune_dir, device="cpu")

    # Extract components
    frozen_enc, frozen_mlp = split_policy_state(model_frozen)
    finetune_enc, finetune_mlp = split_policy_state(model_finetune)
    print(f"Encoder params: {len(frozen_enc)}, MLP params: {len(frozen_mlp)}")

    # We'll reuse the frozen model object for all conditions —
    # just swap the policy weights before each eval.
    model = model_frozen

    # For cross conditions, we use the VecNormalize stats from the agent
    # whose MLP head we're using (the MLP was trained with those stats).
    # This assumes both agents were trained on the same env (gym-default)
    # with similar observation distributions. If VecNormalize stats differ
    # significantly between frozen and fine-tune, cross results may be noisy.
    #
    # 4 conditions: (encoder, mlp_head, config for env, checkpoint_dir for vecnorm)
    conditions = [
        ("frozen_enc+frozen_mlp",     frozen_enc,   frozen_mlp,   config_frozen,   args.frozen_dir),
        ("finetune_enc+finetune_mlp", finetune_enc, finetune_mlp, config_finetune, args.finetune_dir),
        ("frozen_enc+finetune_mlp",   frozen_enc,   finetune_mlp, config_finetune, args.finetune_dir),
        ("finetune_enc+frozen_mlp",   finetune_enc, frozen_mlp,   config_frozen,   args.frozen_dir),
    ]

    results = {}
    for name, enc, mlp, config, ckpt_dir in conditions:
        print(f"\n=== {name} ({args.episodes} episodes)")

        # Swap weights
        state = combine_state_dict(enc, mlp)
        model.policy.load_state_dict(state)
        model.policy.eval()

        # Build env and eval using the standard pipeline
        # IMPORTANT: pass profile from config so agent evals on correct physics
        env_fn = make_env_factory(
            variant=config["variant"],
            frame_size=config.get("frame_size", 84),
            n_rays=config.get("n_rays", 7),
            history_k=config.get("history_k", 8),
            profile=config.get("profile"),
        )
        n_stack = config.get("n_stack", 0)
        model_path = resolve_model_path(ckpt_dir)
        vec_norm_path = resolve_vec_normalize_path(ckpt_dir, model_path)

        result = evaluate_agent(
            model, env_fn,
            n_episodes=args.episodes,
            seed=42,
            vec_normalize_path=vec_norm_path,
            deterministic=True,
            n_stack=n_stack,
        )

        summary = result["summary"]
        landed = summary.get("landed_pct", 0)
        crashed = summary.get("crashed_pct", 0)
        reward = summary.get("mean_reward", 0)
        results[name] = {"landed_pct": landed, "crashed_pct": crashed, "mean_reward": reward}
        print(f"  Landed: {landed}%  Crashed: {crashed}%  Reward: {reward:.1f}")

    # Summary table
    print("\n=== Summary")
    print(f"  {'Condition':40s}  {'Landed%':>8s}  {'Reward':>8s}")
    print("  " + "-" * 60)
    for name, r in results.items():
        print(f"  {name:40s}  {r['landed_pct']:7.1f}%  {r['mean_reward']:8.1f}")

    # Interpretation
    print("\n=== Interpretation")
    base_frozen = results["frozen_enc+frozen_mlp"]["landed_pct"]
    base_finetune = results["finetune_enc+finetune_mlp"]["landed_pct"]
    cross_ft_mlp = results["frozen_enc+finetune_mlp"]["landed_pct"]
    cross_fr_mlp = results["finetune_enc+frozen_mlp"]["landed_pct"]
    print(f"  If MLP head drives improvement: frozen_enc+finetune_mlp ≈ {base_finetune}% (got {cross_ft_mlp}%)")
    print(f"  If encoder drives improvement:  finetune_enc+frozen_mlp ≈ {base_finetune}% (got {cross_fr_mlp}%)")

    # Save
    output_file = output_dir / "encoder_swap_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_file}")


if __name__ == "__main__":
    main()
