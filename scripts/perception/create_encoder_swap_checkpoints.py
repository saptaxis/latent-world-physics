#!/usr/bin/env python3
"""Create encoder swap checkpoint directories.

Loads two visual RL agents (e.g. frozen encoder + fine-tuned encoder),
extracts encoder and MLP head state_dicts, and creates 4 checkpoint
directories with all crossed combinations of (encoder × MLP head).

Each output dir is a valid agent checkpoint (config.json + best/model.zip
+ vec_normalize.pkl) that can be evaluated with run_eval_pipeline.py,
eval_agent.py, cross-config comparison, etc.

Supports multiple seeds — runs the swap for each seed found in both
agent directories.

Usage:
    # Single seed
    python scripts/perception/create_encoder_swap_checkpoints.py \
        --frozen-dir .../frozen-lowlr/s42 \
        --finetune-dir .../finetune-lowlr/s42 \
        --output-dir ~/vsr-tmp/encoder-swap/s42

    # All matching seeds (auto-discovers s42, s123, s456, ...)
    python scripts/perception/create_encoder_swap_checkpoints.py \
        --frozen-parent .../frozen-lowlr \
        --finetune-parent .../finetune-lowlr \
        --output-dir ~/vsr-tmp/encoder-swap

Output structure (single seed):
    output-dir/
      frozen_enc+frozen_mlp/config.json, best/model.zip, vec_normalize.pkl
      finetune_enc+finetune_mlp/...
      frozen_enc+finetune_mlp/...
      finetune_enc+frozen_mlp/...

Output structure (multi-seed):
    output-dir/
      s42/
        frozen_enc+frozen_mlp/...
        finetune_enc+finetune_mlp/...
        ...
      s123/
        ...
"""
import argparse
import shutil
from pathlib import Path

from lwp.agents.model_loading import load_model


CONDITION_NAMES = [
    "frozen_enc+frozen_mlp",
    "finetune_enc+finetune_mlp",
    "frozen_enc+finetune_mlp",
    "finetune_enc+frozen_mlp",
]


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
    """Create a valid checkpoint dir with swapped weights."""
    cond_dir.mkdir(parents=True, exist_ok=True)
    best_dir = cond_dir / "best"
    best_dir.mkdir(exist_ok=True)

    state = combine_state_dict(enc_state, mlp_state)
    model.policy.load_state_dict(state)
    model.save(str(best_dir / "model.zip"))

    src_config = Path(source_ckpt_dir) / "config.json"
    if src_config.exists():
        shutil.copy2(str(src_config), str(cond_dir / "config.json"))

    src_vecnorm = Path(source_ckpt_dir) / "vec_normalize.pkl"
    if src_vecnorm.exists():
        shutil.copy2(str(src_vecnorm), str(cond_dir / "vec_normalize.pkl"))


def find_matching_seeds(frozen_parent: str, finetune_parent: str) -> list[str]:
    """Find seed directories (s42, s123, ...) present in both parents."""
    frozen_seeds = {d.name for d in Path(frozen_parent).iterdir()
                    if d.is_dir() and d.name.startswith("s")}
    finetune_seeds = {d.name for d in Path(finetune_parent).iterdir()
                      if d.is_dir() and d.name.startswith("s")}
    common = sorted(frozen_seeds & finetune_seeds,
                    key=lambda s: int(s[1:]))
    return common


def create_swap_checkpoints(frozen_dir: str, finetune_dir: str, output_dir: Path):
    """Create 4 swapped checkpoint dirs for one seed pair."""
    print(f"  Loading frozen: {Path(frozen_dir).name}")
    model_frozen, _ = load_model(frozen_dir, device="cpu")
    print(f"  Loading finetune: {Path(finetune_dir).name}")
    model_finetune, _ = load_model(finetune_dir, device="cpu")

    frozen_enc, frozen_mlp = split_policy_state(model_frozen)
    finetune_enc, finetune_mlp = split_policy_state(model_finetune)

    model = model_frozen

    conditions = [
        ("frozen_enc+frozen_mlp",     frozen_enc,   frozen_mlp,   frozen_dir),
        ("finetune_enc+finetune_mlp", finetune_enc, finetune_mlp, finetune_dir),
        ("frozen_enc+finetune_mlp",   frozen_enc,   finetune_mlp, finetune_dir),
        ("finetune_enc+frozen_mlp",   finetune_enc, frozen_mlp,   frozen_dir),
    ]

    for name, enc, mlp, src_dir in conditions:
        cond_dir = output_dir / name
        prepare_checkpoint_dir(cond_dir, model, enc, mlp, src_dir)
        print(f"    {name} -> {cond_dir}")

    del model, model_frozen, model_finetune


def main():
    parser = argparse.ArgumentParser(
        description="Create encoder swap checkpoint directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Single seed mode
    parser.add_argument("--frozen-dir",
                        help="Frozen agent seed dir (e.g. .../frozen-lowlr/s42)")
    parser.add_argument("--finetune-dir",
                        help="Fine-tune agent seed dir (e.g. .../finetune-lowlr/s42)")

    # Multi-seed mode
    parser.add_argument("--frozen-parent",
                        help="Frozen agent parent dir (contains s42/, s123/, ...)")
    parser.add_argument("--finetune-parent",
                        help="Fine-tune agent parent dir (contains s42/, s123/, ...)")

    parser.add_argument("--output-dir", required=True,
                        help="Output directory for swapped checkpoints")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Single seed mode
    if args.frozen_dir and args.finetune_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Creating swap checkpoints (single seed)")
        create_swap_checkpoints(args.frozen_dir, args.finetune_dir, output_dir)
        print(f"\nDone. 4 checkpoints in {output_dir}/")

    # Multi-seed mode
    elif args.frozen_parent and args.finetune_parent:
        seeds = find_matching_seeds(args.frozen_parent, args.finetune_parent)
        if not seeds:
            print("No matching seeds found.")
            return
        print(f"Found {len(seeds)} matching seeds: {', '.join(seeds)}")

        for seed_name in seeds:
            seed_output = output_dir / seed_name
            seed_output.mkdir(parents=True, exist_ok=True)
            frozen_seed = str(Path(args.frozen_parent) / seed_name)
            finetune_seed = str(Path(args.finetune_parent) / seed_name)
            print(f"\n=== {seed_name}")
            create_swap_checkpoints(frozen_seed, finetune_seed, seed_output)

        print(f"\nDone. {len(seeds)} seeds × 4 conditions = {len(seeds) * 4} checkpoints in {output_dir}/")

    else:
        parser.error("Provide either --frozen-dir/--finetune-dir (single seed) "
                      "or --frozen-parent/--finetune-parent (multi-seed)")


if __name__ == "__main__":
    main()
