#!/usr/bin/env python3
"""Scan checkpoints for old module paths that need migration.

Reads SB3 model.zip files and torch .pt checkpoints, reports any
pickled references to old module paths (lunar_lander.src.*, models.*,
rl_common.*, world_models.*).

Usage:
    python scripts/utils/scan_old_checkpoints.py /path/to/checkpoints
    python scripts/utils/scan_old_checkpoints.py /path/to/specific/model.zip
    python scripts/utils/scan_old_checkpoints.py /path/to/runs --recursive
"""

import argparse
import struct
import zipfile
from pathlib import Path


# Old module paths that indicate a checkpoint needs migration.
OLD_PREFIXES = [
    b"lunar_lander.src.",
    b"lunar_lander.scripts.",
    b"rl_common.",
    b"world_models.",
    # Bare wm-ladder imports (these are trickier to detect since "models"
    # is generic, but "models.mlp" or "models.gru" are specific enough)
    b"models.mlp",
    b"models.gru",
    b"models.rssm",
    b"models.linear",
    b"models.pixel",
    b"models.factory",
    b"models.base",
    b"models.copy",
    b"training.loop",
    b"training.losses",
    b"training.callbacks",
    b"data.loader",
    b"data.normalization",
    b"evaluation.metrics",
    b"utils.config",
    b"utils.checkpoint",
    b"viz.dream",
]


def scan_bytes(data: bytes, filepath: str) -> list[dict]:
    """Scan raw bytes for old module path references.

    Pickle embeds module paths as UTF-8 strings. We scan the raw bytes
    for known old prefixes. This is a heuristic (could false-positive on
    comments or strings) but in practice pickle opcodes place module paths
    in predictable positions.
    """
    findings = []
    for prefix in OLD_PREFIXES:
        start = 0
        while True:
            idx = data.find(prefix, start)
            if idx == -1:
                break

            # Extract the full module.class string (read until non-identifier char)
            end = idx
            while end < len(data) and (data[end:end+1].isalnum() or data[end:end+1] in (b".", b"_")):
                end += 1
            module_path = data[idx:end].decode("utf-8", errors="replace")

            findings.append({
                "file": filepath,
                "offset": idx,
                "module_path": module_path,
            })
            start = end

    return findings


def scan_zip(zip_path: Path) -> list[dict]:
    """Scan an SB3 model.zip for old module paths.

    SB3 model.zip contains:
    - policy.pth (PyTorch state dict, pickled)
    - policy.optimizer.pth (optimizer state, pickled)
    - data (pickle of hyperparams, includes class references)
    """
    findings = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                with zf.open(name) as f:
                    data = f.read()
                for finding in scan_bytes(data, f"{zip_path}:{name}"):
                    findings.append(finding)
    except (zipfile.BadZipFile, Exception) as e:
        findings.append({
            "file": str(zip_path),
            "offset": -1,
            "module_path": f"ERROR: {e}",
        })
    return findings


def scan_pt(pt_path: Path) -> list[dict]:
    """Scan a .pt checkpoint for old module paths."""
    data = pt_path.read_bytes()
    return scan_bytes(data, str(pt_path))


def scan_path(path: Path, recursive: bool) -> list[dict]:
    """Scan a file or directory for old module paths in checkpoints."""
    findings = []

    if path.is_file():
        if path.suffix == ".zip":
            findings.extend(scan_zip(path))
        elif path.suffix == ".pt":
            findings.extend(scan_pt(path))
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        for f in sorted(path.glob(pattern)):
            if f.suffix == ".zip":
                findings.extend(scan_zip(f))
            elif f.suffix == ".pt":
                findings.extend(scan_pt(f))

    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Scan checkpoints for old module paths that need migration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=Path,
                        help="File or directory to scan")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories (default: top-level only)")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: {args.path} does not exist")
        return

    findings = scan_path(args.path, args.recursive)

    if not findings:
        print("No old module paths found. Checkpoints are clean.")
        return

    # Group by file
    by_file = {}
    for f in findings:
        by_file.setdefault(f["file"], []).append(f)

    print(f"Found {len(findings)} old module references in {len(by_file)} files:\n")

    for filepath, file_findings in sorted(by_file.items()):
        print(f"  {filepath}")
        # Deduplicate module paths for this file
        unique_paths = sorted(set(f["module_path"] for f in file_findings))
        for mp in unique_paths:
            print(f"    {mp}")
        print()

    # Summary: which old prefixes were found
    all_paths = sorted(set(f["module_path"] for f in findings))
    print(f"Unique old module paths ({len(all_paths)}):")
    for mp in all_paths:
        print(f"  {mp}")


if __name__ == "__main__":
    main()
