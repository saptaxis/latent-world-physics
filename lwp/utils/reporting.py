"""Post-training reporting utilities.

Generate markdown tables and summary reports from evaluation results.
"""
from __future__ import annotations

from pathlib import Path


def format_per_dim_table(per_dim_mse: dict[str, float]) -> str:
    """Format per-dimension MSE as a markdown table."""
    lines = ["| Dimension | MSE |", "|-----------|-----|"]
    for name, mse in per_dim_mse.items():
        lines.append(f"| {name} | {mse:.6f} |")
    mean = sum(per_dim_mse.values()) / len(per_dim_mse) if per_dim_mse else 0
    lines.append(f"| **mean** | **{mean:.6f}** |")
    return "\n".join(lines)


def format_horizon_table(horizon_data: dict) -> str:
    """Format horizon error data as a markdown table."""
    lines = ["| Horizon | Mean MSE |", "|---------|----------|"]
    for h in sorted(horizon_data.keys()):
        val = horizon_data[h]
        if isinstance(val, dict):
            mean = sum(val.values()) / len(val) if val else 0
        else:
            mean = val
        lines.append(f"| h={h} | {mean:.6f} |")
    return "\n".join(lines)


def generate_eval_report(run_name: str, results: dict,
                         output_path: str | None = None) -> str:
    """Generate a markdown evaluation report."""
    sections = [f"# Evaluation Report: {run_name}\n"]

    if "per_dim_mse" in results:
        sections.append("## Per-Dimension MSE (1-step)\n")
        sections.append(format_per_dim_table(results["per_dim_mse"]))
        sections.append("")

    if "horizon_mean_mse" in results:
        sections.append("## Horizon Error Curve\n")
        sections.append(format_horizon_table(results["horizon_mean_mse"]))
        sections.append("")

    if "divergence_exponent" in results or "horizon_to_failure" in results:
        sections.append("## Summary Metrics\n")
        if "divergence_exponent" in results:
            lam = results["divergence_exponent"]
            sections.append(f"- **Divergence exponent (lambda):** {lam:.4f}")
        if "horizon_to_failure" in results:
            htf = results["horizon_to_failure"]
            sections.append(f"- **Horizon to failure:** {htf}")
        sections.append("")

    if "cumul_horizon_mean_mse" in results:
        sections.append("## Cumulative Trajectory MSE (Eval B')\n")
        sections.append("Average MSE across all steps 1..h (not just endpoint).\n")
        sections.append(format_horizon_table(results["cumul_horizon_mean_mse"]))
        sections.append("")

    if "horizon_curves" in results:
        sections.append("## Per-Dimension Horizon Curves\n")
        for h in sorted(results["horizon_curves"].keys()):
            sections.append(f"### h={h}\n")
            sections.append(format_per_dim_table(results["horizon_curves"][h]))
            sections.append("")

    report = "\n".join(sections)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(report)

    return report
