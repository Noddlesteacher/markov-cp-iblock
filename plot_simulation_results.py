"""Plot reliability and scaled set-size curves from simulation summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "original": "Original i-block CP",
    "permutation_count": "Permutation-count (D!) weighted",
    "iblock_count": "I-block-count (D) weighted",
}


def plot_reliability(
    summary_df: pd.DataFrame,
    simulation_name: str,
    output_dir: Path,
) -> list[Path]:
    """Create reliability plots for one simulation case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    case_df = summary_df[summary_df["simulation"] == simulation_name]

    for method in sorted(case_df["method"].unique()):
        method_df = case_df[case_df["method"] == method]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)

        for horizon in sorted(method_df["horizon"].unique()):
            horizon_df = method_df[method_df["horizon"] == horizon].sort_values(
                "target_coverage"
            )
            ax.plot(
                horizon_df["target_coverage"],
                horizon_df["empirical_coverage"],
                marker="o",
                label=f"h={horizon}",
            )

        ax.set_title(f"{simulation_name}: {METHOD_LABELS.get(method, method)}")
        ax.set_xlabel("Target coverage (1 - alpha)")
        ax.set_ylabel("Empirical coverage")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Horizon")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"{simulation_name}_reliability_{method}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths.append(path)
        print(f"Saved {path}")

    return saved_paths


def plot_scaled_set_size(
    summary_df: pd.DataFrame,
    simulation_name: str,
    output_dir: Path,
) -> list[Path]:
    """Create scaled prediction-set-size plots for one simulation case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    case_df = summary_df[summary_df["simulation"] == simulation_name]

    for method in sorted(case_df["method"].unique()):
        method_df = case_df[case_df["method"] == method]
        fig, ax = plt.subplots(figsize=(6, 5))

        for horizon in sorted(method_df["horizon"].unique()):
            horizon_df = method_df[method_df["horizon"] == horizon].sort_values(
                "target_coverage"
            )
            ax.plot(
                horizon_df["target_coverage"],
                horizon_df["mean_scaled_set_size"],
                marker="o",
                label=f"h={horizon}",
            )

        ax.set_title(f"{simulation_name}: {METHOD_LABELS.get(method, method)}")
        ax.set_xlabel("Target coverage (1 - alpha)")
        ax.set_ylabel("Mean scaled set size")
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.legend(title="Horizon")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        path = output_dir / f"{simulation_name}_scaled_set_size_{method}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        saved_paths.append(path)
        print(f"Saved {path}")

    return saved_paths


def plot_all(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Plot all simulation cases present in a summary table."""
    saved_paths: list[Path] = []
    for simulation_name in sorted(summary_df["simulation"].unique()):
        saved_paths.extend(plot_reliability(summary_df, simulation_name, output_dir))
        saved_paths.extend(plot_scaled_set_size(summary_df, simulation_name, output_dir))
    return saved_paths


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("simulation_results/sim1_summary.csv"),
        help="summary CSV created by simulation_study.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for PNG plots; defaults to the summary CSV directory",
    )
    return parser


def main() -> None:
    """Read a summary CSV and save reliability and set-size plots."""
    args = build_parser().parse_args()
    summary_df = pd.read_csv(args.summary)
    output_dir = args.output_dir if args.output_dir is not None else args.summary.parent
    plot_all(summary_df, output_dir)


if __name__ == "__main__":
    main()
