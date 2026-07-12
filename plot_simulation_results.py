"""Create figures and raw set-size tables from saved simulation output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "original": "Original i-block",
    "iblock_count": "I-block-count weighted",
}
PRIMARY_METHODS = ("original", "iblock_count")
AXIS_LIMITS = (0.50, 1.00)
AXIS_TICKS = np.linspace(0.50, 1.00, 6)


def output_root_from_summary(summary_path: Path) -> Path:
    """Infer simulation_results/ from an output summary path."""
    if summary_path.parent.name == "output":
        return summary_path.parent.parent
    return summary_path.parent


def _primary_case_df(summary_df: pd.DataFrame, simulation_name: str) -> pd.DataFrame:
    """Return one simulation case restricted to primary plotted methods."""
    case_df = summary_df[
        (summary_df["simulation"] == simulation_name)
        & (summary_df["method"].isin(PRIMARY_METHODS))
    ].copy()
    if case_df.empty:
        raise ValueError(f"no primary-method rows found for {simulation_name}.")
    return case_df


def plot_reliability(
    summary_df: pd.DataFrame,
    simulation_name: str,
    output_dir: Path,
) -> list[Path]:
    """Create the two-panel reliability figure for one simulation case."""
    output_dir.mkdir(parents=True, exist_ok=True)
    case_df = _primary_case_df(summary_df, simulation_name)

    fig, axes = plt.subplots(
        1,
        len(PRIMARY_METHODS),
        figsize=(8.2, 4.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(PRIMARY_METHODS) == 1:
        axes = [axes]

    for ax, method in zip(axes, PRIMARY_METHODS):
        method_df = case_df[case_df["method"] == method]
        ax.plot(
            AXIS_LIMITS,
            AXIS_LIMITS,
            color="black",
            linestyle="--",
            linewidth=1,
        )
        for horizon in sorted(method_df["horizon"].unique()):
            horizon_df = method_df[method_df["horizon"] == horizon].sort_values(
                "target_coverage"
            )
            ax.plot(
                horizon_df["target_coverage"],
                horizon_df["empirical_coverage"],
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=f"h={horizon}",
            )

        ax.set_title(METHOD_LABELS[method])
        ax.set_xlim(*AXIS_LIMITS)
        ax.set_ylim(*AXIS_LIMITS)
        ax.set_xticks(AXIS_TICKS)
        ax.set_yticks(AXIS_TICKS)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Target coverage")
        ax.legend(title="Horizon", loc="upper left", frameon=True)

    axes[0].set_ylabel("Empirical coverage")
    fig.suptitle(f"{simulation_name}: reliability", y=1.03)

    png_path = output_dir / f"{simulation_name}_reliability.png"
    pdf_path = output_dir / f"{simulation_name}_reliability.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return [png_path, pdf_path]


def build_raw_set_size_table(
    summary_df: pd.DataFrame,
    simulation_name: str,
) -> pd.DataFrame:
    """Build raw mean set-size table with no normalization by 3**h."""
    case_df = _primary_case_df(summary_df, simulation_name)
    table = case_df.pivot_table(
        index=["target_coverage", "horizon"],
        columns="method",
        values="mean_set_size",
        aggfunc="first",
    ).reset_index()
    table = table.rename(
        columns={
            "original": "original_mean_set_size",
            "iblock_count": "iblock_count_mean_set_size",
        }
    )
    return table[
        [
            "target_coverage",
            "horizon",
            "original_mean_set_size",
            "iblock_count_mean_set_size",
        ]
    ].sort_values(["target_coverage", "horizon"])


def save_raw_set_size_tables(
    summary_df: pd.DataFrame,
    simulation_name: str,
    output_dir: Path,
    table_target_coverage: float = 0.80,
) -> list[Path]:
    """Save full and target-specific raw set-size tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    full_table = build_raw_set_size_table(summary_df, simulation_name)
    full_path = output_dir / f"{simulation_name}_raw_set_size_summary.csv"
    full_table.to_csv(full_path, index=False)
    print(f"Saved {full_path}")

    target_rows = full_table[
        np.isclose(full_table["target_coverage"], table_target_coverage)
    ].copy()
    if target_rows.empty:
        raise ValueError(
            f"target coverage {table_target_coverage:.2f} is absent from summary."
        )
    target_rows = target_rows.sort_values("horizon")
    target_rows["Horizon"] = target_rows["horizon"].astype(int)
    target_rows["Original i-block"] = target_rows["original_mean_set_size"]
    target_rows["I-block-count weighted"] = target_rows[
        "iblock_count_mean_set_size"
    ]
    display = target_rows[
        ["Horizon", "Original i-block", "I-block-count weighted"]
    ].copy()

    suffix = int(round(table_target_coverage * 100))
    md_path = output_dir / f"{simulation_name}_raw_set_size_target_{suffix:03d}.md"
    tex_path = output_dir / f"{simulation_name}_raw_set_size_target_{suffix:03d}.tex"

    md_lines = [
        "| Horizon | Original i-block | I-block-count weighted |",
        "|---:|---:|---:|",
    ]
    for _, row in display.iterrows():
        md_lines.append(
            f"| {int(row['Horizon'])} | "
            f"{row['Original i-block']:.3f} | "
            f"{row['I-block-count weighted']:.3f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{rcc}",
        "\\hline",
        "Horizon & Original i-block & I-block-count weighted \\\\",
        "\\hline",
    ]
    for _, row in display.iterrows():
        tex_lines.append(
            f"{int(row['Horizon'])} & "
            f"{row['Original i-block']:.3f} & "
            f"{row['I-block-count weighted']:.3f} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", ""])
    tex_path.write_text("\n".join(tex_lines))

    print(f"Saved {md_path}")
    print(f"Saved {tex_path}")
    return [full_path, md_path, tex_path]


def create_outputs(
    summary_df: pd.DataFrame,
    output_root: Path,
    table_target_coverage: float = 0.80,
) -> list[Path]:
    """Create reliability figures and raw set-size tables for all cases."""
    saved_paths: list[Path] = []
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    for simulation_name in sorted(summary_df["simulation"].unique()):
        saved_paths.extend(plot_reliability(summary_df, simulation_name, figures_dir))
        saved_paths.extend(
            save_raw_set_size_tables(
                summary_df,
                simulation_name,
                tables_dir,
                table_target_coverage=table_target_coverage,
            )
        )
    return saved_paths


def plot_all(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Backward-compatible test helper: create all default outputs."""
    return create_outputs(summary_df, output_dir)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("simulation_results/output/sim1_summary.csv"),
        help="summary CSV created by run_simulation_analysis.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output root; defaults to the simulation_results directory",
    )
    parser.add_argument(
        "--table-target-coverage",
        type=float,
        default=0.80,
        help="target coverage for compact manuscript-ready set-size table",
    )
    return parser


def main() -> None:
    """Read saved summary output and create figures/tables."""
    args = build_parser().parse_args()
    summary_df = pd.read_csv(args.summary)
    output_root = (
        args.output_dir
        if args.output_dir is not None
        else output_root_from_summary(args.summary)
    )
    create_outputs(
        summary_df,
        output_root,
        table_target_coverage=args.table_target_coverage,
    )


if __name__ == "__main__":
    main()
