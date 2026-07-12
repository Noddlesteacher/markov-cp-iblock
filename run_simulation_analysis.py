"""Run CP analysis on saved simulation inputs and diagnostic cases.

This script loads already-generated state sequences, calls the reusable
conformal routines, and saves raw and summarized output. It does not generate
the main stochastic simulation input and does not create plots.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import pandas as pd

from markov_cp_routines import (
    aggregate_auxiliary_rows,
    auxiliary_candidate_table,
    original_iblock_table,
)
from simulation_study import (
    DEFAULT_SEED,
    DEFAULT_TARGET_COVERAGES,
    DEFAULT_T,
    P_SIM1,
    PI_SIM1,
    simulate_markov_chain,
)


DEFAULT_HORIZONS = (1, 2, 3)
DEFAULT_MAX_PERMUTATIONS = 500
DEFAULT_RANDOMIZED_TIES = False
DEFAULT_METHODS = ("original", "iblock_count")
LEGACY_METHOD = "permutation_count"
ADJACENCY = np.ones((3, 3), dtype=int)


def path_text(path: tuple[int, ...]) -> str:
    """Format a candidate path compactly for CSV output."""
    return "(" + ",".join(str(state) for state in path) + ")"


def cp_set_text(paths: Sequence[tuple[int, ...]]) -> str:
    """Format a prediction set as space-separated path strings."""
    return " ".join(path_text(path) for path in sorted(paths))


def target_coverages_to_alphas(target_coverages: Sequence[float]) -> list[float]:
    """Convert target coverages to alpha values, supporting target 1.00."""
    alphas = []
    for coverage in target_coverages:
        if coverage < 0 or coverage > 1:
            raise ValueError("target coverages must be between 0 and 1.")
        alpha = max(0.0, min(1.0, 1.0 - float(coverage)))
        alphas.append(round(alpha, 12))
    return alphas


def _set_cp_seed(seed: int) -> None:
    """Set global RNGs used by sampled block permutations and randomized ties."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def cp_set_from_original_rows(original_rows, alpha: float) -> list[tuple[int, ...]]:
    """Return original candidates whose p-value exceeds alpha."""
    return sorted(row.candidate for row in original_rows if row.p_value > alpha)


def cp_set_from_aggregated_results(results) -> list[tuple[int, ...]]:
    """Return original candidates included by an auxiliary aggregate."""
    return sorted(result.original_candidate for result in results if result.included)


def load_simulation_inputs(input_path: Path) -> dict:
    """Load saved sequences and metadata from simulation_study.py output."""
    data = np.load(input_path, allow_pickle=False)
    return {
        "simulation": str(data["simulation"].item()),
        "sequences": np.asarray(data["sequences"], dtype=int),
        "P": np.asarray(data["P"], dtype=float),
        "pi": np.asarray(data["pi"], dtype=float),
        "T": int(data["T"].item()),
        "horizons": tuple(int(value) for value in data["horizons"].tolist()),
        "target_coverages": tuple(
            float(value) for value in data["target_coverages"].tolist()
        ),
        "seed": int(data["seed"].item()),
        "state_labels": tuple(int(value) for value in data["state_labels"].tolist()),
    }


def check_nested_history_and_future(
    full_sequence: Sequence[int],
    training_length: int,
    horizons: Sequence[int],
) -> dict[int, tuple[list[int], tuple[int, ...]]]:
    """Slice one full sequence and verify common history plus nested futures."""
    sorted_horizons = sorted(int(horizon) for horizon in horizons)
    if len(sorted_horizons) == 0:
        raise ValueError("horizons must be nonempty.")

    slices: dict[int, tuple[list[int], tuple[int, ...]]] = {}
    for horizon in sorted_horizons:
        history = list(full_sequence[:training_length])
        true_future = tuple(full_sequence[training_length : training_length + horizon])
        if len(true_future) != horizon:
            raise ValueError("true future has the wrong length.")
        slices[horizon] = (history, true_future)

    reference_history = np.asarray(slices[sorted_horizons[0]][0], dtype=int)
    for horizon in sorted_horizons[1:]:
        assert np.array_equal(reference_history, np.asarray(slices[horizon][0], dtype=int))

    for previous, current in zip(sorted_horizons, sorted_horizons[1:]):
        previous_future = slices[previous][1]
        current_future = slices[current][1]
        assert previous_future == current_future[:previous]

    return slices


def evaluate_history_for_horizons(
    full_sequence: Sequence[int],
    training_length: int,
    horizons: Sequence[int],
    target_coverages: Sequence[float],
    max_permutations: int,
    randomized_ties: bool,
    cp_seed: int,
    simulation_name: str,
    replicate: int,
    methods: Sequence[str] = DEFAULT_METHODS,
) -> list[dict]:
    """Compute CP outputs once per horizon, then threshold over alpha values."""
    slices = check_nested_history_and_future(full_sequence, training_length, horizons)
    alphas = target_coverages_to_alphas(target_coverages)
    rows: list[dict] = []

    for horizon in sorted(slices):
        history, true_future = slices[horizon]
        horizon_seed = int(cp_seed) + 1009 * int(horizon)

        _set_cp_seed(horizon_seed)
        original_rows = original_iblock_table(
            history,
            horizon,
            ADJACENCY,
            max_permutations=max_permutations,
            randomized_ties=randomized_ties,
        )

        auxiliary_rows = None
        if "iblock_count" in methods or LEGACY_METHOD in methods:
            _set_cp_seed(horizon_seed + 100_000)
            auxiliary_rows = auxiliary_candidate_table(
                history,
                horizon,
                ADJACENCY,
                max_permutations=max_permutations,
                randomized_ties=randomized_ties,
            )

        for target_coverage, alpha in zip(target_coverages, alphas):
            method_sets: dict[str, list[tuple[int, ...]]] = {}
            if "original" in methods:
                method_sets["original"] = cp_set_from_original_rows(original_rows, alpha)
            if "iblock_count" in methods:
                iblock_results = aggregate_auxiliary_rows(
                    auxiliary_rows,
                    alpha,
                    weighting="iblock_count",
                )
                method_sets["iblock_count"] = cp_set_from_aggregated_results(
                    iblock_results
                )
            if LEGACY_METHOD in methods:
                perm_results = aggregate_auxiliary_rows(
                    auxiliary_rows,
                    alpha,
                    weighting="permutation_count",
                )
                method_sets[LEGACY_METHOD] = cp_set_from_aggregated_results(
                    perm_results
                )

            for method_name, cp_set in method_sets.items():
                if any(len(path) != horizon for path in cp_set):
                    raise ValueError("final CP set contains a path with wrong horizon.")
                rows.append(
                    {
                        "simulation": simulation_name,
                        "replicate": replicate,
                        "horizon": horizon,
                        "alpha": alpha,
                        "target_coverage": float(target_coverage),
                        "method": method_name,
                        "covered": int(true_future in cp_set),
                        "set_size": len(cp_set),
                        "true_future": path_text(true_future),
                        "cp_set": cp_set_text(cp_set),
                    }
                )

    return rows


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize empirical coverage and raw prediction-set cardinality."""
    grouped = raw_df.groupby(
        ["simulation", "method", "horizon", "alpha", "target_coverage"],
        as_index=False,
    )
    summary = grouped.agg(
        n_sim=("covered", "count"),
        empirical_coverage=("covered", "mean"),
        mean_set_size=("set_size", "mean"),
        sd_set_size=("set_size", "std"),
    )
    summary["coverage_mcse"] = summary.apply(
        lambda row: math.sqrt(
            row["empirical_coverage"] * (1.0 - row["empirical_coverage"]) / row["n_sim"]
        ),
        axis=1,
    )
    summary = summary[
        [
            "simulation",
            "method",
            "horizon",
            "alpha",
            "target_coverage",
            "n_sim",
            "empirical_coverage",
            "coverage_mcse",
            "mean_set_size",
            "sd_set_size",
        ]
    ]
    return summary.fillna(0.0)


def output_suffix_from_input(input_path: Path, quick: bool) -> str:
    """Use a quick suffix when the input file or CLI indicates quick mode."""
    return "_quick" if quick or input_path.stem.endswith("_quick") else ""


def run_saved_input_analysis(
    input_path: Path,
    output_dir: Path,
    max_permutations: int,
    randomized_ties: bool,
    include_legacy: bool,
    quick: bool = False,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    """Analyze saved stochastic simulation sequences."""
    loaded = load_simulation_inputs(input_path)
    sequences = loaded["sequences"]
    methods = list(DEFAULT_METHODS)
    if include_legacy:
        methods.append(LEGACY_METHOD)

    print(f"Simulation: {loaded['simulation']}")
    print(f"Number of replicates: {len(sequences)}")
    print(f"Training length: {loaded['T']}")
    print(f"Horizons: {', '.join(str(h) for h in loaded['horizons'])}")
    print(
        "Target coverages: "
        + ", ".join(f"{coverage:.2f}" for coverage in loaded["target_coverages"])
    )
    print(f"Maximum permutations: {max_permutations}")
    print(f"Randomized ties: {randomized_ties}")
    print(f"Seed: {loaded['seed']}")

    all_rows: list[dict] = []
    for replicate_index, full_sequence in enumerate(sequences, start=1):
        if len(sequences) <= 20 or replicate_index == 1 or replicate_index == len(sequences):
            print(f"  replicate {replicate_index}/{len(sequences)}")
        all_rows.extend(
            evaluate_history_for_horizons(
                full_sequence=full_sequence.tolist(),
                training_length=loaded["T"],
                horizons=loaded["horizons"],
                target_coverages=loaded["target_coverages"],
                max_permutations=max_permutations,
                randomized_ties=randomized_ties,
                cp_seed=loaded["seed"] + 10_000 * replicate_index,
                simulation_name=loaded["simulation"],
                replicate=replicate_index,
                methods=methods,
            )
        )

    analysis_dir = output_dir / "output"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_suffix_from_input(input_path, quick)
    raw_path = analysis_dir / f"{loaded['simulation']}_raw_results{suffix}.csv"
    summary_path = analysis_dir / f"{loaded['simulation']}_summary{suffix}.csv"

    raw_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(raw_df)
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved raw results to {raw_path}")
    print(f"Saved summary to {summary_path}")
    return raw_path, summary_path, raw_df, summary_df


def make_case2_sequence(
    training_length: int = DEFAULT_T,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Generate one genuine dense Markov sequence for the Case 2 diagnostic."""
    rng = np.random.default_rng(seed)
    return simulate_markov_chain(
        P_SIM1,
        PI_SIM1,
        length=training_length + max(horizons),
        rng=rng,
    )


def deterministic_sequence(
    case: str,
    training_length: int,
    horizons: Sequence[int],
) -> list[int]:
    """Construct deterministic diagnostic sequences."""
    total_length = training_length + max(horizons)
    if case == "sweden":
        return [1] * total_length
    if case == "two_cycle":
        return [1 if index % 2 == 0 else 2 for index in range(total_length)]
    if case == "three_cycle":
        return [(index % 3) + 1 for index in range(total_length)]
    raise ValueError(f"unknown deterministic case: {case}")


def run_single_sequence_diagnostic(
    case_name: str,
    full_sequence: Sequence[int],
    training_length: int,
    horizons: Sequence[int],
    target_coverages: Sequence[float],
    output_dir: Path,
    max_permutations: int,
    randomized_ties: bool,
    seed: int,
    include_legacy: bool = False,
) -> tuple[Path, pd.DataFrame]:
    """Run one diagnostic sequence over horizons and target coverages."""
    methods = list(DEFAULT_METHODS)
    if include_legacy:
        methods.append(LEGACY_METHOD)

    rows = evaluate_history_for_horizons(
        full_sequence=full_sequence,
        training_length=training_length,
        horizons=horizons,
        target_coverages=target_coverages,
        max_permutations=max_permutations,
        randomized_ties=randomized_ties,
        cp_seed=seed,
        simulation_name=case_name,
        replicate=1,
        methods=methods,
    )
    df = pd.DataFrame(rows)
    df.insert(0, "case", case_name)

    output_path = output_dir / "output" / f"{case_name}_diagnostic.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    slices = check_nested_history_and_future(full_sequence, training_length, horizons)
    horizon_text = ", ".join(f"h={horizon}" for horizon in sorted(slices))
    print(f"CASE: {case_name}")
    print(f"Training length: {training_length}")
    print(f"Same training history for {horizon_text}: True")
    print(f"First 20 training states: {list(full_sequence[:20])}")
    print(f"Final 10 training states: {list(full_sequence[training_length - 10:training_length])}")
    for horizon in sorted(slices):
        print(f"h = {horizon} true future: {path_text(slices[horizon][1])}")

    if case_name == "case2":
        print("Transition matrix P:")
        print(P_SIM1)
        print(f"Initial distribution pi: {PI_SIM1.tolist()}")
        print(f"Random seed: {seed}")

    alpha_020 = df[np.isclose(df["target_coverage"], 0.80)]
    if not alpha_020.empty:
        print("Diagnostic alpha = 0.20 results:")
        for _, row in alpha_020.iterrows():
            print(
                f"  h={int(row['horizon'])}, {row['method']}: "
                f"covered={bool(row['covered'])}, set_size={int(row['set_size'])}, "
                f"set={row['cp_set']}"
            )

    print(f"Saved diagnostic to {output_path}")
    return output_path, df


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean."""
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def parse_float_tuple(value: str) -> tuple[float, ...]:
    """Parse comma-separated floats from the command line."""
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use quick settings")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="saved .npz sequence input from simulation_study.py",
    )
    parser.add_argument(
        "--case",
        choices=["case2", "sweden", "two_cycle", "three_cycle"],
        default=None,
        help="run one diagnostic case instead of saved stochastic input",
    )
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-permutations", type=int, default=DEFAULT_MAX_PERMUTATIONS)
    parser.add_argument("--randomized-ties", type=parse_bool, default=DEFAULT_RANDOMIZED_TIES)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument(
        "--target-coverages",
        type=parse_float_tuple,
        default=DEFAULT_TARGET_COVERAGES,
        help="comma-separated target coverages",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_results"),
    )
    return parser


def main() -> None:
    """Run saved-input analysis or one deterministic diagnostic."""
    args = build_parser().parse_args()
    if args.quick:
        training_length = 30
        target_coverages = (0.80,)
        max_permutations = min(args.max_permutations, 50)
    else:
        training_length = args.T
        target_coverages = tuple(args.target_coverages)
        max_permutations = args.max_permutations

    if args.case is None:
        input_path = args.input
        if input_path is None:
            suffix = "_quick" if args.quick else ""
            input_path = args.output_dir / "input" / f"sim1_sequences{suffix}.npz"
        run_saved_input_analysis(
            input_path=input_path,
            output_dir=args.output_dir,
            max_permutations=max_permutations,
            randomized_ties=args.randomized_ties,
            include_legacy=args.include_legacy,
            quick=args.quick,
        )
        return

    horizons = (1,) if args.quick else DEFAULT_HORIZONS
    if args.case == "case2":
        full_sequence = make_case2_sequence(
            training_length=training_length,
            horizons=horizons,
            seed=args.seed,
        )
        case_name = "case2"
    else:
        full_sequence = deterministic_sequence(args.case, training_length, horizons)
        case_name = args.case

    run_single_sequence_diagnostic(
        case_name=case_name,
        full_sequence=full_sequence,
        training_length=training_length,
        horizons=horizons,
        target_coverages=target_coverages,
        output_dir=args.output_dir,
        max_permutations=max_permutations,
        randomized_ties=args.randomized_ties,
        seed=args.seed,
        include_legacy=args.include_legacy,
    )


if __name__ == "__main__":
    main()
