"""Simulation study for dense three-state Markov i-block CP experiments.

The core conformal routines live in markov_cp_routines.py. This script only
simulates Markov chains, calls the existing CP APIs, and summarizes empirical
coverage and prediction-set size.
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
    AuxiliaryDiagnostic,
    aggregate_auxiliary_rows,
    auxiliary_candidate_table,
    original_iblock_table,
)


P_SIM1 = np.array(
    [
        [0.7, 0.15, 0.15],
        [0.3, 0.6, 0.1],
        [0.2, 0.2, 0.6],
    ],
    dtype=float,
)
PI_SIM1 = np.array([0.4, 0.3, 0.3], dtype=float)

P_SIM2 = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)
PI_SIM2 = np.array([1.0, 0.0, 0.0], dtype=float)

P_SIM3 = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)
PI_SIM3 = np.array([1.0, 0.0, 0.0], dtype=float)

# Sweden-like stress test: the observed process stays in state 1.
# The CP adjacency remains dense; only the data-generating chain is single-state.
P_SIM4 = np.array(
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)
PI_SIM4 = np.array([1.0, 0.0, 0.0], dtype=float)

SIMULATION_CASES = {
    "sim1": (P_SIM1, PI_SIM1),
    "sim2": (P_SIM2, PI_SIM2),
    "sim3": (P_SIM3, PI_SIM3),
    "sim4": (P_SIM4, PI_SIM4),
}

N_SIM = 500
T = 500
HORIZONS = [1, 2, 3]
ALPHAS = [round(float(alpha), 2) for alpha in np.arange(0.05, 0.51, 0.05)]
MAX_PERMUTATIONS = 500
RANDOMIZED_TIES = False
MASTER_SEED = 20260701
ADJACENCY = np.ones((3, 3), dtype=int)


def validate_transition_matrix(P: np.ndarray, pi: np.ndarray) -> None:
    """Validate a finite-state transition matrix and initial distribution."""
    P_array = np.asarray(P, dtype=float)
    pi_array = np.asarray(pi, dtype=float)

    if P_array.ndim != 2:
        raise ValueError("P must be a two-dimensional square matrix.")

    if P_array.shape[0] != P_array.shape[1]:
        raise ValueError("P must be square.")

    if np.any(P_array < 0):
        raise ValueError("P entries must be nonnegative.")

    if not np.allclose(P_array.sum(axis=1), 1.0):
        raise ValueError("each row of P must sum to 1.")

    if pi_array.ndim != 1:
        raise ValueError("pi must be a one-dimensional vector.")

    if len(pi_array) != P_array.shape[0]:
        raise ValueError("pi length must match the number of states in P.")

    if np.any(pi_array < 0):
        raise ValueError("pi entries must be nonnegative.")

    if not np.isclose(pi_array.sum(), 1.0):
        raise ValueError("pi must sum to 1.")


def simulate_markov_chain(
    P: np.ndarray,
    pi: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> list[int]:
    """Simulate a Markov chain and return 1-based state labels."""
    validate_transition_matrix(P, pi)

    if length < 1:
        raise ValueError("length must be positive.")

    num_states = P.shape[0]
    zero_based_states = np.arange(num_states)
    current = int(rng.choice(zero_based_states, p=pi))
    sequence = [current + 1]

    for _ in range(length - 1):
        current = int(rng.choice(zero_based_states, p=P[current]))
        sequence.append(current + 1)

    return sequence


def path_text(path: tuple[int, ...]) -> str:
    """Format a candidate path compactly for CSV output."""
    return "(" + ",".join(str(state) for state in path) + ")"


def cp_set_text(paths: Sequence[tuple[int, ...]]) -> str:
    """Format a prediction set as space-separated path strings."""
    return " ".join(path_text(path) for path in sorted(paths))


def cp_set_from_original_rows(original_rows, alpha: float) -> list[tuple[int, ...]]:
    """Return original candidates whose p-value exceeds alpha."""
    return sorted(row.candidate for row in original_rows if row.p_value > alpha)


def cp_set_from_aggregated_results(results) -> list[tuple[int, ...]]:
    """Return original candidates included by an aggregated auxiliary rule."""
    return sorted(result.original_candidate for result in results if result.included)


def aggregate_auxiliary_rows_for_simulation(
    auxiliary_rows: Sequence[AuxiliaryDiagnostic],
    alpha: float,
    weighting: str,
) -> dict[tuple[int, ...], dict[str, float | bool]]:
    """Aggregate already-computed auxiliary rows for simulation bookkeeping."""
    if weighting not in ("permutation_count", "iblock_count"):
        raise ValueError("weighting must be 'permutation_count' or 'iblock_count'.")

    results = aggregate_auxiliary_rows(
        auxiliary_rows,
        alpha,
        weighting=weighting,
    )
    return {
        result.original_candidate: {
            "score": result.q_tilde,
            "included": result.included,
        }
        for result in results
    }


def run_one_replicate_for_horizon(
    sequence: list[int],
    T: int,
    h: int,
    alpha_values: list[float],
    adjacency: np.ndarray,
    max_permutations: int,
    randomized_ties: bool,
    rng_seed_for_cp: int,
    simulation_name: str = "simulation",
    rep_index: int = 0,
) -> list[dict]:
    """Run all three CP methods for one simulated sequence and one horizon."""
    if len(sequence) != T + h:
        raise ValueError("sequence must have length T + h.")

    history = sequence[:T]
    true_future = tuple(sequence[T : T + h])
    if len(true_future) != h:
        raise ValueError("true_future must have length h.")

    num_states = int(np.asarray(adjacency).shape[0])

    random.seed(int(rng_seed_for_cp))
    np.random.seed(int(rng_seed_for_cp))
    original_rows = original_iblock_table(
        history,
        h,
        adjacency,
        max_permutations=max_permutations,
        randomized_ties=randomized_ties,
    )

    random.seed(int(rng_seed_for_cp))
    np.random.seed(int(rng_seed_for_cp))
    auxiliary_rows = auxiliary_candidate_table(
        history,
        h,
        adjacency,
        max_permutations=max_permutations,
        randomized_ties=randomized_ties,
    )

    result_rows: list[dict] = []
    for alpha in alpha_values:
        original_set = cp_set_from_original_rows(original_rows, alpha)
        perm_results = aggregate_auxiliary_rows(
            auxiliary_rows,
            alpha,
            weighting="permutation_count",
        )
        iblock_results = aggregate_auxiliary_rows(
            auxiliary_rows,
            alpha,
            weighting="iblock_count",
        )
        method_sets = {
            "original": original_set,
            "permutation_count": cp_set_from_aggregated_results(perm_results),
            "iblock_count": cp_set_from_aggregated_results(iblock_results),
        }

        for method_name, cp_set in method_sets.items():
            if any(len(path) != h for path in cp_set):
                raise ValueError("final CP set contains a path with wrong horizon.")

            result_rows.append(
                {
                    "simulation": simulation_name,
                    "replicate": rep_index,
                    "horizon": h,
                    "alpha": alpha,
                    "target_coverage": 1.0 - alpha,
                    "method": method_name,
                    "covered": int(true_future in cp_set),
                    "set_size": len(cp_set),
                    "scaled_set_size": len(cp_set) / (num_states**h),
                    "true_future": path_text(true_future),
                    "cp_set": cp_set_text(cp_set),
                }
            )

    return result_rows


def run_simulation_case(
    name: str,
    P: np.ndarray,
    pi: np.ndarray,
    n_sim: int,
    T: int,
    horizons: list[int],
    alpha_values: list[float],
    max_permutations: int,
    randomized_ties: bool,
    master_seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    """Run one simulation case and save raw replicate-level results."""
    validate_transition_matrix(P, pi)

    if n_sim < 1:
        raise ValueError("n_sim must be positive.")

    if T < 1:
        raise ValueError("T must be positive.")

    if len(horizons) == 0 or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers.")

    output_dir.mkdir(parents=True, exist_ok=True)
    master_rng = np.random.default_rng(master_seed)
    max_horizon = max(horizons)
    all_rows: list[dict] = []

    print(
        f"Running {name}: n_sim={n_sim}, T={T}, horizons={horizons}, "
        f"alphas={alpha_values}"
    )
    for rep_index in range(1, n_sim + 1):
        if n_sim <= 20 or rep_index == 1 or rep_index == n_sim or rep_index % max(1, n_sim // 10) == 0:
            print(f"  replicate {rep_index}/{n_sim}")

        sequence_seed = int(master_rng.integers(0, 2**32 - 1))
        cp_seed = int(master_rng.integers(0, 2**32 - 1))
        sequence_rng = np.random.default_rng(sequence_seed)
        full_sequence = simulate_markov_chain(
            P,
            pi,
            length=T + max_horizon,
            rng=sequence_rng,
        )

        for horizon in horizons:
            all_rows.extend(
                run_one_replicate_for_horizon(
                    sequence=full_sequence[: T + horizon],
                    T=T,
                    h=horizon,
                    alpha_values=alpha_values,
                    adjacency=ADJACENCY,
                    max_permutations=max_permutations,
                    randomized_ties=randomized_ties,
                    rng_seed_for_cp=cp_seed + horizon,
                    simulation_name=name,
                    rep_index=rep_index,
                )
            )

    raw_df = pd.DataFrame(all_rows)
    raw_path = output_dir / f"{name}_raw_results.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"Saved raw results to {raw_path}")
    return raw_df


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize empirical coverage and prediction-set size."""
    grouped = raw_df.groupby(
        ["simulation", "method", "horizon", "alpha", "target_coverage"],
        as_index=False,
    )
    summary = grouped.agg(
        n_sim=("covered", "count"),
        empirical_coverage=("covered", "mean"),
        mean_set_size=("set_size", "mean"),
        sd_set_size=("set_size", "std"),
        mean_scaled_set_size=("scaled_set_size", "mean"),
        sd_scaled_set_size=("scaled_set_size", "std"),
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
            "mean_scaled_set_size",
            "sd_scaled_set_size",
        ]
    ]
    return summary.fillna(0.0)


def save_summary(summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Save per-case summaries and a combined all_summary.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for simulation_name, case_df in summary_df.groupby("simulation"):
        path = output_dir / f"{simulation_name}_summary.csv"
        case_df.to_csv(path, index=False)
        saved_paths.append(path)
        print(f"Saved summary to {path}")

    all_path = output_dir / "all_summary.csv"
    summary_df.to_csv(all_path, index=False)
    saved_paths.append(all_path)
    print(f"Saved combined summary to {all_path}")
    return saved_paths


def parse_bool(value: str) -> bool:
    """Parse a command-line boolean."""
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes", "y"):
        return True
    if normalized in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run a tiny smoke test")
    parser.add_argument(
        "--sim",
        choices=["sim1", "sim2", "sim3", "sim4", "all"],
        default="sim1",
        help="which simulation case to run",
    )
    parser.add_argument("--n-sim", type=int, default=N_SIM)
    parser.add_argument("--T", type=int, default=T)
    parser.add_argument("--max-permutations", type=int, default=MAX_PERMUTATIONS)
    parser.add_argument("--randomized-ties", type=parse_bool, default=RANDOMIZED_TIES)
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_results"),
    )
    return parser


def main() -> None:
    """Run the requested simulation study."""
    args = build_parser().parse_args()

    if args.quick:
        n_sim = 5
        training_length = 30
        horizons = [1]
        alpha_values = [0.2]
        max_permutations = 50
    else:
        n_sim = args.n_sim
        training_length = args.T
        horizons = HORIZONS
        alpha_values = ALPHAS
        max_permutations = args.max_permutations

    if args.sim == "all":
        case_names = ["sim1", "sim2", "sim3", "sim4"]
    else:
        case_names = [args.sim]

    all_raw_dfs: list[pd.DataFrame] = []
    for case_offset, case_name in enumerate(case_names):
        P, pi = SIMULATION_CASES[case_name]
        raw_df = run_simulation_case(
            name=case_name,
            P=P,
            pi=pi,
            n_sim=n_sim,
            T=training_length,
            horizons=horizons,
            alpha_values=alpha_values,
            max_permutations=max_permutations,
            randomized_ties=args.randomized_ties,
            master_seed=args.seed + case_offset,
            output_dir=args.output_dir,
        )
        all_raw_dfs.append(raw_df)

    combined_raw = pd.concat(all_raw_dfs, ignore_index=True)
    summary_df = summarize_results(combined_raw)
    save_summary(summary_df, args.output_dir)


if __name__ == "__main__":
    main()
