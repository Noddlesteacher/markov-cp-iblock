"""
Quick demo for dense three-state Markov i-block experiments.

For a new experiment, edit the "Quick experiment controls" block below, then run:

    python run_demo.py
"""

from collections import Counter, defaultdict
import random
from statistics import mean

import numpy as np

from markov_cp_routines import (
    AggregatedCandidateResult,
    IBlockDiagnostic,
    cardinality_weighted_auxiliary_cp,
    original_iblock_table,
)


# ---------------------------------------------------------------------
# Quick experiment controls
#
# For most experiments, edit only this block.
# ---------------------------------------------------------------------

NUM_STATES = 3
ADJACENCY = np.ones((NUM_STATES, NUM_STATES), dtype=int)
HORIZON = 1
ALPHA = 0.2
MAX_PERMUTATIONS = 500

# Put the observed training history here.
HISTORY = [1] * 100

# Change this one number to reproduce one detailed randomized p-value table.
DETAIL_RANDOM_SEED = 1

# This is the simple repeated-run loop: range(1, 11) gives ten runs.
RANDOM_SEEDS = range(1, 11)

# Set this to True only when you want to rerun the two built-in examples.
RUN_BUILT_IN_CASES = False


# ---------------------------------------------------------------------
# Optional built-in histories
# ---------------------------------------------------------------------

TRAINING_LENGTH = 100
HISTORY_SEED = 8


def set_random_seed(seed: int) -> None:
    """Set all randomness used by randomized tie-breaking and sampling."""
    random.seed(seed)
    np.random.seed(seed)


def make_mixed_history() -> list[int]:
    """Generate one fixed random/mixed history and then hold it fixed."""
    random.seed(HISTORY_SEED)
    return [
        random.randint(1, NUM_STATES)
        for _ in range(TRAINING_LENGTH)
    ]


def path_text(path: tuple[int, ...]) -> str:
    """Format a candidate path compactly for aligned terminal tables."""
    return "(" + ",".join(str(state) for state in path) + ")"


def group_size_text(value: int | None) -> str:
    """Show small exact group sizes and mark large factorials by log size."""
    if value is None:
        return "large"
    return str(value)


def print_history_summary(history: list[int]) -> None:
    """Print enough of the history to identify the experiment."""
    counts = Counter(history)
    print(f"history length: {len(history)}")
    print(f"state counts: {dict(sorted(counts.items()))}")
    print(f"first 20 states: {history[:20]}")


def print_original_table(rows: list[IBlockDiagnostic]) -> None:
    """Print original i-block candidate p-values and cardinalities."""
    print("\nOriginal i-block candidate table")
    print("y       p_block      D    log|Pi|   |Pi|      n_eval")
    print("-" * 62)
    for row in rows:
        print(
            f"{path_text(row.candidate):<7}"
            f"{row.p_value:>9.4f}"
            f"{row.n_permutable_blocks:>7}"
            f"{row.log_full_group_size:>10.2f}"
            f"{group_size_text(row.full_group_size):>8}"
            f"{row.n_permutations_evaluated:>10}"
        )


def print_auxiliary_rows(results: list[AggregatedCandidateResult]) -> None:
    """Print all extended-candidate rows used by q_tilde."""
    print("\nAuxiliary extended-candidate table")
    print("y       u   z       p_block      D    log|Pi|   weight    n_eval")
    print("-" * 76)

    for result in results:
        for row in result.auxiliary_rows:
            print(
                f"{path_text(row.original_candidate):<7}"
                f"{row.auxiliary_state:>2}"
                f"{path_text(row.extended_candidate):>8}"
                f"{row.p_value:>11.4f}"
                f"{row.n_permutable_blocks:>7}"
                f"{row.log_full_group_size:>10.2f}"
                f"{row.normalized_cardinality_weight:>9.4f}"
                f"{row.n_permutations_evaluated:>10}"
            )


def print_comparison(
    original_rows: list[IBlockDiagnostic],
    weighted_results: list[AggregatedCandidateResult],
) -> None:
    """Print original versus cardinality-weighted inclusion decisions."""
    original_by_candidate = {
        row.candidate: row
        for row in original_rows
    }

    print("\nOriginal versus cardinality-weighted auxiliary CP")
    print("y       orig_p    q_tilde   orig_in   new_in")
    print("-" * 50)

    for result in weighted_results:
        original_row = original_by_candidate[result.original_candidate]
        original_included = original_row.p_value > ALPHA
        print(
            f"{path_text(result.original_candidate):<7}"
            f"{original_row.p_value:>8.4f}"
            f"{result.q_tilde:>10.4f}"
            f"{str(original_included):>10}"
            f"{str(result.included):>9}"
        )


def print_seed_by_seed_rows(
    case_name: str,
    seed_rows: list[tuple[int, tuple[int, ...], float, bool, float, bool]],
) -> None:
    """Print the direct for-loop output: one row per seed and candidate."""
    print("\n" + "=" * 88)
    print(f"{case_name}: p-values from each random seed")
    print("=" * 88)
    print("seed    y       original_p   orig_in   q_tilde    new_in")
    print("-" * 62)

    for seed, candidate, original_p, original_included, q_tilde, new_included in seed_rows:
        print(
            f"{seed:<8}"
            f"{path_text(candidate):<8}"
            f"{original_p:>10.4f}"
            f"{str(original_included):>10}"
            f"{q_tilde:>10.4f}"
            f"{str(new_included):>10}"
        )


def run_detailed_case(case_name: str, history: list[int]) -> None:
    """Run one detailed seed and print every intermediate row."""
    print("\n" + "=" * 88)
    print(f"{case_name}: detailed diagnostic, random seed = {DETAIL_RANDOM_SEED}")
    print("=" * 88)
    print_history_summary(history)

    set_random_seed(DETAIL_RANDOM_SEED)
    original_rows = original_iblock_table(
        history,
        HORIZON,
        ADJACENCY,
        max_permutations=MAX_PERMUTATIONS,
    )

    set_random_seed(DETAIL_RANDOM_SEED)
    weighted_results = cardinality_weighted_auxiliary_cp(
        history,
        HORIZON,
        ALPHA,
        ADJACENCY,
        max_permutations=MAX_PERMUTATIONS,
    )

    print_original_table(original_rows)
    print_auxiliary_rows(weighted_results)
    print_comparison(original_rows, weighted_results)


def repeat_case(case_name: str, history: list[int]) -> None:
    """Repeat the fixed-history experiment over random seeds."""
    summaries: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: {
            "original_p": [],
            "original_included": [],
            "q_tilde": [],
            "new_included": [],
        }
    )
    seed_rows: list[tuple[int, tuple[int, ...], float, bool, float, bool]] = []

    for seed in RANDOM_SEEDS:
        set_random_seed(seed)
        original_rows = original_iblock_table(
            history,
            HORIZON,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
        )

        set_random_seed(seed)
        weighted_results = cardinality_weighted_auxiliary_cp(
            history,
            HORIZON,
            ALPHA,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
        )

        original_by_candidate = {
            row.candidate: row
            for row in original_rows
        }

        for result in weighted_results:
            original_row = original_by_candidate[result.original_candidate]
            bucket = summaries[result.original_candidate]
            bucket["original_p"].append(original_row.p_value)
            bucket["original_included"].append(float(original_row.p_value > ALPHA))
            bucket["q_tilde"].append(result.q_tilde)
            bucket["new_included"].append(float(result.included))
            seed_rows.append(
                (
                    seed,
                    result.original_candidate,
                    original_row.p_value,
                    original_row.p_value > ALPHA,
                    result.q_tilde,
                    result.included,
                )
            )

    print_seed_by_seed_rows(case_name, seed_rows)

    print("\n" + "=" * 88)
    print(f"{case_name}: repeated over random seeds {list(RANDOM_SEEDS)}")
    print("=" * 88)
    print("y       mean_orig_p   orig_freq   mean_q_tilde   new_freq")
    print("-" * 62)

    for candidate in sorted(summaries):
        values = summaries[candidate]
        print(
            f"{path_text(candidate):<7}"
            f"{mean(values['original_p']):>12.4f}"
            f"{mean(values['original_included']):>12.2f}"
            f"{mean(values['q_tilde']):>15.4f}"
            f"{mean(values['new_included']):>11.2f}"
        )


def run_case(case_name: str, history: list[int]) -> None:
    """Run the detailed and repeated views for one fixed history."""
    run_detailed_case(case_name, history)
    repeat_case(case_name, history)


def main() -> None:
    print("Dense three-state Markov i-block experiments")
    print(f"NUM_STATES = {NUM_STATES}")
    print(f"HORIZON = {HORIZON}")
    print(f"ALPHA = {ALPHA}")
    print(f"MAX_PERMUTATIONS = {MAX_PERMUTATIONS}")
    print(f"DETAIL_RANDOM_SEED = {DETAIL_RANDOM_SEED}")

    if RUN_BUILT_IN_CASES:
        mixed_history = make_mixed_history()
        dominant_history = [1] * TRAINING_LENGTH

        print(f"HISTORY_SEED = {HISTORY_SEED}")
        run_case("Case 1: fixed random/mixed history", mixed_history)
        run_case("Case 2: dominant-state history", dominant_history)
    else:
        run_case("Editable history", HISTORY)


if __name__ == "__main__":
    main()
