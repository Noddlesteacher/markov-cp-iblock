"""
Run the two dense three-state experiments requested for the current meeting.

Edit the constants below to change the history length, permutation cap, alpha
level, or tie-breaking seeds. The mathematical routines live in
markov_cp_routines.py.
"""

from collections import Counter, defaultdict
import random
from statistics import mean

import numpy as np

from markov_cp_routines import (
    AggregatedCandidateResult,
    IBlockDiagnostic,
    auxiliary_candidate_table,
    cardinality_weighted_auxiliary_cp,
    original_iblock_table,
)


# ---------------------------------------------------------------------
# User-editable experiment settings
# ---------------------------------------------------------------------

NUM_STATES = 3
ADJACENCY = np.ones((NUM_STATES, NUM_STATES), dtype=int)
TRAINING_LENGTH = 100
HORIZON = 1
ALPHA = 0.2
MAX_PERMUTATIONS = 500

HISTORY_SEED = 8
DETAIL_TIE_SEED = 1
TIE_BREAKING_SEEDS = range(1, 11)


def set_tie_seed(seed: int) -> None:
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
    """Print original vs cardinality-weighted inclusion decisions."""
    original_by_candidate = {
        row.candidate: row
        for row in original_rows
    }

    print("\nOriginal vs cardinality-weighted auxiliary CP")
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


def run_detailed_case(case_name: str, history: list[int]) -> None:
    """Run one detailed seed and print every intermediate row."""
    print("\n" + "=" * 88)
    print(f"{case_name}: detailed diagnostic, tie seed = {DETAIL_TIE_SEED}")
    print("=" * 88)
    print_history_summary(history)

    set_tie_seed(DETAIL_TIE_SEED)
    original_rows = original_iblock_table(
        history,
        HORIZON,
        ADJACENCY,
        max_permutations=MAX_PERMUTATIONS,
    )

    set_tie_seed(DETAIL_TIE_SEED)
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

    direct_rows = auxiliary_candidate_table(
        history,
        HORIZON,
        ADJACENCY,
        max_permutations=MAX_PERMUTATIONS,
    )
    print(f"\nDirect auxiliary_candidate_table row count: {len(direct_rows)}")


def repeat_case(case_name: str, history: list[int]) -> None:
    """Repeat the fixed-history experiment over tie-breaking seeds."""
    summaries: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: {
            "original_p": [],
            "original_included": [],
            "q_tilde": [],
            "new_included": [],
        }
    )

    for seed in TIE_BREAKING_SEEDS:
        set_tie_seed(seed)
        original_rows = original_iblock_table(
            history,
            HORIZON,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
        )

        set_tie_seed(seed)
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

    print("\n" + "=" * 88)
    print(f"{case_name}: repeated over tie seeds {list(TIE_BREAKING_SEEDS)}")
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
    mixed_history = make_mixed_history()
    dominant_history = [1] * TRAINING_LENGTH

    print("Dense three-state Markov i-block experiments")
    print(f"NUM_STATES = {NUM_STATES}")
    print(f"TRAINING_LENGTH = {TRAINING_LENGTH}")
    print(f"HORIZON = {HORIZON}")
    print(f"ALPHA = {ALPHA}")
    print(f"MAX_PERMUTATIONS = {MAX_PERMUTATIONS}")
    print(f"HISTORY_SEED = {HISTORY_SEED}")

    run_case("Case 1: fixed random/mixed history", mixed_history)
    run_case("Case 2: dominant-state history", dominant_history)


if __name__ == "__main__":
    main()
