"""
Detailed demo for dense three-state Markov i-block experiments.

For a quick live-meeting edit, use quick_meeting_demo.py. This script runs the
four fixed diagnostic cases requested for comparing original i-block CP,
permutation-count (D!) weighting, and i-block-count (D) weighting.
"""

from collections import Counter, defaultdict
import random
from statistics import mean

import numpy as np

from markov_cp_routines import (
    AggregatedCandidateResult,
    IBlockDiagnostic,
    aggregate_auxiliary_rows,
    auxiliary_candidate_table,
    original_iblock_table,
)


# ---------------------------------------------------------------------
# Experiment controls
# ---------------------------------------------------------------------

NUM_STATES = 3
ADJACENCY = np.ones((NUM_STATES, NUM_STATES), dtype=int)
HORIZONS = (1, 2)
ALPHA = 0.20
MAX_PERMUTATIONS = 500

TRAINING_LENGTH = 100
HISTORY_SEED = 8
DETAIL_RANDOM_SEED = 1
RANDOM_SEEDS = range(1, 4)
RUN_REPEATED_SEEDS = False
RANDOMIZED_TIES = False


def set_random_seed(seed: int) -> None:
    """Set randomness used by sampled block permutations and optional tie draws."""
    random.seed(seed)
    np.random.seed(seed)


def make_random_history() -> list[int]:
    """Generate one fixed random history without changing the global RNG."""
    rng = random.Random(HISTORY_SEED)
    return [
        rng.randint(1, NUM_STATES)
        for _ in range(TRAINING_LENGTH)
    ]


def experiment_cases() -> list[tuple[str, list[int]]]:
    """Return the four fixed histories used in the detailed demo."""
    return [
        ("Case 1: same-state history", [1] * 100),
        ("Case 2: fixed random history", make_random_history()),
        ("Case 3: two-cycle history", [1, 2] * 100),
        ("Case 4: three-cycle history", [1, 2, 3] * 100),
    ]


def path_text(path: tuple[int, ...]) -> str:
    """Format a candidate path compactly for aligned terminal tables."""
    return "(" + ",".join(str(state) for state in path) + ")"


def set_text(paths: list[tuple[int, ...]]) -> str:
    """Format a prediction set of original candidate paths."""
    if len(paths) == 0:
        return "{}"
    return "{" + ", ".join(path_text(path) for path in sorted(paths)) + "}"


def group_size_text(value: int | None) -> str:
    """Show small exact group sizes and mark large factorials compactly."""
    if value is None:
        return "large"
    return str(value)


def result_by_candidate(
    results: list[AggregatedCandidateResult],
) -> dict[tuple[int, ...], AggregatedCandidateResult]:
    """Index aggregated results by original candidate path."""
    return {result.original_candidate: result for result in results}


def print_history_summary(history: list[int]) -> None:
    """Print enough of the history to identify the experiment."""
    counts = Counter(history)
    print(f"history length: {len(history)}")
    print(f"state counts: {dict(sorted(counts.items()))}")
    print(f"first 20 states: {history[:20]}")


def print_original_table(rows: list[IBlockDiagnostic]) -> None:
    """Print original i-block candidate p-values and cardinalities."""
    print("\nOriginal i-block candidate table")
    print("y          p_block      D    log|Pi|   |Pi|      n_eval")
    print("-" * 66)
    for row in rows:
        print(
            f"{path_text(row.candidate):<10}"
            f"{row.p_value:>9.4f}"
            f"{row.n_permutable_blocks:>7}"
            f"{row.log_full_group_size:>10.2f}"
            f"{group_size_text(row.full_group_size):>8}"
            f"{row.n_permutations_evaluated:>10}"
        )


def print_auxiliary_rows(
    perm_results: list[AggregatedCandidateResult],
    iblock_results: list[AggregatedCandidateResult],
) -> None:
    """Print all extended-candidate rows used by both weighted methods."""
    iblock_by_candidate = result_by_candidate(iblock_results)

    print("\nAuxiliary extended-candidate table")
    print(
        "y          u   z          p_block      D    log|Pi|"
        "   perm_w    perm_c   iblock_w  iblock_c   n_eval"
    )
    print("-" * 110)

    for perm_result in perm_results:
        iblock_result = iblock_by_candidate[perm_result.original_candidate]
        iblock_rows = {
            row.auxiliary_state: row
            for row in iblock_result.auxiliary_rows
        }

        for row in perm_result.auxiliary_rows:
            iblock_row = iblock_rows[row.auxiliary_state]
            perm_contrib = row.normalized_cardinality_weight * row.p_value
            iblock_contrib = iblock_row.normalized_iblock_weight * iblock_row.p_value
            print(
                f"{path_text(row.original_candidate):<10}"
                f"{row.auxiliary_state:>2}"
                f"{path_text(row.extended_candidate):>11}"
                f"{row.p_value:>11.4f}"
                f"{row.n_permutable_blocks:>7}"
                f"{row.log_full_group_size:>10.2f}"
                f"{row.normalized_cardinality_weight:>9.4f}"
                f"{perm_contrib:>10.4f}"
                f"{row.normalized_iblock_weight:>10.4f}"
                f"{iblock_contrib:>10.4f}"
                f"{row.n_permutations_evaluated:>9}"
            )


def print_comparison(
    original_rows: list[IBlockDiagnostic],
    perm_results: list[AggregatedCandidateResult],
    iblock_results: list[AggregatedCandidateResult],
) -> None:
    """Print original versus both auxiliary weighted inclusion decisions."""
    original_by_candidate = {row.candidate: row for row in original_rows}
    iblock_by_candidate = result_by_candidate(iblock_results)

    print("\nOriginal versus auxiliary weighted CP")
    print("y          original_p   orig_in   q_perm   perm_in   q_iblock   iblock_in")
    print("-" * 78)

    for perm_result in perm_results:
        candidate = perm_result.original_candidate
        original_row = original_by_candidate[candidate]
        iblock_result = iblock_by_candidate[candidate]
        original_included = original_row.p_value > ALPHA
        print(
            f"{path_text(candidate):<10}"
            f"{original_row.p_value:>10.4f}"
            f"{str(original_included):>10}"
            f"{perm_result.q_tilde:>9.4f}"
            f"{str(perm_result.included):>10}"
            f"{iblock_result.q_tilde:>11.4f}"
            f"{str(iblock_result.included):>12}"
        )


def print_final_sets(
    original_rows: list[IBlockDiagnostic],
    perm_results: list[AggregatedCandidateResult],
    iblock_results: list[AggregatedCandidateResult],
) -> None:
    """Print final prediction sets explicitly."""
    original_set = sorted(row.candidate for row in original_rows if row.p_value > ALPHA)
    perm_set = sorted(
        result.original_candidate
        for result in perm_results
        if result.included
    )
    iblock_set = sorted(
        result.original_candidate
        for result in iblock_results
        if result.included
    )

    print("\nFINAL CP SETS")
    print(f"Original i-block CP set: {set_text(original_set)}")
    print(f"Permutation-count (D!) weighted CP set: {set_text(perm_set)}")
    print(f"I-block-count (D) weighted CP set: {set_text(iblock_set)}")

    print("\nFINAL CP SET DETAILS")
    print("method                         y          value")
    print("-" * 56)

    any_detail = False
    for row in original_rows:
        if row.p_value > ALPHA:
            any_detail = True
            print(
                f"{'Original i-block':<31}"
                f"{path_text(row.candidate):<10}"
                f"p={row.p_value:.4f}"
            )

    for result in perm_results:
        if result.included:
            any_detail = True
            print(
                f"{'Permutation-count (D!)':<31}"
                f"{path_text(result.original_candidate):<10}"
                f"q_perm={result.q_tilde:.4f}"
            )

    for result in iblock_results:
        if result.included:
            any_detail = True
            print(
                f"{'I-block-count (D)':<31}"
                f"{path_text(result.original_candidate):<10}"
                f"q_iblock={result.q_tilde:.4f}"
            )

    if not any_detail:
        print("(no candidates included by any method)")


def run_detailed_case(case_name: str, history: list[int], horizon: int) -> None:
    """Run one detailed seed and print every intermediate row."""
    print("\n" + "=" * 100)
    print(f"{case_name}; horizon={horizon}; detail seed={DETAIL_RANDOM_SEED}")
    print("=" * 100)
    print_history_summary(history)

    set_random_seed(DETAIL_RANDOM_SEED)
    original_rows = original_iblock_table(
        history,
        horizon,
        ADJACENCY,
        max_permutations=MAX_PERMUTATIONS,
        randomized_ties=RANDOMIZED_TIES,
    )

    set_random_seed(DETAIL_RANDOM_SEED)
    auxiliary_rows = auxiliary_candidate_table(
        history,
        horizon,
        ADJACENCY,
        max_permutations=MAX_PERMUTATIONS,
        randomized_ties=RANDOMIZED_TIES,
    )

    perm_results = aggregate_auxiliary_rows(
        auxiliary_rows,
        ALPHA,
        weighting="permutation_count",
    )
    iblock_results = aggregate_auxiliary_rows(
        auxiliary_rows,
        ALPHA,
        weighting="iblock_count",
    )

    print(f"number of original candidate paths: {len(original_rows)}")
    print(f"number of auxiliary extended-candidate rows: {len(auxiliary_rows)}")
    print_original_table(original_rows)
    print_auxiliary_rows(perm_results, iblock_results)
    print_comparison(original_rows, perm_results, iblock_results)
    print_final_sets(original_rows, perm_results, iblock_results)


def repeat_case(case_name: str, history: list[int], horizon: int) -> None:
    """Optionally repeat a fixed-history experiment over random seeds."""
    summaries: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: {
            "original_p": [],
            "original_included": [],
            "q_perm": [],
            "perm_included": [],
            "q_iblock": [],
            "iblock_included": [],
        }
    )

    for seed in RANDOM_SEEDS:
        set_random_seed(seed)
        original_rows = original_iblock_table(
            history,
            horizon,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
            randomized_ties=RANDOMIZED_TIES,
        )

        set_random_seed(seed)
        auxiliary_rows = auxiliary_candidate_table(
            history,
            horizon,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
            randomized_ties=RANDOMIZED_TIES,
        )

        perm_results = aggregate_auxiliary_rows(
            auxiliary_rows,
            ALPHA,
            weighting="permutation_count",
        )
        iblock_results = aggregate_auxiliary_rows(
            auxiliary_rows,
            ALPHA,
            weighting="iblock_count",
        )
        original_by_candidate = {row.candidate: row for row in original_rows}
        iblock_by_candidate = result_by_candidate(iblock_results)

        for perm_result in perm_results:
            candidate = perm_result.original_candidate
            original_row = original_by_candidate[candidate]
            iblock_result = iblock_by_candidate[candidate]
            bucket = summaries[candidate]
            bucket["original_p"].append(original_row.p_value)
            bucket["original_included"].append(float(original_row.p_value > ALPHA))
            bucket["q_perm"].append(perm_result.q_tilde)
            bucket["perm_included"].append(float(perm_result.included))
            bucket["q_iblock"].append(iblock_result.q_tilde)
            bucket["iblock_included"].append(float(iblock_result.included))

    print("\n" + "=" * 100)
    print(f"{case_name}; horizon={horizon}; repeated over seeds {list(RANDOM_SEEDS)}")
    print("=" * 100)
    print(
        "y          mean_orig_p   orig_freq   mean_q_perm   perm_freq"
        "   mean_q_iblock   iblock_freq"
    )
    print("-" * 96)

    for candidate in sorted(summaries):
        values = summaries[candidate]
        print(
            f"{path_text(candidate):<10}"
            f"{mean(values['original_p']):>12.4f}"
            f"{mean(values['original_included']):>12.2f}"
            f"{mean(values['q_perm']):>14.4f}"
            f"{mean(values['perm_included']):>12.2f}"
            f"{mean(values['q_iblock']):>16.4f}"
            f"{mean(values['iblock_included']):>14.2f}"
        )


def main() -> None:
    print("Dense three-state Markov i-block experiments")
    print(f"NUM_STATES = {NUM_STATES}")
    print(f"HORIZONS = {HORIZONS}")
    print(f"ALPHA = {ALPHA}")
    print(f"MAX_PERMUTATIONS = {MAX_PERMUTATIONS}")
    print(f"DETAIL_RANDOM_SEED = {DETAIL_RANDOM_SEED}")
    print(f"RANDOMIZED_TIES = {RANDOMIZED_TIES}")
    print(f"RUN_REPEATED_SEEDS = {RUN_REPEATED_SEEDS}")
    print(f"HISTORY_SEED = {HISTORY_SEED}")

    for case_name, history in experiment_cases():
        for horizon in HORIZONS:
            run_detailed_case(case_name, history, horizon)
            if RUN_REPEATED_SEEDS:
                repeat_case(case_name, history, horizon)


if __name__ == "__main__":
    main()
