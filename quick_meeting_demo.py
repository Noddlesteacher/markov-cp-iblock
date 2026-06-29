"""
Fast live-meeting demo for the dense three-state auxiliary experiment.

For most meeting checks, edit only the small control block below, then run:

    python quick_meeting_demo.py
"""

from collections import defaultdict
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
# Meeting controls
# ---------------------------------------------------------------------

HISTORY = [1, 2] * 100
HORIZON = 1
SEEDS = [1]
ALPHA = 0.20
MAX_PERMUTATIONS = 500
RANDOMIZED_TIES = False

ADJACENCY = np.ones((3, 3), dtype=int)


def set_random_seed(seed: int) -> None:
    """Set randomness used by sampled block permutations and optional tie draws."""
    random.seed(seed)
    np.random.seed(seed)


def path_text(path: tuple[int, ...]) -> str:
    """Format a candidate path compactly for terminal output."""
    return "(" + ",".join(str(state) for state in path) + ")"


def group_size_text(value: int | None) -> str:
    """Show exact small factorials and mark larger ones compactly."""
    if value is None:
        return "large"
    return str(value)


def set_text(paths: list[tuple[int, ...]]) -> str:
    """Format a prediction set of original candidate paths."""
    if len(paths) == 0:
        return "{}"
    return "{" + ", ".join(path_text(path) for path in sorted(paths)) + "}"


def result_by_candidate(
    results: list[AggregatedCandidateResult],
) -> dict[tuple[int, ...], AggregatedCandidateResult]:
    """Index aggregated rows by original candidate path."""
    return {result.original_candidate: result for result in results}


def print_auxiliary_table(
    perm_results: list[AggregatedCandidateResult],
    iblock_results: list[AggregatedCandidateResult],
) -> None:
    """Print all auxiliary rows with both weighting schemes."""
    iblock_by_candidate = result_by_candidate(iblock_results)

    print(
        "y       u   z       anchor   p_block      D    log|Pi|"
        "   |Pi|      n_eval   perm_weight  perm_contrib"
        "   iblock_weight  iblock_contrib"
    )
    print("-" * 154)

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
                f"{path_text(row.original_candidate):<7}"
                f"{row.auxiliary_state:>2}"
                f"{path_text(row.extended_candidate):>8}"
                f"{row.anchor_state:>9}"
                f"{row.p_value:>10.4f}"
                f"{row.n_permutable_blocks:>7}"
                f"{row.log_full_group_size:>10.2f}"
                f"{group_size_text(row.full_group_size):>8}"
                f"{row.n_permutations_evaluated:>10}"
                f"{row.normalized_cardinality_weight:>14.8f}"
                f"{perm_contrib:>14.6f}"
                f"{row.normalized_iblock_weight:>15.8f}"
                f"{iblock_contrib:>15.6f}"
            )


def print_candidate_comparison(
    original_rows: list[IBlockDiagnostic],
    perm_results: list[AggregatedCandidateResult],
    iblock_results: list[AggregatedCandidateResult],
) -> None:
    """Print the original and two auxiliary inclusion decisions."""
    original_by_candidate = {row.candidate: row for row in original_rows}
    iblock_by_candidate = result_by_candidate(iblock_results)

    print("\nOriginal versus auxiliary weighted CP")
    print("y       original_p   q_perm   perm_included   q_iblock   iblock_included")
    print("-" * 78)

    for perm_result in perm_results:
        candidate = perm_result.original_candidate
        original_row = original_by_candidate[candidate]
        iblock_result = iblock_by_candidate[candidate]
        print(
            f"{path_text(candidate):<7}"
            f"{original_row.p_value:>10.4f}"
            f"{perm_result.q_tilde:>10.4f}"
            f"{str(perm_result.included):>16}"
            f"{iblock_result.q_tilde:>11.4f}"
            f"{str(iblock_result.included):>17}"
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


def main() -> None:
    summary: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: {
            "q_perm": [],
            "perm_included": [],
            "q_iblock": [],
            "iblock_included": [],
        }
    )

    print("quick meeting demo")
    print(f"history length={len(HISTORY)}")
    print(f"horizon={HORIZON}")
    print(f"seeds={list(SEEDS)}")
    print(f"alpha={ALPHA}")
    print(f"max_permutations={MAX_PERMUTATIONS}")
    print(f"randomized_ties={RANDOMIZED_TIES}")

    for seed in SEEDS:
        print("\n" + "=" * 154)
        print(f"seed={seed}")
        print("=" * 154)

        set_random_seed(seed)
        original_rows = original_iblock_table(
            HISTORY,
            HORIZON,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
            randomized_ties=RANDOMIZED_TIES,
        )

        set_random_seed(seed)
        auxiliary_rows = auxiliary_candidate_table(
            HISTORY,
            HORIZON,
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

        print_auxiliary_table(perm_results, iblock_results)
        print_candidate_comparison(original_rows, perm_results, iblock_results)
        print_final_sets(original_rows, perm_results, iblock_results)

        iblock_by_candidate = result_by_candidate(iblock_results)
        for perm_result in perm_results:
            candidate = perm_result.original_candidate
            iblock_result = iblock_by_candidate[candidate]
            summary[candidate]["q_perm"].append(perm_result.q_tilde)
            summary[candidate]["perm_included"].append(float(perm_result.included))
            summary[candidate]["q_iblock"].append(iblock_result.q_tilde)
            summary[candidate]["iblock_included"].append(float(iblock_result.included))

    print("\nsummary across seeds")
    print("y       mean_q_perm   perm_freq   mean_q_iblock   iblock_freq")
    print("-" * 64)
    for candidate in sorted(summary):
        values = summary[candidate]
        print(
            f"{path_text(candidate):<7}"
            f"{mean(values['q_perm']):>12.4f}"
            f"{mean(values['perm_included']):>12.2f}"
            f"{mean(values['q_iblock']):>15.4f}"
            f"{mean(values['iblock_included']):>13.2f}"
        )


if __name__ == "__main__":
    main()
