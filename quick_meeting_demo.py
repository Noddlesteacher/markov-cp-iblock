"""
Fast live-meeting demo for the dense three-state auxiliary experiment.

For most meeting checks, edit only HISTORY and SEEDS below, then run:

    python quick_meeting_demo.py
"""

from collections import defaultdict
import random
from statistics import mean

import numpy as np

from markov_cp_routines import cardinality_weighted_auxiliary_cp


# ---------------------------------------------------------------------
# Meeting controls
# ---------------------------------------------------------------------

HISTORY = [1,2] * 5
SEEDS = range(1)
ALPHA = 0.20
MAX_PERMUTATIONS = 500

ADJACENCY = np.ones((3, 3), dtype=int)
HORIZON = 1


def set_random_seed(seed: int) -> None:
    """Set randomness used when sampled block permutations are drawn."""
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


def main() -> None:
    summary: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: {"q_tilde": [], "included": []}
    )

    print("quick meeting demo")
    print(f"history length={len(HISTORY)}")
    print(f"seeds={SEEDS}")
    print(f"alpha={ALPHA}")
    print(f"max_permutations={MAX_PERMUTATIONS}")

    for seed in SEEDS:
        set_random_seed(seed)
        results = cardinality_weighted_auxiliary_cp(
            HISTORY,
            HORIZON,
            ALPHA,
            ADJACENCY,
            max_permutations=MAX_PERMUTATIONS,
        )

        print("\n" + "=" * 116)
        print(f"seed={seed}")
        print("=" * 116)
        print(
            "y       u   z       anchor   p_block      D    log|Pi|"
            "   |Pi|      n_eval   weight    contrib"
        )
        print("-" * 116)

        for result in results:
            for row in result.auxiliary_rows:
                contribution = row.normalized_cardinality_weight * row.p_value
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
                    f"{row.normalized_cardinality_weight:>9.4f}"
                    f"{contribution:>11.4f}"
                )

            summary[result.original_candidate]["q_tilde"].append(result.q_tilde)
            summary[result.original_candidate]["included"].append(float(result.included))
            print(
                f"q_tilde{path_text(result.original_candidate)} = "
                f"{result.q_tilde:.4f}; included={result.included}"
            )

    print("\nsummary across seeds")
    print("y       mean_q_tilde   inclusion_freq")
    print("-" * 40)
    for candidate in sorted(summary):
        values = summary[candidate]
        print(
            f"{path_text(candidate):<7}"
            f"{mean(values['q_tilde']):>13.4f}"
            f"{mean(values['included']):>16.2f}"
        )


if __name__ == "__main__":
    main()
