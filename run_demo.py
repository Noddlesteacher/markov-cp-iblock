"""
run_demo.py

Demo / run file for the Markov CP routines.

This is the file to edit when we want to quickly test a new training sequence,
a new adjacency matrix, or a new forecast horizon. The helper functions and
main methods are kept in markov_cp_routines.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from markov_cp_routines import (
    block_p_value_for_candidate,
    enumerate_paths,
    iblock_cp_prediction_set,
    likelihood_prediction_set,
    plot_method_compositions,
    print_prediction_set_summary,
    run_all_methods,
)


# ---------------------------------------------------------------------
# Problem setup helpers
# ---------------------------------------------------------------------

CONFLICT_ADJACENCY = np.array(
    [
        [1, 1, 0, 0],  # state 1 -> state 1 or 2
        [0, 0, 1, 1],  # state 2 -> state 3 or 4
        [0, 0, 1, 1],  # state 3 -> state 3 or 4
        [1, 1, 0, 0],  # state 4 -> state 1 or 2
    ],
    dtype=int,
)


def summarize_results(title: str, results: dict) -> None:
    """Print a compact summary of all three methods."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"horizon = {results['horizon']}, alpha = {results['alpha']}")
    print(f"number of candidate paths = {len(results['candidate_paths'])}")
    print_prediction_set_summary("Likelihood baseline", results["likelihood"])
    print_prediction_set_summary("Original i-block CP", results["original_iblock"])
    print_prediction_set_summary("CP+1", results["cp_plus_one"])


# ---------------------------------------------------------------------
# Demo 1: simple 3-state fully connected example
# ---------------------------------------------------------------------

def demo_three_state_fully_connected() -> None:
    """Small abstract example requested by the meeting discussion."""
    adjacency = np.ones((3, 3), dtype=int)  # all transitions are allowed
    history = [1, 1, 2, 3, 1, 2]
    horizon = 2
    alpha = 0.2
    max_permutations = 200
    random_seed = 123

    results = run_all_methods(
        history=history,
        horizon=horizon,
        alpha=alpha,
        adjacency=adjacency,
        max_permutations=max_permutations,
        cp_plus_one_anchor=1,
        random_seed=random_seed,
    )

    summarize_results("Demo 1: three-state fully connected Markov chain", results)


# ---------------------------------------------------------------------
# Demo 2: short Sweden-like history
# ---------------------------------------------------------------------

def demo_sweden_like_short() -> None:
    """Short dominant-state history: less training information."""
    history = [1] * 6
    horizon = 3
    alpha = 0.2
    max_permutations = 200
    random_seed = 123

    results = run_all_methods(
        history=history,
        horizon=horizon,
        alpha=alpha,
        adjacency=CONFLICT_ADJACENCY,
        max_permutations=max_permutations,
        cp_plus_one_anchor=1,
        random_seed=random_seed,
    )

    summarize_results("Demo 2: Sweden-like short history, T = 6, H = 3", results)


# ---------------------------------------------------------------------
# Demo 3: long Sweden-like history
# ---------------------------------------------------------------------

def demo_sweden_like_long(save_plots: bool = True) -> None:
    """Long dominant-state history: many repeated state-1 observations.

    The meeting suggestion was to first keep the horizon small, such as H = 3,
    so that the code is easy to test while we work on the theory.
    """
    history = [1] * 420
    horizon = 3
    alpha = 0.2
    max_permutations = 200
    random_seed = 123

    results = run_all_methods(
        history=history,
        horizon=horizon,
        alpha=alpha,
        adjacency=CONFLICT_ADJACENCY,
        max_permutations=max_permutations,
        cp_plus_one_anchor=1,
        random_seed=random_seed,
    )

    summarize_results("Demo 3: Sweden-like long history, T = 420, H = 3", results)

    if save_plots:
        out_dir = Path("demo_outputs")
        out_dir.mkdir(exist_ok=True)
        plot_method_compositions(
            results,
            filename_prefix=str(out_dir / "sweden_like_long_H3"),
            show=False,
        )
        print(f"Saved state-composition plots to: {out_dir.resolve()}")


# ---------------------------------------------------------------------
# Optional diagnostic: one candidate p-value
# ---------------------------------------------------------------------

def demo_one_candidate_diagnostic() -> None:
    """Show how to inspect the i-block structure for one candidate."""
    history = [1] * 6
    candidate = (1, 1, 2)
    max_permutations = 50
    seed = 123

    import random

    rng = random.Random(seed)
    p_value, details = block_p_value_for_candidate(
        history=history,
        candidate=candidate,
        adjacency=CONFLICT_ADJACENCY,
        max_permutations=max_permutations,
        rng=rng,
        return_details=True,
    )

    print("\n" + "=" * 80)
    print("Diagnostic example: one candidate")
    print("=" * 80)
    print(f"history length: {len(history)}")
    print(f"candidate: {candidate}")
    print(f"p-value: {p_value:.4f}")
    print(f"anchor state i = candidate[-1]: {details['anchor_state']}")
    print(f"I0: {details['I0']}")
    print(f"number of permutable blocks: {len(details['blocks'])}")
    print(f"number of permutations used: {details['num_permutations_used']}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    demo_three_state_fully_connected()
    demo_sweden_like_short()
    demo_sweden_like_long(save_plots=True)
    demo_one_candidate_diagnostic()


if __name__ == "__main__":
    main()
