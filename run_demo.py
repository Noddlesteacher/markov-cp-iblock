"""
run_demo.py

Small demo file for the Markov CP routines.

Edit this file to try a different adjacency matrix, training sequence,
forecast horizon, alpha level, or permutation cap. The reusable algorithm
functions are in markov_cp_routines.py.
"""

import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEMO_OUTPUT_DIR = PROJECT_DIR / "demo_outputs"
MPL_CONFIG_DIR = DEMO_OUTPUT_DIR / ".matplotlib"
FONT_CACHE_DIR = DEMO_OUTPUT_DIR / ".cache"

MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(FONT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from markov_cp_routines import (
    block_p_value_for_candidate,
    generate_i_block_permutations,
    print_prediction_set_summary,
    run_all_methods,
    split_i_blocks,
)


# ---------------------------------------------------------------------
# Example adjacency matrices
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


FULLY_CONNECTED_THREE_STATE = np.ones((3, 3), dtype=int)


# ---------------------------------------------------------------------
# Small printing and plotting helpers for the demo
# ---------------------------------------------------------------------

def summarize_results(title, results):
    """Print a compact summary of all three methods."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"H = {results['horizon']}, alpha = {results['alpha']}")
    print(f"number of candidate paths = {len(results['candidate_paths'])}")
    print_prediction_set_summary("Likelihood baseline", results["likelihood"])
    print_prediction_set_summary("Original i-block CP", results["original_iblock"])
    print_prediction_set_summary("CP+1", results["cp_plus_one"])


def plot_state_composition(composition, title, filename):
    """Save a simple bar plot of state proportions across forecast steps."""
    H, num_states = composition.shape
    x = np.arange(H)
    bottom = np.zeros(H)

    plt.figure(figsize=(7, 4))

    for state_index in range(num_states):
        values = composition[:, state_index]
        plt.bar(
            x,
            values,
            bottom=bottom,
            label=f"state {state_index + 1}",
        )
        bottom = bottom + values

    plt.xticks(x, [f"t+{h + 1}" for h in range(H)])
    plt.ylim(0, 1)
    plt.ylabel("proportion")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Demo 1: three-state fully connected graph
# ---------------------------------------------------------------------

def demo_three_state_fully_connected():
    """Small example where every transition is allowed."""
    adjacency = FULLY_CONNECTED_THREE_STATE
    history = [1, 1, 2, 3, 1, 2]
    H = 2
    alpha = 0.2
    max_permutations = 200
    seed = 123

    results = run_all_methods(
        history=history,
        horizon=H,
        alpha=alpha,
        adjacency=adjacency,
        max_permutations=max_permutations,
        random_seed=seed,
    )

    summarize_results("Demo 1: three-state fully connected Markov chain", results)


# ---------------------------------------------------------------------
# Demo 2: short four-state conflict graph history
# ---------------------------------------------------------------------

def demo_conflict_graph_short():
    """Short example using the four-state conflict graph."""
    adjacency = CONFLICT_ADJACENCY
    history = [1] * 6
    H = 3
    alpha = 0.2
    max_permutations = 200
    seed = 123

    results = run_all_methods(
        history=history,
        horizon=H,
        alpha=alpha,
        adjacency=adjacency,
        max_permutations=max_permutations,
        random_seed=seed,
    )

    summarize_results("Demo 2: short four-state conflict graph history", results)


# ---------------------------------------------------------------------
# Demo 3: longer four-state conflict graph history
# ---------------------------------------------------------------------

def demo_conflict_graph_long(save_plots=True):
    """Longer example using the same graph and repeated state-1 observations."""
    adjacency = CONFLICT_ADJACENCY
    history = [1] * 420
    H = 3
    alpha = 0.2
    max_permutations = 200
    seed = 123

    results = run_all_methods(
        history=history,
        horizon=H,
        alpha=alpha,
        adjacency=adjacency,
        max_permutations=max_permutations,
        random_seed=seed,
    )

    summarize_results("Demo 3: long four-state conflict graph history", results)

    if save_plots:
        out_dir = DEMO_OUTPUT_DIR
        out_dir.mkdir(exist_ok=True)

        plot_state_composition(
            results["state_composition"]["likelihood"],
            "Likelihood baseline",
            out_dir / "long_conflict_likelihood.png",
        )
        plot_state_composition(
            results["state_composition"]["original_iblock"],
            "Original i-block CP",
            out_dir / "long_conflict_iblock.png",
        )
        plot_state_composition(
            results["state_composition"]["cp_plus_one"],
            "CP+1",
            out_dir / "long_conflict_cp_plus_one.png",
        )

        print(f"Saved plots to: {out_dir.resolve()}")


# ---------------------------------------------------------------------
# Optional diagnostic: one candidate p-value
# ---------------------------------------------------------------------

def demo_one_candidate_diagnostic():
    """Show the i-block pieces used for one candidate path."""
    adjacency = CONFLICT_ADJACENCY
    history = [1] * 6
    candidate = (1, 1, 2)
    max_permutations = 50
    seed = 123

    random.seed(seed)
    p_value = block_p_value_for_candidate(
        history=history,
        candidate=candidate,
        adjacency=adjacency,
        max_permutations=max_permutations,
    )

    aug_seq = list(history) + list(candidate)
    i = candidate[-1]
    I0, blocks, tail = split_i_blocks(aug_seq, i)
    permuted_sequences = generate_i_block_permutations(
        I0,
        blocks,
        tail,
        max_permutations=max_permutations,
    )

    print("\n" + "=" * 80)
    print("Diagnostic example: one candidate")
    print("=" * 80)
    print(f"history length: {len(history)}")
    print(f"candidate: {candidate}")
    print(f"p-value: {p_value:.4f}")
    print(f"anchor state i = candidate[-1]: {i}")
    print(f"I0: {I0}")
    print(f"number of permutable blocks: {len(blocks)}")
    print(f"tail: {tail}")
    print(f"number of permutations used: {len(permuted_sequences)}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    demo_three_state_fully_connected()
    demo_conflict_graph_short()
    demo_conflict_graph_long(save_plots=True)
    demo_one_candidate_diagnostic()


if __name__ == "__main__":
    main()
