"""
Minimal Markov CP example.

Edit the training_sequence, horizon, alpha, or adjacency_matrix below to try a
different small forecasting experiment.
"""

import numpy as np

from markov_cp_routines import (
    cp_plus_one_prediction_set,
    iblock_cp_prediction_set,
    likelihood_prediction_set,
)


def main():
    training_sequence = [1, 2, 3, 1, 2, 3]
    horizon = 2
    alpha = 0.2
    max_permutations = 100

    adjacency_matrix = np.ones((3, 3), dtype=int)

    likelihood_set = likelihood_prediction_set(
        history=training_sequence,
        H=horizon,
        alpha=alpha,
        adjacency=adjacency_matrix,
    )

    iblock_set = iblock_cp_prediction_set(
        history=training_sequence,
        H=horizon,
        alpha=alpha,
        adjacency=adjacency_matrix,
        max_permutations=max_permutations,
    )

    cp_plus_one_set = cp_plus_one_prediction_set(
        history=training_sequence,
        H=horizon,
        alpha=alpha,
        adjacency=adjacency_matrix,
        anchor_state=1,
        max_permutations=max_permutations,
    )

    print("Minimal Markov CP example")
    print("-------------------------")
    print(f"training_sequence: {training_sequence}")
    print(f"horizon: {horizon}")
    print(f"alpha: {alpha}")
    print(f"adjacency_matrix:\n{adjacency_matrix}")
    print(f"likelihood prediction set: {likelihood_set}")
    print(f"original i-block CP set: {iblock_set}")
    print(f"CP+1 prediction set: {cp_plus_one_set}")


if __name__ == "__main__":
    main()
