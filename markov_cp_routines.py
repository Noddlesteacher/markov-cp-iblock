"""
markov_cp_routines.py

Reusable routines for likelihood-based prediction sets, original i-block
conformal prediction, and CP+1 for discrete-state Markov chains.

State labels are assumed to be 1, 2, ..., S, where
S = number of rows/columns in the adjacency matrix.
"""

from __future__ import annotations

import math
import random
from itertools import permutations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


State = int
Path = Tuple[State, ...]
SequenceLike = Sequence[State]


# ---------------------------------------------------------------------
# Basic validation and graph utilities
# ---------------------------------------------------------------------

def as_adjacency(adjacency: Union[np.ndarray, Sequence[Sequence[int]]]) -> np.ndarray:
    """Convert an input adjacency matrix to a checked square NumPy array.

    Parameters
    ----------
    adjacency:
        Square S x S matrix. adjacency[a-1, b-1] = 1 means a -> b is allowed.

    Returns
    -------
    np.ndarray
        Integer-valued S x S adjacency matrix.
    """
    A = np.asarray(adjacency, dtype=int)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix.")
    if not np.all((A == 0) | (A == 1)):
        raise ValueError("adjacency must contain only 0/1 entries.")
    if np.any(A.sum(axis=1) == 0):
        raise ValueError("each state must have at least one allowed outgoing transition.")
    return A


def num_states(adjacency: Union[np.ndarray, Sequence[Sequence[int]]]) -> int:
    """Return the number of states implied by the adjacency matrix."""
    return as_adjacency(adjacency).shape[0]


def check_state(state: State, adjacency: Union[np.ndarray, Sequence[Sequence[int]]]) -> None:
    """Raise an error if `state` is not in {1, ..., S}."""
    S = num_states(adjacency)
    if not isinstance(state, (int, np.integer)) or not (1 <= int(state) <= S):
        raise ValueError(f"state must be an integer in {{1, ..., {S}}}; got {state}.")


def allowed_next(state: State, adjacency: Union[np.ndarray, Sequence[Sequence[int]]]) -> List[State]:
    """List all states b such that state -> b is allowed."""
    A = as_adjacency(adjacency)
    check_state(state, A)
    row = A[int(state) - 1]
    return [j + 1 for j, allowed in enumerate(row) if allowed == 1]


def is_allowed_transition(
    a: State,
    b: State,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
) -> bool:
    """Return True if transition a -> b is allowed by the adjacency matrix."""
    A = as_adjacency(adjacency)
    check_state(a, A)
    check_state(b, A)
    return bool(A[int(a) - 1, int(b) - 1])


def validate_sequence(
    seq: SequenceLike,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    require_allowed_transitions: bool = True,
) -> None:
    """Validate state labels and, optionally, all transitions in a sequence."""
    A = as_adjacency(adjacency)
    if len(seq) == 0:
        raise ValueError("sequence must be nonempty.")
    for state in seq:
        check_state(state, A)
    if require_allowed_transitions:
        for a, b in zip(seq[:-1], seq[1:]):
            if not is_allowed_transition(a, b, A):
                raise ValueError(f"forbidden transition encountered: {a} -> {b}.")


# ---------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------

def enumerate_paths(
    start_state: State,
    horizon: int,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
) -> List[Path]:
    """Enumerate all legal future paths of length `horizon`.

    Parameters
    ----------
    start_state:
        Current state X_T.
    horizon:
        Number of future states to forecast.
    adjacency:
        Square adjacency matrix defining allowed transitions.

    Returns
    -------
    list[tuple[int, ...]]
        All legal paths (x_{T+1}, ..., x_{T+horizon}).
    """
    A = as_adjacency(adjacency)
    check_state(start_state, A)
    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")
    if horizon == 0:
        return [()]

    paths: List[Path] = []
    for next_state in allowed_next(start_state, A):
        for subpath in enumerate_paths(next_state, horizon - 1, A):
            paths.append((next_state,) + subpath)
    return paths


# ---------------------------------------------------------------------
# Transition counts and transition-matrix estimation
# ---------------------------------------------------------------------

def transition_counts(seq: SequenceLike) -> Dict[Tuple[State, State], int]:
    """Count transitions a -> b in a state sequence."""
    if len(seq) < 2:
        return {}
    counts: Dict[Tuple[State, State], int] = {}
    for a, b in zip(seq[:-1], seq[1:]):
        counts[(int(a), int(b))] = counts.get((int(a), int(b)), 0) + 1
    return counts


def estimate_transition_matrix(
    seq: SequenceLike,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    require_allowed_transitions: bool = True,
) -> np.ndarray:
    """Estimate the Markov transition matrix by row-normalized counts.

    This is the unsmoothed MLE used in the likelihood baseline and in
    Algorithm 1 when it estimates P_hat from the augmented sequence.

    Rows with zero observed outgoing transitions are left as all zeros.
    That behavior matches the simple count-based MLE and avoids adding
    any extra smoothing parameter.
    """
    A = as_adjacency(adjacency)
    validate_sequence(seq, A, require_allowed_transitions=require_allowed_transitions)

    S = A.shape[0]
    P_hat = np.zeros((S, S), dtype=float)
    row_totals = np.zeros(S, dtype=float)

    counts = transition_counts(seq)
    for (a, b), count in counts.items():
        row_totals[a - 1] += count

    for (a, b), count in counts.items():
        if row_totals[a - 1] > 0:
            P_hat[a - 1, b - 1] = count / row_totals[a - 1]

    return P_hat


# ---------------------------------------------------------------------
# Likelihood baseline
# ---------------------------------------------------------------------

def path_probability(start_state: State, path: Path, P_hat: np.ndarray) -> float:
    """Compute product_j P_hat[x_{j-1}, x_j] for a candidate path."""
    prob = 1.0
    current = int(start_state)
    for next_state in path:
        prob *= float(P_hat[current - 1, int(next_state) - 1])
        current = int(next_state)
    return prob


def likelihood_path_probabilities(
    history: SequenceLike,
    horizon: int,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
) -> List[Tuple[Path, float]]:
    """Compute and sort likelihood probabilities for all legal candidate paths."""
    A = as_adjacency(adjacency)
    validate_sequence(history, A, require_allowed_transitions=True)

    P_hat = estimate_transition_matrix(history, A)
    start_state = int(history[-1])
    candidate_paths = enumerate_paths(start_state, horizon, A)

    path_probs = [
        (path, path_probability(start_state, path, P_hat))
        for path in candidate_paths
    ]
    return sorted(path_probs, key=lambda item: item[1], reverse=True)


def likelihood_prediction_set(
    history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
) -> List[Path]:
    """Highest-probability-mass likelihood prediction set.

    Add candidate paths from highest to lowest estimated probability until
    cumulative probability reaches 1 - alpha. If the estimated mass is less
    than 1 because some rows of P_hat are unobserved, this function simply
    returns all candidate paths after exhausting the list.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")

    sorted_probs = likelihood_path_probabilities(history, horizon, adjacency)
    target = 1.0 - alpha

    selected: List[Path] = []
    cumulative = 0.0
    for path, prob in sorted_probs:
        selected.append(path)
        cumulative += prob
        if cumulative >= target:
            break
    return selected


# ---------------------------------------------------------------------
# i-block construction
# ---------------------------------------------------------------------

def split_i_blocks(seq: SequenceLike, anchor_state: State) -> Tuple[Optional[List[State]], List[List[State]], List[State]]:
    """Split a sequence into I0, permutable i-blocks, and a terminal tail block.

    An i-block starts at an occurrence of `anchor_state` and continues up to,
    but not including, the next occurrence of `anchor_state`. The last block is
    treated as the non-permutable terminal tail.
    """
    seq_list = [int(s) for s in seq]
    positions = [idx for idx, state in enumerate(seq_list) if state == int(anchor_state)]

    if len(positions) == 0:
        raise ValueError("anchor_state does not appear in the sequence.")

    I0 = None if positions[0] == 0 else seq_list[:positions[0]]

    all_blocks: List[List[State]] = []
    for k, start in enumerate(positions):
        if k < len(positions) - 1:
            end = positions[k + 1]
            block = seq_list[start:end]
        else:
            block = seq_list[start:]
        all_blocks.append(block)

    tail_block = all_blocks[-1]
    permutable_blocks = all_blocks[:-1]
    return I0, permutable_blocks, tail_block


def build_sequence_from_blocks(
    I0: Optional[SequenceLike],
    ordered_blocks: Sequence[SequenceLike],
    tail: SequenceLike,
) -> List[State]:
    """Reconstruct a sequence from fixed prefix, ordered blocks, and tail."""
    seq: List[State] = []
    if I0 is not None:
        seq.extend(int(s) for s in I0)
    for block in ordered_blocks:
        seq.extend(int(s) for s in block)
    seq.extend(int(s) for s in tail)
    return seq


def generate_i_block_permutations(
    I0: Optional[SequenceLike],
    blocks: Sequence[SequenceLike],
    tail: SequenceLike,
    max_permutations: Optional[int] = None,
    rng: Optional[random.Random] = None,
    max_full_enumeration: int = 50_000,
) -> List[List[State]]:
    """Generate full or sampled i-block permutations.

    The identity ordering is always included. If max_permutations is None and
    the number of full permutations is too large, an error is raised to avoid
    accidentally trying to materialize a huge factorial object.
    """
    rng = rng or random.Random()
    blocks = [list(block) for block in blocks]
    n_blocks = len(blocks)

    if n_blocks == 0:
        return [build_sequence_from_blocks(I0, [], tail)]

    # If no cap is provided, enumerate all permutations only when safe.
    if max_permutations is None:
        total_perms = math.factorial(n_blocks)
        if total_perms > max_full_enumeration:
            raise ValueError(
                f"{n_blocks}! permutations is too large. "
                "Pass max_permutations to sample permutations."
            )
        return [
            build_sequence_from_blocks(I0, ordered_blocks, tail)
            for ordered_blocks in permutations(blocks)
        ]

    if max_permutations <= 0:
        raise ValueError("max_permutations must be positive or None.")

    # If full enumeration is small enough and below the requested cap, do it.
    if n_blocks <= 10:
        total_perms = math.factorial(n_blocks)
        if total_perms <= max_permutations:
            return [
                build_sequence_from_blocks(I0, ordered_blocks, tail)
                for ordered_blocks in permutations(blocks)
            ]

    # Otherwise sample unique block orderings without replacement.
    permuted_sequences: List[List[State]] = []
    seen_orders = set()

    # Always include identity.
    identity_order = tuple(range(n_blocks))
    seen_orders.add(identity_order)
    permuted_sequences.append(build_sequence_from_blocks(I0, blocks, tail))

    max_attempts = max(100, max_permutations * 100)
    attempts = 0

    while len(permuted_sequences) < max_permutations and attempts < max_attempts:
        order = list(range(n_blocks))
        rng.shuffle(order)
        order_tuple = tuple(order)

        if order_tuple not in seen_orders:
            seen_orders.add(order_tuple)
            ordered_blocks = [blocks[k] for k in order]
            permuted_sequences.append(build_sequence_from_blocks(I0, ordered_blocks, tail))

        attempts += 1

    return permuted_sequences


# ---------------------------------------------------------------------
# Nonconformity score and original i-block CP
# ---------------------------------------------------------------------

def _transition_matrix_powers(P_hat: np.ndarray, horizon: int) -> List[np.ndarray]:
    """Return [P_hat^1, ..., P_hat^horizon] using iterative multiplication."""
    powers: List[np.ndarray] = []
    current = np.eye(P_hat.shape[0])
    for _ in range(horizon):
        current = current @ P_hat
        powers.append(current.copy())
    return powers


def score_sequence(
    seq_perm: SequenceLike,
    T: int,
    horizon: int,
    P_hat: np.ndarray,
    powers: Optional[List[np.ndarray]] = None,
) -> float:
    """Compute the Algorithm-1 nonconformity score.

    S(pi) = 1 - (1/horizon) * sum_{j=1}^horizon
             P_hat^j[ X_T^(pi), X_{T+j}^(pi) ]

    Larger score means more nonconforming.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive for scoring.")
    if len(seq_perm) < T + horizon:
        raise ValueError("seq_perm is shorter than T + horizon.")

    powers = powers or _transition_matrix_powers(P_hat, horizon)

    state_T = int(seq_perm[T - 1])
    sum_prob = 0.0
    for j in range(1, horizon + 1):
        future_state = int(seq_perm[T + j - 1])
        prob = powers[j - 1][state_T - 1, future_state - 1]
        sum_prob += float(prob)

    return 1.0 - (sum_prob / horizon)


def block_p_value_for_candidate(
    history: SequenceLike,
    candidate: Path,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    max_permutations: Optional[int] = None,
    rng: Optional[random.Random] = None,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, object]]]:
    """Compute the randomized i-block p-value for one candidate path.

    Parameters
    ----------
    history:
        Observed calibration sequence X_1, ..., X_T.
    candidate:
        Proposed future path x_{T+1}, ..., x_{T+H}.
    adjacency:
        Allowed-transition matrix.
    max_permutations:
        Maximum number of sampled i-block permutations. Identity is included.
        Use None only when the full permutation group is small.
    rng:
        Python random.Random object for reproducible sampling and tie-breaking.
    return_details:
        If True, also return useful diagnostic information.
    """
    A = as_adjacency(adjacency)
    rng = rng or random.Random()

    history_list = [int(s) for s in history]
    candidate_tuple = tuple(int(s) for s in candidate)
    validate_sequence(history_list, A, require_allowed_transitions=True)
    validate_sequence(history_list[-1:] + list(candidate_tuple), A, require_allowed_transitions=True)

    T = len(history_list)
    horizon = len(candidate_tuple)
    if horizon <= 0:
        raise ValueError("candidate must have positive length.")

    aug_seq = history_list + list(candidate_tuple)
    P_hat = estimate_transition_matrix(aug_seq, A)

    anchor_state = candidate_tuple[-1]
    I0, blocks, tail = split_i_blocks(aug_seq, anchor_state)
    permuted_sequences = generate_i_block_permutations(
        I0,
        blocks,
        tail,
        max_permutations=max_permutations,
        rng=rng,
    )

    powers = _transition_matrix_powers(P_hat, horizon)
    S_identity = score_sequence(aug_seq, T, horizon, P_hat, powers=powers)
    scores = [score_sequence(seq_perm, T, horizon, P_hat, powers=powers)
              for seq_perm in permuted_sequences]

    tol = 1e-12
    num_greater = sum(S > S_identity + tol for S in scores)
    num_equal = sum(abs(S - S_identity) <= tol for S in scores)

    u = rng.random()
    p_value = (num_greater + u * num_equal) / len(scores)

    if not return_details:
        return p_value

    details: Dict[str, object] = {
        "augmented_sequence": aug_seq,
        "P_hat": P_hat,
        "anchor_state": anchor_state,
        "I0": I0,
        "blocks": blocks,
        "tail": tail,
        "num_permutations_used": len(permuted_sequences),
        "S_identity": S_identity,
        "scores": scores,
        "num_greater": num_greater,
        "num_equal": num_equal,
        "tie_break_u": u,
    }
    return p_value, details


def iblock_cp_prediction_set(
    history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    max_permutations: Optional[int] = None,
    seed: Optional[int] = None,
    return_p_values: bool = False,
) -> Union[List[Path], Tuple[List[Path], Dict[Path, float]]]:
    """Original i-block conformal prediction set."""
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")

    A = as_adjacency(adjacency)
    validate_sequence(history, A, require_allowed_transitions=True)
    rng = random.Random(seed)

    candidate_paths = enumerate_paths(int(history[-1]), horizon, A)
    prediction_set: List[Path] = []
    p_values: Dict[Path, float] = {}

    for candidate in candidate_paths:
        p_value = float(block_p_value_for_candidate(
            history,
            candidate,
            A,
            max_permutations=max_permutations,
            rng=rng,
        ))
        p_values[candidate] = p_value
        if p_value > alpha:
            prediction_set.append(candidate)

    if return_p_values:
        return prediction_set, p_values
    return prediction_set


# ---------------------------------------------------------------------
# CP+1 naive baseline
# ---------------------------------------------------------------------

def extend_candidate_with_anchor(candidate: Path, anchor_state: State = 1) -> Path:
    """Append a fixed anchor state to a candidate path."""
    return tuple(int(s) for s in candidate) + (int(anchor_state),)


def cp_plus_one_p_value_for_candidate(
    history: SequenceLike,
    candidate: Path,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    anchor_state: State = 1,
    max_permutations: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> float:
    """Compute p-value for CP+1 by appending a fixed anchor state.

    If candidate[-1] -> anchor_state is forbidden, return 0.0.
    """
    A = as_adjacency(adjacency)
    rng = rng or random.Random()

    if len(candidate) == 0:
        raise ValueError("candidate must be nonempty.")
    check_state(anchor_state, A)

    if not is_allowed_transition(candidate[-1], anchor_state, A):
        return 0.0

    extended_candidate = extend_candidate_with_anchor(candidate, anchor_state=anchor_state)
    return float(block_p_value_for_candidate(
        history,
        extended_candidate,
        A,
        max_permutations=max_permutations,
        rng=rng,
    ))


def cp_plus_one_prediction_set(
    history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    anchor_state: State = 1,
    max_permutations: Optional[int] = None,
    seed: Optional[int] = None,
    return_p_values: bool = False,
) -> Union[List[Path], Tuple[List[Path], Dict[Path, float]]]:
    """Graph-constrained CP+1 prediction set.

    The returned paths have original length `horizon`; the artificial anchor
    is only used internally to compute the p-value.
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1.")

    A = as_adjacency(adjacency)
    validate_sequence(history, A, require_allowed_transitions=True)
    rng = random.Random(seed)

    candidate_paths = enumerate_paths(int(history[-1]), horizon, A)
    prediction_set: List[Path] = []
    p_values: Dict[Path, float] = {}

    for candidate in candidate_paths:
        p_value = cp_plus_one_p_value_for_candidate(
            history,
            candidate,
            A,
            anchor_state=anchor_state,
            max_permutations=max_permutations,
            rng=rng,
        )
        p_values[candidate] = p_value
        if p_value > alpha:
            prediction_set.append(candidate)

    if return_p_values:
        return prediction_set, p_values
    return prediction_set


# ---------------------------------------------------------------------
# Summaries and plotting
# ---------------------------------------------------------------------

def state_composition(
    prediction_set: Sequence[Path],
    horizon: int,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
) -> np.ndarray:
    """Compute state proportions at each forecast horizon.

    Returns an array with shape (horizon, num_states).
    """
    A = as_adjacency(adjacency)
    S = A.shape[0]
    composition = np.zeros((horizon, S), dtype=float)

    if len(prediction_set) == 0:
        return composition

    for path in prediction_set:
        if len(path) != horizon:
            raise ValueError("all paths in prediction_set must have length equal to horizon.")
        for h, state in enumerate(path):
            check_state(state, A)
            composition[h, int(state) - 1] += 1

    return composition / len(prediction_set)


def print_prediction_set_summary(name: str, prediction_set: Sequence[Path]) -> None:
    """Print a small summary for a prediction set."""
    print(f"{name}: size = {len(prediction_set)}")
    if len(prediction_set) <= 20:
        print(f"  paths = {list(prediction_set)}")
    else:
        print(f"  first 10 paths = {list(prediction_set)[:10]}")


def run_all_methods(
    history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Union[np.ndarray, Sequence[Sequence[int]]],
    max_permutations: Optional[int] = 200,
    cp_plus_one_anchor: State = 1,
    random_seed: Optional[int] = 123,
) -> Dict[str, object]:
    """Convenience wrapper to run likelihood, original i-block CP, and CP+1."""
    A = as_adjacency(adjacency)
    validate_sequence(history, A, require_allowed_transitions=True)

    C_like = likelihood_prediction_set(history, horizon, alpha, A)
    C_cp, pvals_cp = iblock_cp_prediction_set(
        history,
        horizon,
        alpha,
        A,
        max_permutations=max_permutations,
        seed=random_seed,
        return_p_values=True,
    )
    C_plus, pvals_plus = cp_plus_one_prediction_set(
        history,
        horizon,
        alpha,
        A,
        anchor_state=cp_plus_one_anchor,
        max_permutations=max_permutations,
        seed=random_seed,
        return_p_values=True,
    )

    return {
        "history": list(history),
        "horizon": horizon,
        "alpha": alpha,
        "adjacency": A,
        "random_seed": random_seed,
        "candidate_paths": enumerate_paths(int(history[-1]), horizon, A),
        "likelihood": C_like,
        "original_iblock": C_cp,
        "cp_plus_one": C_plus,
        "original_iblock_p_values": pvals_cp,
        "cp_plus_one_p_values": pvals_plus,
        "state_composition": {
            "likelihood": state_composition(C_like, horizon, A),
            "original_iblock": state_composition(C_cp, horizon, A),
            "cp_plus_one": state_composition(C_plus, horizon, A),
        },
    }


def plot_state_composition(
    composition: np.ndarray,
    title: str,
    state_labels: Optional[Sequence[str]] = None,
    filename: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot stacked state-composition bars for one prediction set.

    Matplotlib is imported inside the function so that the routine file can be
    used on servers without plotting unless this function is called.
    """
    import matplotlib.pyplot as plt

    if composition.ndim != 2:
        raise ValueError("composition must be a 2D array of shape (horizon, num_states).")

    horizon, S = composition.shape
    horizons = np.arange(1, horizon + 1)
    if state_labels is None:
        state_labels = [f"State {s}" for s in range(1, S + 1)]

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(horizon)

    for s in range(S):
        plt.bar(
            horizons,
            composition[:, s],
            bottom=bottom,
            label=state_labels[s],
        )
        bottom += composition[:, s]

    plt.xlabel("Forecast horizon")
    plt.ylabel("Proportion")
    plt.title(title)
    plt.xticks(horizons)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_method_compositions(
    results: Dict[str, object],
    filename_prefix: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot state compositions for likelihood, original i-block, and CP+1."""
    comps = results["state_composition"]
    titles = {
        "likelihood": "Likelihood baseline",
        "original_iblock": "Original i-block CP",
        "cp_plus_one": "CP+1",
    }
    for key, title in titles.items():
        filename = None
        if filename_prefix is not None:
            filename = f"{filename_prefix}_{key}.png"
        plot_state_composition(comps[key], title=title, filename=filename, show=show)
