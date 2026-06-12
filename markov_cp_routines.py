"""
markov_cp_routines.py

Core routines for likelihood-based prediction, original i-block conformal
prediction, and the graph-constrained CP+1 variant for finite-state Markov
chains.

State labels are assumed to be 1, 2, ..., S, where S is inferred from the
number of rows in the adjacency matrix.
"""

import math
import random
from itertools import permutations

import numpy as np


# ---------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------

def check_adjacency(adjacency):
    """Return adjacency as a checked NumPy array."""
    adjacency = np.asarray(adjacency, dtype=int)

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix.")

    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError("adjacency must contain only 0/1 entries.")

    if np.any(adjacency.sum(axis=1) == 0):
        raise ValueError("each state needs at least one allowed next state.")

    return adjacency


def allowed_next(state, adjacency):
    """Return all states that can follow the current state."""
    adjacency = check_adjacency(adjacency)
    num_states = adjacency.shape[0]

    if state < 1 or state > num_states:
        raise ValueError(f"state must be between 1 and {num_states}.")

    row = adjacency[state - 1]
    next_states = []

    for j in range(num_states):
        if row[j] == 1:
            next_states.append(j + 1)

    return next_states


def is_allowed_transition(a, b, adjacency):
    """Return True if the transition a -> b is allowed."""
    adjacency = check_adjacency(adjacency)
    num_states = adjacency.shape[0]

    if a < 1 or a > num_states or b < 1 or b > num_states:
        return False

    return adjacency[a - 1, b - 1] == 1


def check_sequence(seq, adjacency):
    """Check that a sequence uses valid states and allowed transitions."""
    adjacency = check_adjacency(adjacency)
    num_states = adjacency.shape[0]

    if len(seq) == 0:
        raise ValueError("sequence must be nonempty.")

    for state in seq:
        if state < 1 or state > num_states:
            raise ValueError(f"state must be between 1 and {num_states}.")

    for t in range(len(seq) - 1):
        if not is_allowed_transition(seq[t], seq[t + 1], adjacency):
            raise ValueError(f"forbidden transition: {seq[t]} -> {seq[t + 1]}")


# ---------------------------------------------------------------------
# Candidate paths and likelihood baseline
# ---------------------------------------------------------------------

def enumerate_paths(start_state, H, adjacency):
    """Enumerate all allowed future paths of length H."""
    if H == 0:
        return [()]

    paths = []

    for next_state in allowed_next(start_state, adjacency):
        subpaths = enumerate_paths(next_state, H - 1, adjacency)

        for subpath in subpaths:
            new_path = (next_state,) + subpath
            paths.append(new_path)

    return paths


def transition_counts(seq):
    """Count transitions in a sequence."""
    counts = {}

    for t in range(len(seq) - 1):
        a = seq[t]
        b = seq[t + 1]
        counts[(a, b)] = counts.get((a, b), 0) + 1

    return counts


def estimate_transition_matrix(seq, adjacency):
    """Estimate the transition matrix from transition counts."""
    adjacency = check_adjacency(adjacency)
    check_sequence(seq, adjacency)

    num_states = adjacency.shape[0]
    counts = transition_counts(seq)

    P = np.zeros((num_states, num_states))
    row_totals = np.zeros(num_states)

    for (a, b), count in counts.items():
        row_totals[a - 1] += count

    for (a, b), count in counts.items():
        if row_totals[a - 1] > 0:
            P[a - 1, b - 1] = count / row_totals[a - 1]

    return P


def path_probability(start_state, path, P):
    """Compute the estimated probability of one candidate path.

    NOTE:
    This function computes the product of transition probabilities along a path,
    conditional on the first/current state, which is passed here as start_state.
    It does not multiply by the initial state probability pi[start_state].
    Therefore, it is not intended for computing the full joint probability of
    an entire Markov chain path unless the initial probability is included
    separately. In particular, this does not handle the special case where the
    training data length is one and an initial state probability is required.
    """
    prob = 1.0
    current = start_state

    for next_state in path:
        prob = prob * P[current - 1, next_state - 1]
        current = next_state

    return prob


def path_probabilities(history, H, adjacency):
    """Compute estimated probabilities for all allowed candidate paths."""
    check_sequence(history, adjacency)

    P = estimate_transition_matrix(history, adjacency)
    start_state = history[-1]

    paths = enumerate_paths(start_state, H, adjacency)
    paths_probs = []

    for path in paths:
        prob = path_probability(start_state, path, P)
        paths_probs.append((path, prob))

    paths_probs_sorted = sorted(
        paths_probs,
        key=lambda item: item[1],
        reverse=True,
    )

    return paths_probs_sorted


def likelihood_prediction_set(history, H, alpha, adjacency):
    """Return the likelihood prediction set with target coverage 1 - alpha."""
    path_probs_sorted = path_probabilities(history, H, adjacency)

    selected_paths = []
    cum_prob = 0.0
    target_coverage = 1 - alpha

    for path, prob in path_probs_sorted:
        selected_paths.append(path)
        cum_prob += prob

        if cum_prob >= target_coverage:
            break

    return selected_paths


# ---------------------------------------------------------------------
# i-block conformal prediction
# ---------------------------------------------------------------------

def split_i_blocks(seq, i):
    """Split a sequence into I0, permutable i-blocks, and the final tail."""
    positions = []

    for idx, state in enumerate(seq):
        if state == i:
            positions.append(idx)

    if len(positions) == 0:
        raise ValueError("State i does not appear in the sequence.")

    if positions[0] == 0:
        I0 = None
    else:
        I0 = seq[:positions[0]]

    all_blocks = []

    for k in range(len(positions)):
        start = positions[k]

        if k < len(positions) - 1:
            end = positions[k + 1]
            block = seq[start:end]
        else:
            block = seq[start:]

        all_blocks.append(block)

    tail = all_blocks[-1]
    blocks = all_blocks[:-1]

    return I0, blocks, tail


def build_sequence_from_blocks(I0, ordered_blocks, tail):
    """Build one full sequence from I0, ordered i-blocks, and tail."""
    seq = []

    if I0 is not None:
        seq.extend(I0)

    for block in ordered_blocks:
        seq.extend(block)

    seq.extend(tail)

    return seq


def generate_i_block_permutations(I0, blocks, tail=None, max_permutations=None):
    """Generate or sample i-block permutations.

    Typical internal use:
        generate_i_block_permutations(I0, blocks, tail, max_permutations)

    Convenience use for quick checks:
        generate_i_block_permutations(sequence, i, max_permutations=None)

    For small examples, max_permutations can be None and all permutations are
    generated. For larger examples, max_permutations samples shuffled block
    orders directly instead of first building every factorial permutation.
    """
    if tail is None and isinstance(blocks, (int, np.integer)):
        I0, blocks, tail = split_i_blocks(I0, int(blocks))

    if tail is None:
        raise ValueError("tail must be provided unless calling with sequence and i.")

    if max_permutations is None:
        number_of_permutations = math.factorial(len(blocks))
        if number_of_permutations > 50000:
            raise ValueError("too many permutations; set max_permutations.")

        seen_sequences = set()
        permuted_sequences = []

        for ordered_blocks in permutations(blocks):
            seq_perm = build_sequence_from_blocks(I0, ordered_blocks, tail)
            seq_key = tuple(seq_perm)

            if seq_key not in seen_sequences:
                seen_sequences.add(seq_key)
                permuted_sequences.append(seq_perm)

        return permuted_sequences
    else:
        if max_permutations <= 0:
            return []

        seen_sequences = set()
        permuted_sequences = []

        identity_sequence = build_sequence_from_blocks(I0, blocks, tail)
        identity_key = tuple(identity_sequence)
        seen_sequences.add(identity_key)
        permuted_sequences.append(identity_sequence)

        max_attempts = max(100, max_permutations * 20)
        attempts = 0

        while len(permuted_sequences) < max_permutations and attempts < max_attempts:
            order = list(range(len(blocks)))
            random.shuffle(order)
            ordered_blocks = [blocks[k] for k in order]

            seq_perm = build_sequence_from_blocks(I0, ordered_blocks, tail)
            seq_key = tuple(seq_perm)

            if seq_key not in seen_sequences:
                seen_sequences.add(seq_key)
                permuted_sequences.append(seq_perm)

            attempts += 1

        return permuted_sequences


def score_sequence(seq_perm, T, H, P_hat):
    """Compute the nonconformity score from Algorithm 1."""
    state_T = seq_perm[T - 1]

    sum_prob = 0.0

    for j in range(1, H + 1):
        future_state = seq_perm[T + j - 1]

        P_power = np.linalg.matrix_power(P_hat, j)
        prob = P_power[state_T - 1, future_state - 1]

        sum_prob += prob

    average_prob = sum_prob / H
    score = 1 - average_prob

    return score


def block_p_value_for_candidate(history, candidate, adjacency, max_permutations=None):
    """Compute the i-block conformal p-value for one candidate path."""
    check_sequence(history, adjacency)
    check_sequence([history[-1]] + list(candidate), adjacency)

    T = len(history)
    H = len(candidate)

    aug_seq = list(history) + list(candidate)

    P_hat = estimate_transition_matrix(aug_seq, adjacency)

    i = candidate[-1]

    I0, blocks, tail = split_i_blocks(aug_seq, i)

    permuted_sequences = generate_i_block_permutations(
        I0,
        blocks,
        tail,
        max_permutations=max_permutations,
    )

    S_identity = score_sequence(aug_seq, T, H, P_hat)

    scores = []
    for seq_perm in permuted_sequences:
        S_perm = score_sequence(seq_perm, T, H, P_hat)
        scores.append(S_perm)

    tol = 1e-12
    num_greater = sum(S > S_identity + tol for S in scores)
    num_equal = sum(abs(S - S_identity) <= tol for S in scores)

    u = random.random()

    p_value = (num_greater + u * num_equal) / len(scores)

    return p_value


def iblock_cp_prediction_set(history, H, alpha, adjacency, max_permutations=None):
    """Return the original i-block conformal prediction set."""
    start_state = history[-1]

    candidate_paths = enumerate_paths(start_state, H, adjacency)
    C_cp = []

    for candidate in candidate_paths:
        p_value = block_p_value_for_candidate(
            history,
            candidate,
            adjacency,
            max_permutations=max_permutations,
        )

        if p_value > alpha:
            C_cp.append(candidate)

    return C_cp


# ---------------------------------------------------------------------
# CP+1
# ---------------------------------------------------------------------

def extend_candidate_plus_one(candidate, anchor_state=1):
    """Append one artificial anchor state to a candidate path."""
    return tuple(candidate) + (anchor_state,)


def cp_plus_one_p_value_for_candidate(
    history,
    candidate,
    adjacency,
    anchor_state=1,
    max_permutations=None,
):
    """Compute the CP+1 p-value for one original candidate path."""
    if not is_allowed_transition(candidate[-1], anchor_state, adjacency):
        return 0.0

    extended_candidate = extend_candidate_plus_one(
        candidate,
        anchor_state=anchor_state,
    )

    p_value = block_p_value_for_candidate(
        history,
        extended_candidate,
        adjacency,
        max_permutations=max_permutations,
    )

    return p_value


def cp_plus_one_prediction_set(
    history,
    H,
    alpha,
    adjacency,
    anchor_state=1,
    max_permutations=None,
):
    """Return the graph-constrained CP+1 prediction set."""
    start_state = history[-1]

    candidate_paths = enumerate_paths(start_state, H, adjacency)
    C_cp_plus_one = []

    for candidate in candidate_paths:
        p_value = cp_plus_one_p_value_for_candidate(
            history,
            candidate,
            adjacency,
            anchor_state=anchor_state,
            max_permutations=max_permutations,
        )

        if p_value > alpha:
            C_cp_plus_one.append(candidate)

    return C_cp_plus_one


# ---------------------------------------------------------------------
# Small summary helpers used by run_demo.py
# ---------------------------------------------------------------------

def state_composition(C, H, adjacency):
    """Compute proportions of states at each future time."""
    adjacency = check_adjacency(adjacency)
    num_states = adjacency.shape[0]

    composition = np.zeros((H, num_states))

    if len(C) == 0:
        return composition

    for path in C:
        for h, state in enumerate(path):
            composition[h, state - 1] += 1

    composition = composition / len(C)

    return composition


def run_all_methods(
    history,
    horizon,
    alpha,
    adjacency,
    max_permutations=None,
    random_seed=123,
    cp_plus_one_anchor=1,
):
    """Run likelihood, original i-block CP, and CP+1 for one setup."""
    random.seed(random_seed)

    H = horizon

    C_like = likelihood_prediction_set(history, H, alpha, adjacency)
    C_cp = iblock_cp_prediction_set(
        history,
        H,
        alpha,
        adjacency,
        max_permutations=max_permutations,
    )
    C_plus_one = cp_plus_one_prediction_set(
        history,
        H,
        alpha,
        adjacency,
        anchor_state=cp_plus_one_anchor,
        max_permutations=max_permutations,
    )

    candidate_paths = enumerate_paths(history[-1], H, adjacency)

    results = {
        "history": list(history),
        "horizon": H,
        "alpha": alpha,
        "adjacency": check_adjacency(adjacency),
        "candidate_paths": candidate_paths,
        "likelihood": C_like,
        "original_iblock": C_cp,
        "cp_plus_one": C_plus_one,
        "state_composition": {
            "likelihood": state_composition(C_like, H, adjacency),
            "original_iblock": state_composition(C_cp, H, adjacency),
            "cp_plus_one": state_composition(C_plus_one, H, adjacency),
        },
    }

    return results


def print_prediction_set_summary(name, C):
    """Print a compact prediction-set summary."""
    print(f"{name}: size = {len(C)}")

    if len(C) <= 20:
        print(f"  paths = {C}")
    else:
        print(f"  first 10 paths = {C[:10]}")
