"""
Reusable routines for finite-state Markov-chain i-block conformal experiments.

The active research workflow for this branch is intentionally small:
candidate diagnostics for the original i-block procedure, plus a dense
three-state cardinality-weighted auxiliary-state experiment.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from itertools import permutations
import math
import random
from typing import Sequence

import numpy as np


State = int
Path = tuple[State, ...]
SequenceLike = Sequence[State]

MAX_EXACT_PERMUTATIONS = 50_000
MAX_FACTORIAL_INT_D = 20


@_dataclass(frozen=True)
class IBlockDiagnostic:
    """One candidate-level original i-block conformal diagnostic."""

    candidate: Path
    anchor_state: State
    p_value: float
    n_permutable_blocks: int
    log_full_group_size: float
    full_group_size: int | None
    n_permutations_evaluated: int


@_dataclass(frozen=True)
class AuxiliaryDiagnostic:
    """One extended candidate row for the cardinality-weighted method."""

    original_candidate: Path
    auxiliary_state: State
    extended_candidate: Path
    anchor_state: State
    p_value: float
    n_permutable_blocks: int
    log_full_group_size: float
    full_group_size: int | None
    n_permutations_evaluated: int
    normalized_cardinality_weight: float


@_dataclass(frozen=True)
class AggregatedCandidateResult:
    """Aggregated q_tilde result for one original candidate."""

    original_candidate: Path
    q_tilde: float
    included: bool
    auxiliary_rows: tuple[AuxiliaryDiagnostic, ...]


def check_adjacency(adjacency: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Return a checked square 0/1 adjacency matrix."""
    adjacency_array = np.asarray(adjacency, dtype=int)

    if adjacency_array.ndim != 2:
        raise ValueError("adjacency must be a two-dimensional square matrix.")

    if adjacency_array.shape[0] != adjacency_array.shape[1]:
        raise ValueError("adjacency must be square.")

    if not np.all((adjacency_array == 0) | (adjacency_array == 1)):
        raise ValueError("adjacency must contain only 0/1 entries.")

    if np.any(adjacency_array.sum(axis=1) == 0):
        raise ValueError("each state must have at least one allowed next state.")

    return adjacency_array


def check_sequence(
    sequence: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> list[State]:
    """Validate state labels and allowed transitions, then return a list."""
    adjacency_array = check_adjacency(adjacency)
    num_states = adjacency_array.shape[0]
    sequence_list = [int(state) for state in sequence]

    if len(sequence_list) == 0:
        raise ValueError("sequence must be nonempty.")

    for state in sequence_list:
        if state < 1 or state > num_states:
            raise ValueError(f"state must be between 1 and {num_states}.")

    for t in range(len(sequence_list) - 1):
        if adjacency_array[sequence_list[t] - 1, sequence_list[t + 1] - 1] != 1:
            raise ValueError(
                f"forbidden transition: {sequence_list[t]} -> {sequence_list[t + 1]}"
            )

    return sequence_list


def allowed_next(
    state: State,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> list[State]:
    """Return all states that may follow state."""
    adjacency_array = check_adjacency(adjacency)
    num_states = adjacency_array.shape[0]

    if state < 1 or state > num_states:
        raise ValueError(f"state must be between 1 and {num_states}.")

    return [
        next_state + 1
        for next_state, allowed in enumerate(adjacency_array[state - 1])
        if allowed == 1
    ]


def enumerate_paths(
    start_state: State,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> list[Path]:
    """Enumerate all allowed future paths of length horizon."""
    adjacency_array = check_adjacency(adjacency)
    num_states = adjacency_array.shape[0]

    if horizon < 0:
        raise ValueError("horizon must be nonnegative.")

    if start_state < 1 or start_state > num_states:
        raise ValueError(f"start_state must be between 1 and {num_states}.")

    def extend(current_state: State, steps_left: int) -> list[Path]:
        if steps_left == 0:
            return [()]

        paths: list[Path] = []
        for next_state in allowed_next(current_state, adjacency_array):
            for suffix in extend(next_state, steps_left - 1):
                paths.append((next_state,) + suffix)

        return paths

    return extend(start_state, horizon)


def transition_counts(sequence: SequenceLike) -> dict[tuple[State, State], int]:
    """Count adjacent transitions in a state sequence."""
    counts: dict[tuple[State, State], int] = {}
    for first, second in zip(sequence[:-1], sequence[1:]):
        key = (int(first), int(second))
        counts[key] = counts.get(key, 0) + 1

    return counts


def estimate_transition_matrix(
    sequence: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Estimate the unsmoothed transition matrix from one augmented sequence."""
    adjacency_array = check_adjacency(adjacency)
    sequence_list = check_sequence(sequence, adjacency_array)
    num_states = adjacency_array.shape[0]

    transition_matrix = np.zeros((num_states, num_states), dtype=float)
    row_totals = np.zeros(num_states, dtype=float)
    counts = transition_counts(sequence_list)

    for (first, _second), count in counts.items():
        row_totals[first - 1] += count

    for (first, second), count in counts.items():
        if row_totals[first - 1] > 0:
            transition_matrix[first - 1, second - 1] = count / row_totals[first - 1]

    return transition_matrix


def split_i_blocks(sequence: SequenceLike, anchor_state: State) -> tuple[list[State] | None, list[list[State]], list[State]]:
    """Split sequence into fixed I0, middle i-blocks, and fixed final tail."""
    sequence_list = [int(state) for state in sequence]
    positions = [
        index
        for index, state in enumerate(sequence_list)
        if state == int(anchor_state)
    ]

    if len(positions) == 0:
        raise ValueError("anchor_state does not appear in the sequence.")

    if positions[0] == 0:
        initial_block = None
    else:
        initial_block = sequence_list[: positions[0]]

    all_i_blocks: list[list[State]] = []
    for position_index, start in enumerate(positions):
        if position_index < len(positions) - 1:
            end = positions[position_index + 1]
            block = sequence_list[start:end]
        else:
            block = sequence_list[start:]

        all_i_blocks.append(block)

    tail = all_i_blocks[-1]
    permutable_blocks = all_i_blocks[:-1]

    return initial_block, permutable_blocks, tail


def build_sequence_from_blocks(
    initial_block: SequenceLike | None,
    ordered_blocks: Sequence[SequenceLike],
    tail: SequenceLike,
) -> list[State]:
    """Reconstruct a full sequence from fixed and permuted i-block pieces."""
    sequence: list[State] = []

    if initial_block is not None:
        sequence.extend(int(state) for state in initial_block)

    for block in ordered_blocks:
        sequence.extend(int(state) for state in block)

    sequence.extend(int(state) for state in tail)
    return sequence


def permutation_group_summary(n_permutable_blocks: int) -> tuple[float, int | None]:
    """Return log(D!) and, when display is practical, D! as an integer.

    Here D is the number of permutable middle i-blocks. The full mathematical
    permutation-group cardinality is |Pi| = D!, independent of any sampling cap.
    """
    if n_permutable_blocks < 0:
        raise ValueError("n_permutable_blocks must be nonnegative.")

    log_full_group_size = math.lgamma(n_permutable_blocks + 1)

    if n_permutable_blocks <= MAX_FACTORIAL_INT_D:
        full_group_size = math.factorial(n_permutable_blocks)
    else:
        full_group_size = None

    return log_full_group_size, full_group_size


def block_index_orders(
    n_permutable_blocks: int,
    max_permutations: int | None = None,
) -> list[tuple[int, ...]]:
    """Enumerate or sample indexed block-order permutations.

    A permutation is an ordering of block indices, not a resulting state
    sequence. Identical block contents are still different indexed blocks.
    """
    if n_permutable_blocks < 0:
        raise ValueError("n_permutable_blocks must be nonnegative.")

    full_group_size = math.factorial(n_permutable_blocks)

    if max_permutations is None:
        if full_group_size > MAX_EXACT_PERMUTATIONS:
            raise ValueError("too many exact block permutations; set max_permutations.")

        return list(permutations(range(n_permutable_blocks)))

    if max_permutations < 1:
        raise ValueError("max_permutations must be positive or None.")

    if full_group_size <= max_permutations and full_group_size <= MAX_EXACT_PERMUTATIONS:
        return list(permutations(range(n_permutable_blocks)))

    identity_order = tuple(range(n_permutable_blocks))
    orders = [identity_order]
    seen_orders = {identity_order}
    target_count = min(max_permutations, full_group_size)
    attempts = 0
    max_attempts = max(100, target_count * 20)

    while len(orders) < target_count and attempts < max_attempts:
        order_list = list(range(n_permutable_blocks))
        random.shuffle(order_list)
        order = tuple(order_list)

        if order not in seen_orders:
            seen_orders.add(order)
            orders.append(order)

        attempts += 1

    return orders


def nonconformity_score(
    sequence: SequenceLike,
    training_length: int,
    horizon: int,
    transition_matrix: np.ndarray,
) -> float:
    """Compute the original average-transition-probability nonconformity score."""
    if horizon < 1:
        raise ValueError("horizon must be positive for an i-block score.")

    state_at_training_end = int(sequence[training_length - 1])
    probability_sum = 0.0

    for step in range(1, horizon + 1):
        future_state = int(sequence[training_length + step - 1])
        transition_power = np.linalg.matrix_power(transition_matrix, step)
        probability_sum += transition_power[state_at_training_end - 1, future_state - 1]

    return 1.0 - probability_sum / horizon


def iblock_candidate_diagnostic(
    history: SequenceLike,
    candidate: SequenceLike,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    max_permutations: int | None = None,
) -> IBlockDiagnostic:
    """Compute one original randomized i-block p-value with cardinality details."""
    adjacency_array = check_adjacency(adjacency)
    history_list = check_sequence(history, adjacency_array)
    candidate_path = tuple(int(state) for state in candidate)

    if len(candidate_path) == 0:
        raise ValueError("candidate must be nonempty.")

    check_sequence([history_list[-1]] + list(candidate_path), adjacency_array)

    augmented_sequence = history_list + list(candidate_path)
    transition_matrix = estimate_transition_matrix(augmented_sequence, adjacency_array)
    anchor_state = candidate_path[-1]
    initial_block, permutable_blocks, tail = split_i_blocks(
        augmented_sequence,
        anchor_state,
    )

    n_permutable_blocks = len(permutable_blocks)
    block_orders = block_index_orders(n_permutable_blocks, max_permutations)
    identity_score = nonconformity_score(
        augmented_sequence,
        training_length=len(history_list),
        horizon=len(candidate_path),
        transition_matrix=transition_matrix,
    )

    scores: list[float] = []
    for order in block_orders:
        ordered_blocks = [permutable_blocks[index] for index in order]
        permuted_sequence = build_sequence_from_blocks(initial_block, ordered_blocks, tail)
        scores.append(
            nonconformity_score(
                permuted_sequence,
                training_length=len(history_list),
                horizon=len(candidate_path),
                transition_matrix=transition_matrix,
            )
        )

    tolerance = 1e-12
    n_greater = sum(score > identity_score + tolerance for score in scores)
    n_equal = sum(abs(score - identity_score) <= tolerance for score in scores)
    # Conservative tie handling: tied scores count toward the p-value.
    p_value = (n_greater + n_equal) / len(scores)
    log_full_group_size, full_group_size = permutation_group_summary(
        n_permutable_blocks
    )

    return IBlockDiagnostic(
        candidate=candidate_path,
        anchor_state=anchor_state,
        p_value=p_value,
        n_permutable_blocks=n_permutable_blocks,
        log_full_group_size=log_full_group_size,
        full_group_size=full_group_size,
        n_permutations_evaluated=len(block_orders),
    )


def original_iblock_table(
    history: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    max_permutations: int | None = None,
) -> list[IBlockDiagnostic]:
    """Return one original i-block diagnostic row for every candidate path."""
    adjacency_array = check_adjacency(adjacency)
    history_list = check_sequence(history, adjacency_array)

    candidates = enumerate_paths(history_list[-1], horizon, adjacency_array)

    return [
        iblock_candidate_diagnostic(
            history_list,
            candidate,
            adjacency_array,
            max_permutations=max_permutations,
        )
        for candidate in candidates
    ]


def original_iblock_prediction_set(
    history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    max_permutations: int | None = None,
) -> list[Path]:
    """Return original i-block candidates whose p-value exceeds alpha."""
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1.")

    return [
        row.candidate
        for row in original_iblock_table(
            history,
            horizon,
            adjacency,
            max_permutations=max_permutations,
        )
        if row.p_value > alpha
    ]


def _stable_cardinality_weights(log_group_sizes: Sequence[float]) -> list[float]:
    """Normalize log cardinalities for one fixed original candidate."""
    if len(log_group_sizes) == 0:
        raise ValueError("at least one log group size is required.")

    if not all(math.isfinite(value) for value in log_group_sizes):
        raise ValueError("log group sizes must be finite.")

    max_log_size = max(log_group_sizes)
    raw_weights = [math.exp(value - max_log_size) for value in log_group_sizes]
    raw_total = sum(raw_weights)

    if raw_total <= 0 or not math.isfinite(raw_total):
        raise ValueError("could not normalize cardinality weights.")

    return [weight / raw_total for weight in raw_weights]


def auxiliary_candidate_table(
    history: SequenceLike,
    horizon: int,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    max_permutations: int | None = None,
) -> list[AuxiliaryDiagnostic]:
    """Return all one-auxiliary-state diagnostic rows with cardinality weights."""
    if horizon < 1:
        raise ValueError("horizon must be positive for auxiliary candidates.")

    adjacency_array = check_adjacency(adjacency)
    history_list = check_sequence(history, adjacency_array)
    original_candidates = enumerate_paths(history_list[-1], horizon, adjacency_array)
    rows: list[AuxiliaryDiagnostic] = []

    for original_candidate in original_candidates:
        base_rows: list[IBlockDiagnostic] = []
        auxiliary_states = allowed_next(original_candidate[-1], adjacency_array)

        for auxiliary_state in auxiliary_states:
            extended_candidate = original_candidate + (auxiliary_state,)
            base_rows.append(
                iblock_candidate_diagnostic(
                    history_list,
                    extended_candidate,
                    adjacency_array,
                    max_permutations=max_permutations,
                )
            )

        weights = _stable_cardinality_weights(
            [row.log_full_group_size for row in base_rows]
        )

        for diagnostic, weight in zip(base_rows, weights):
            rows.append(
                AuxiliaryDiagnostic(
                    original_candidate=original_candidate,
                    auxiliary_state=diagnostic.candidate[-1],
                    extended_candidate=diagnostic.candidate,
                    anchor_state=diagnostic.anchor_state,
                    p_value=diagnostic.p_value,
                    n_permutable_blocks=diagnostic.n_permutable_blocks,
                    log_full_group_size=diagnostic.log_full_group_size,
                    full_group_size=diagnostic.full_group_size,
                    n_permutations_evaluated=diagnostic.n_permutations_evaluated,
                    normalized_cardinality_weight=weight,
                )
            )

    return rows


def cardinality_weighted_auxiliary_cp(
    history: SequenceLike,
    horizon: int,
    alpha: float,
    adjacency: Sequence[Sequence[int]] | np.ndarray,
    max_permutations: int | None = None,
) -> list[AggregatedCandidateResult]:
    """Aggregate extended p-values using normalized full-cardinality weights.

    For each original candidate y,
        q_tilde(y) = sum_u weight(y, u) * p_block(y, u),
    where weight(y, u) is based on the full group size |Pi(y, u)| = D(y, u)!.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1.")

    auxiliary_rows = auxiliary_candidate_table(
        history,
        horizon,
        adjacency,
        max_permutations=max_permutations,
    )

    grouped_rows: dict[Path, list[AuxiliaryDiagnostic]] = {}
    for row in auxiliary_rows:
        grouped_rows.setdefault(row.original_candidate, []).append(row)

    aggregated_results: list[AggregatedCandidateResult] = []
    for original_candidate, rows in grouped_rows.items():
        q_tilde = sum(
            row.normalized_cardinality_weight * row.p_value
            for row in rows
        )

        aggregated_results.append(
            AggregatedCandidateResult(
                original_candidate=original_candidate,
                q_tilde=q_tilde,
                included=q_tilde > alpha,
                auxiliary_rows=tuple(rows),
            )
        )

    return aggregated_results
