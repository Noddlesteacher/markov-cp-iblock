"""Generate saved Markov-chain inputs for the simulation workflow.

This script does not run conformal prediction and does not create plots. It
only simulates state sequences and saves them for run_simulation_analysis.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_N_SIM = 500
DEFAULT_T = 500
DEFAULT_HORIZONS = (1, 2, 3)
DEFAULT_SEED = 20260701
DEFAULT_TARGET_COVERAGES = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
)

P_SIM1 = np.array(
    [
        [0.70, 0.15, 0.15],
        [0.30, 0.60, 0.10],
        [0.20, 0.20, 0.60],
    ],
    dtype=float,
)
PI_SIM1 = np.array([0.40, 0.30, 0.30], dtype=float)

SIMULATION_CASES = {
    "sim1": (P_SIM1, PI_SIM1),
}


def validate_transition_matrix(P: np.ndarray, pi: np.ndarray) -> None:
    """Validate a finite-state Markov transition matrix and initial law."""
    P_array = np.asarray(P, dtype=float)
    pi_array = np.asarray(pi, dtype=float)

    if P_array.ndim != 2:
        raise ValueError("P must be a two-dimensional square matrix.")
    if P_array.shape[0] != P_array.shape[1]:
        raise ValueError("P must be square.")
    if np.any(P_array < -1e-12):
        raise ValueError("P entries must be nonnegative.")
    if not np.allclose(P_array.sum(axis=1), 1.0):
        raise ValueError("each row of P must sum to 1.")

    if pi_array.ndim != 1:
        raise ValueError("pi must be a one-dimensional vector.")
    if len(pi_array) != P_array.shape[0]:
        raise ValueError("pi length must match the number of states in P.")
    if np.any(pi_array < -1e-12):
        raise ValueError("pi entries must be nonnegative.")
    if not np.isclose(pi_array.sum(), 1.0):
        raise ValueError("pi must sum to 1.")


def simulate_markov_chain(
    P: np.ndarray,
    pi: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> list[int]:
    """Simulate a Markov chain using 1-based state labels."""
    validate_transition_matrix(P, pi)

    if length < 1:
        raise ValueError("length must be positive.")

    num_states = int(np.asarray(P).shape[0])
    zero_based_states = np.arange(num_states)
    current = int(rng.choice(zero_based_states, p=pi))
    sequence = [current + 1]

    for _ in range(length - 1):
        current = int(rng.choice(zero_based_states, p=P[current]))
        sequence.append(current + 1)

    if len(sequence) != length:
        raise RuntimeError("simulated sequence has the wrong length.")
    return sequence


def generate_sequences(
    P: np.ndarray,
    pi: np.ndarray,
    n_sim: int,
    training_length: int,
    horizons: Sequence[int],
    seed: int,
) -> np.ndarray:
    """Generate one length T + max(horizons) sequence per replicate."""
    if n_sim < 1:
        raise ValueError("n_sim must be positive.")
    if training_length < 1:
        raise ValueError("training_length must be positive.")
    if len(horizons) == 0 or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive integers.")

    max_horizon = max(int(horizon) for horizon in horizons)
    sequence_length = training_length + max_horizon
    rng = np.random.default_rng(seed)
    sequences = [
        simulate_markov_chain(P, pi, length=sequence_length, rng=rng)
        for _ in range(n_sim)
    ]
    return np.asarray(sequences, dtype=int)


def save_simulation_inputs(
    simulation_name: str,
    sequences: np.ndarray,
    P: np.ndarray,
    pi: np.ndarray,
    training_length: int,
    horizons: Sequence[int],
    target_coverages: Sequence[float],
    seed: int,
    output_dir: Path,
    quick: bool = False,
) -> tuple[Path, Path]:
    """Save simulated sequences plus a small metadata JSON file."""
    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_quick" if quick else ""
    npz_path = input_dir / f"{simulation_name}_sequences{suffix}.npz"
    metadata_path = input_dir / f"{simulation_name}_metadata{suffix}.json"
    state_labels = list(range(1, int(np.asarray(P).shape[0]) + 1))

    np.savez(
        npz_path,
        sequences=sequences,
        P=np.asarray(P, dtype=float),
        pi=np.asarray(pi, dtype=float),
        T=np.asarray(training_length, dtype=int),
        horizons=np.asarray(tuple(horizons), dtype=int),
        target_coverages=np.asarray(tuple(target_coverages), dtype=float),
        seed=np.asarray(seed, dtype=int),
        state_labels=np.asarray(state_labels, dtype=int),
        simulation=np.asarray(simulation_name),
    )

    metadata = {
        "simulation": simulation_name,
        "n_sim": int(sequences.shape[0]),
        "sequence_length": int(sequences.shape[1]),
        "T": int(training_length),
        "horizons": [int(horizon) for horizon in horizons],
        "target_coverages": [float(value) for value in target_coverages],
        "seed": int(seed),
        "state_labels": state_labels,
        "P": np.asarray(P, dtype=float).tolist(),
        "pi": np.asarray(pi, dtype=float).tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return npz_path, metadata_path


def parse_float_tuple(value: str) -> tuple[float, ...]:
    """Parse comma-separated floats from the command line."""
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers from the command line."""
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="generate tiny inputs")
    parser.add_argument(
        "--simulation",
        choices=sorted(SIMULATION_CASES),
        default="sim1",
        help="simulation input design to generate",
    )
    parser.add_argument("--n-sim", type=int, default=DEFAULT_N_SIM)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--horizons",
        type=parse_int_tuple,
        default=DEFAULT_HORIZONS,
        help="comma-separated horizons, for example 1,2,3",
    )
    parser.add_argument(
        "--target-coverages",
        type=parse_float_tuple,
        default=DEFAULT_TARGET_COVERAGES,
        help="comma-separated target coverages, for example 0.5,0.8,1.0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_results"),
    )
    return parser


def main() -> None:
    """Generate and save simulation inputs."""
    args = build_parser().parse_args()

    if args.quick:
        n_sim = 3
        training_length = 30
        horizons = (1,)
        target_coverages = (0.80,)
    else:
        n_sim = args.n_sim
        training_length = args.T
        horizons = tuple(args.horizons)
        target_coverages = tuple(args.target_coverages)

    P, pi = SIMULATION_CASES[args.simulation]
    sequence_length = training_length + max(horizons)

    print(f"Simulation: {args.simulation}")
    print(f"Number of replicates: {n_sim}")
    print(f"Training length: {training_length}")
    print(f"Horizons: {', '.join(str(horizon) for horizon in horizons)}")
    print(
        "Target coverages: "
        + ", ".join(f"{coverage:.2f}" for coverage in target_coverages)
    )
    print(f"Sequence length: {sequence_length}")
    print(f"Seed: {args.seed}")

    sequences = generate_sequences(
        P,
        pi,
        n_sim=n_sim,
        training_length=training_length,
        horizons=horizons,
        seed=args.seed,
    )
    npz_path, metadata_path = save_simulation_inputs(
        args.simulation,
        sequences,
        P,
        pi,
        training_length,
        horizons,
        target_coverages,
        args.seed,
        args.output_dir,
        quick=args.quick,
    )
    print(f"Saved sequences to {npz_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
