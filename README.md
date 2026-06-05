# Markov CP i-block

Preliminary research code for Markov conformal prediction experiments. The repository implements small, readable routines for:

- a likelihood-based prediction baseline;
- the original i-block conformal prediction method;
- a graph-constrained CP+1 variant;
- simple demo examples for testing and comparison.

The core routines do not hard-code a specific graph. They take an adjacency matrix as input and infer the number of states with `adjacency.shape[0]`. State labels are assumed to be `1, 2, ..., S`.

## Repository Structure

```text
markov-cp-iblock/
├── README.md
├── requirements.txt
├── .gitignore
├── markov_cp_routines.py
├── run_demo.py
└── demo_outputs/
```

## Files

- `markov_cp_routines.py`: reusable validation helpers, path enumeration, likelihood baseline, original i-block CP, CP+1, summaries, and plotting helpers.
- `run_demo.py`: user-facing demo script where you set the adjacency matrix, training history, forecast horizon, alpha level, permutation cap, and random seed.
- `demo_outputs/`: generated demo plots. The folder is kept in Git with `.gitkeep`, but generated files inside it are ignored.

## Install

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python run_demo.py
```

The demo prints prediction-set summaries and saves state-composition plots for the longer Sweden-like example to `demo_outputs/`.

## Included Demos

- Demo 1: a three-state fully connected Markov chain with a short mixed history.
- Demo 2: a short four-state conflict-graph history with repeated state 1 observations.
- Demo 3: a longer four-state conflict-graph history with repeated state 1 observations and saved plots.
- Diagnostic example: computes one candidate i-block p-value and prints the block decomposition details.

## Modify An Experiment

Edit `run_demo.py` to change:

- `adjacency`: an `S x S` NumPy array where `adjacency[a-1, b-1] = 1` means transition `a -> b` is allowed.
- `history`: observed state sequence using labels `1, ..., S`.
- `horizon`: number of future states to forecast.
- `alpha`: miscoverage level.
- `max_permutations`: maximum number of i-block permutations to enumerate or sample.
- `random_seed`: seed for reproducible sampled permutations and tie-breaking.

For example:

```python
import numpy as np
from markov_cp_routines import run_all_methods

adjacency = np.ones((3, 3), dtype=int)
history = [1, 1, 2, 3, 1, 2]
results = run_all_methods(
    history=history,
    horizon=2,
    alpha=0.2,
    adjacency=adjacency,
    max_permutations=500,
    random_seed=123,
)
```

This code is intended as preliminary research code for Markov conformal prediction, with clarity and easy modification prioritized over packaging complexity.
