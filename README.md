# Markov CP i-block

Preliminary research code for conformal prediction with finite-state Markov
chains. The active workflow is intentionally small so that dense three-state
experiments can be edited and rerun quickly.

## Active File Structure

```text
markov-cp-iblock/
├── README.md
├── markov_cp_routines.py
├── quick_meeting_demo.py
├── run_demo.py
├── requirements.txt
└── tests/
    └── test_markov_cp.py
```

`markov_cp_routines.py` contains reusable mathematical routines only. It has no
experiment-specific histories, plotting code, or giant method wrapper.

`run_demo.py` is the more detailed diagnostic demo. It prints the original
i-block table, all auxiliary rows, the comparison table, and repeated-seed
summaries.

`quick_meeting_demo.py` is the fastest live-meeting demo. It is meant for
quickly editing one fixed history and a short list of random seeds.

## Current Research Setting

This branch focuses on the simplified dense-adjacency setting:

```python
X = {1, 2, 3}
A = np.ones((3, 3), dtype=int)
```

The default demo uses:

- training length `T = 100`
- forecast horizon `L = 1`
- one auxiliary state `K = 1`
- `alpha = 0.2`

The sparse four-state conflict graph and the two-auxiliary-state extension are
not part of this refactor.

## Original i-block Diagnostic

For a candidate future path `y`, the code forms the augmented sequence
`history + y`, estimates the transition matrix from that candidate-specific
augmented sequence, decomposes the sequence into i-blocks using
`anchor_state = y[-1]`, and computes the randomized original i-block p-value.

The candidate-level diagnostic reports:

- `p_value`
- `n_permutable_blocks`, written mathematically as `D`
- `log_full_group_size = log(D!)`
- `full_group_size = D!` when the integer is small enough to print usefully
- `n_permutations_evaluated`, which may be capped by `max_permutations`

Block permutations are indexed block-order permutations. If two blocks have the
same state contents, swapping their labels still counts as a different
permutation-group element.

## Cardinality-weighted Auxiliary Method

For each original candidate $y$ and one auxiliary state $u$, define the
extended candidate:

$$ z = (y, u). $$

The code computes the original i-block p-value on each extended candidate:

$$ p_{\mathrm{block}}(y, u). $$

For each extended candidate, let $D(y, u)$ be the number of middle permutable
i-blocks. This excludes both the fixed prefix $I_0$ and the final fixed tail
block. The full mathematical permutation-group size is:

$$ |\Pi(y, u)| = D(y, u)!. $$

The cardinality-weighted aggregated value is:

$$ \tilde q(y) = \sum_{u} \frac{|\Pi(y, u)|}{\sum_{v} |\Pi(y, v)|} p_{\mathrm{block}}(y, u). $$

The new exploratory prediction set is:

$$ C_{\mathrm{new}} = \{y : \tilde q(y) > \alpha\}. $$

This is an exploratory computational method. A formal validity proof is not
included in this branch.

## Cardinality vs Evaluated Permutations

The implementation deliberately separates three quantities:

- `n_permutable_blocks`: `D(y, u)`
- `full_group_size`: the theoretical `D(y, u)!`
- `n_permutations_evaluated`: the number of sampled or enumerated permutations
  actually used in the p-value calculation

The normalized cardinality weights use `full_group_size`, via the stable
quantity `log_full_group_size = lgamma(D + 1)`. They do not use
`n_permutations_evaluated`.

## Quick Experiment Setup

For the fastest live-meeting check, run:

```bash
python quick_meeting_demo.py
```

Usually edit only this small control block near the top of
`quick_meeting_demo.py`:

```python
HISTORY = [1] * 100
SEEDS = [1, 2, 3]
ALPHA = 0.20
MAX_PERMUTATIONS = 500
```

For a more detailed diagnostic run, use:

```bash
python run_demo.py
```

For a new dense three-state experiment, edit the small control block near the
top of `run_demo.py`:

```python
HISTORY = [1] * 100
DETAIL_RANDOM_SEED = 1
RANDOM_SEEDS = range(1, 11)
MAX_PERMUTATIONS = 500
ALPHA = 0.2
```

For example, to test a different observed training sequence:

```python
history = [1, 2, 3, 1, 2, 3, 1, 1, 2]
```

change only:

```python
HISTORY = [1, 2, 3, 1, 2, 3, 1, 1, 2]
```

Then run:

```bash
python run_demo.py
```

The output prints the three original candidates, all nine extended candidates,
the cardinality weights, `q_tilde`, and the repeated-seed results.

The repeated randomized p-values are controlled by:

```python
RANDOM_SEEDS = range(1, 11)
```

`RANDOM_SEEDS` in `run_demo.py` and `SEEDS` in `quick_meeting_demo.py` control
both randomized tie-breaking and sampled block permutations. They should not be
read as only tie-breaking seeds.

This prints a table named `p-values from each random seed`, with one row for
each seed and candidate path. For the default `range(1, 11)`, this gives ten
runs for each candidate path. To run only three repeats while checking code
quickly, use:

```python
RANDOM_SEEDS = range(1, 4)
```

The reusable mathematical code in `markov_cp_routines.py` usually does not need
to be edited for these quick checks.

Current implementation note: `p_block(y, u)` passes the full extended candidate
`(y, u)` into the original i-block diagnostic. Therefore, the current
nonconformity score evaluates the full extended path `(y, u)`. Do not change
this unless the advisor confirms that `u` should only be used as an anchor while
the score evaluates only `y`.

## Built-in Demo Cases

If you want the two built-in examples, set:

```python
RUN_BUILT_IN_CASES = True
```

Those built-in cases are:

1. A fixed random/mixed training sequence of length 100 from states `{1,2,3}`,
   generated once using `HISTORY_SEED`.
2. A dominant-state training sequence:

   ```python
   history = [1] * 100
   ```

For each case, the script prints:

- the original i-block candidate table;
- all nine extended candidate rows `(y, u)`;
- each row's p-value, `D`, `log(D!)`, cardinality weight, and evaluated
  permutation count;
- the comparison between original p-values and `q_tilde(y)`;
- a ten-seed summary of mean original p-values, original inclusion frequency,
  mean `q_tilde`, and new inclusion frequency.

## Tests

Run:

```bash
python -m py_compile markov_cp_routines.py run_demo.py quick_meeting_demo.py
python -m unittest discover -s tests
```

The tests cover adjacency validation, candidate enumeration, i-block
decomposition, indexed block-order permutations, factorial cardinalities,
cardinality weights, p-value ranges, `q_tilde` ranges, seed reproducibility,
core signatures without `random_seed`, and successful demo execution.
