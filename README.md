# Markov CP i-block

Preliminary research code for conformal prediction with finite-state Markov
chains. The active workflow is intentionally small so dense three-state
experiments can be edited and rerun during meetings.

## Active File Structure

```text
markov-cp-iblock/
├── README.md
├── markov_cp_routines.py
├── quick_meeting_demo.py
├── run_demo.py
├── emmett_tests.py
├── requirements.txt
└── tests/
    └── test_markov_cp.py
```

`markov_cp_routines.py` contains reusable mathematical routines only. It has no
plotting code, no experiment-specific global setup, and no giant all-method
wrapper.

`quick_meeting_demo.py` is the fastest live-meeting script. Usually edit only
`HISTORY`, `SEEDS`, `ALPHA`, `MAX_PERMUTATIONS`, and `RANDOMIZED_TIES`.

`run_demo.py` is the more detailed diagnostic script. It runs the four fixed
history cases and both horizons requested for comparing the methods.

`emmett_tests.py` is an advisor-added scratch/testing script and is preserved.

## Original i-block Diagnostic

For a candidate future path `y`, the code forms the augmented sequence
`history + y`, estimates the transition matrix from that candidate-specific
augmented sequence, decomposes the sequence into i-blocks using
`anchor_state = y[-1]`, and computes the original i-block p-value.

The candidate-level diagnostic reports:

- `p_value`
- `n_permutable_blocks`, written mathematically as `D`
- `log_full_group_size = log(D!)`
- `full_group_size = D!` when the integer is small enough to print
- `n_permutations_evaluated`, which may be capped by `max_permutations`

Here `D` counts only the middle permutable i-blocks. It excludes the fixed
initial piece `I0` and the final fixed tail/anchoring block.

Block permutations are indexed block-order permutations. If two blocks have the
same state contents, swapping their labels still counts as a different
permutation-group element.

## Auxiliary Extensions

For each original candidate `y` and one auxiliary state `u`, define the full
extended candidate:

$$
z = (y, u).
$$

The current implementation passes the full extended candidate `(y,u)` into the
original i-block diagnostic. Therefore, the current nonconformity score
evaluates the full extended path `(y,u)`. Do not change this unless the advisor
confirms that `u` should only be used as an anchor while the score evaluates
only `y`.

For each extended candidate, the code computes:

$$
p_{\mathrm{block}}(y,u).
$$

The same auxiliary table contains both weights below, so the two weighted
methods can be compared using exactly the same computed p-values.

## Method 1: Permutation-count (D!) Weighted

This is the old cardinality-weighted method, retained as a comparison method.
For fixed `y`, its normalized weight is:

$$
\omega^{\mathrm{perm}}_{y,u}
=
\frac{D_{y,u}!}{\sum_v D_{y,v}!}.
$$

The aggregate is:

$$
\tilde q_{\mathrm{perm}}(y)
=
\sum_u \omega^{\mathrm{perm}}_{y,u} p_{\mathrm{block}}(y,u).
$$

The corresponding exploratory set is:

$$
C_{\mathrm{perm}}
=
\{y : \tilde q_{\mathrm{perm}}(y) > \alpha\}.
$$

The old public wrapper `cardinality_weighted_auxiliary_cp(...)` remains
available for backward compatibility and uses this D!-weighted rule.

## Method 2: I-block-count (D) Weighted

The new comparison method weights by the number of middle permutable i-blocks:

$$
\omega^{\mathrm{block}}_{y,u}
=
\frac{D_{y,u}}{\sum_v D_{y,v}}.
$$

The aggregate is:

$$
\tilde q_{\mathrm{block}}(y)
=
\sum_u \omega^{\mathrm{block}}_{y,u} p_{\mathrm{block}}(y,u).
$$

The corresponding exploratory set is:

$$
C_{\mathrm{block}}
=
\{y : \tilde q_{\mathrm{block}}(y) > \alpha\}.
$$

No pseudocount is used. If one row has `D = 0`, its D-weight is exactly zero.
If all auxiliary continuations for a fixed `y` have `D = 0`, the D-weighted rule
is currently undefined and the code raises a `ValueError`.

Neither weighted aggregate is currently claimed to have exact finite-sample
conformal validity. These are computational comparison methods for ongoing
research discussion.

## Cardinality vs Evaluated Permutations

The implementation deliberately separates three quantities:

- `n_permutable_blocks`: `D(y,u)`
- `full_group_size`: the theoretical `D(y,u)!`
- `n_permutations_evaluated`: the number of sampled or enumerated permutations
  actually used in the p-value calculation

The permutation-count method uses `full_group_size` through
`log_full_group_size = lgamma(D + 1)`. It does not use
`n_permutations_evaluated`.

## Tie Handling

Core routines accept:

```python
randomized_ties=False
```

When `randomized_ties=False`, tied scores are handled conservatively by fixing
the tie randomizer to `xi = 1`, so:

$$
p = \frac{n_> + n_=}{n_{\mathrm{eval}}}.
$$

When `randomized_ties=True`, the code samples `xi` from `Uniform(0,1)` and uses:

$$
p = \frac{n_> + \xi n_=}{n_{\mathrm{eval}}}.
$$

No core mathematical function takes a `random_seed` argument. The demo scripts
set seeds externally. Seeds can still affect sampled block permutations even
when ties are nonrandomized.

## Quick Meeting Demo

For the fastest live-meeting check, run:

```bash
python quick_meeting_demo.py
```

Usually edit only this block near the top of `quick_meeting_demo.py`:

```python
HISTORY = [1, 2] * 100
HORIZON = 1
SEEDS = [1]
ALPHA = 0.20
MAX_PERMUTATIONS = 500
RANDOMIZED_TIES = False
```

The script prints all auxiliary rows and then explicitly prints:

```text
FINAL CP SETS

Original i-block CP set:
Permutation-count (D!) weighted CP set:
I-block-count (D) weighted CP set:
```

## Detailed Demo

Run:

```bash
python run_demo.py
```

The detailed demo runs:

1. same-state history: `[1] * 100`
2. one fixed random history of length 100 generated with `HISTORY_SEED`
3. two-cycle history: `[1, 2] * 100`
4. three-cycle history: `[1, 2, 3] * 100`

For every case, it runs:

```python
HORIZONS = (1, 2)
```

For `horizon=1`, there are 3 original candidate paths and 9 extended auxiliary
rows. For `horizon=2`, there are 9 original candidate paths and 27 extended
auxiliary rows. Final prediction sets contain only the original candidate path
`y`; the auxiliary state is never returned as part of a final set.

Repeated-seed summaries are optional:

```python
RUN_REPEATED_SEEDS = False
RANDOM_SEEDS = range(1, 4)
```

Set `RUN_REPEATED_SEEDS = True` when you want the repeated-run summaries.

## Tests

Run:

```bash
python -m py_compile markov_cp_routines.py run_demo.py quick_meeting_demo.py emmett_tests.py
python -m unittest discover -s tests
python quick_meeting_demo.py
python run_demo.py
git diff --check
```

The tests cover adjacency validation, candidate enumeration, i-block
decomposition, indexed block-order permutations, factorial cardinalities,
tie-handling modes, D! weights, D weights, shared-row aggregation, p-value
ranges, `q_tilde` ranges, seed reproducibility, public signatures without
`random_seed`, final prediction-set path lengths, and successful demo execution.
