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
├── simulation_study.py
├── plot_simulation_results.py
├── emmett_tests.py
├── requirements.txt
├── simulation_results/
│   └── .gitkeep
└── tests/
    ├── test_markov_cp.py
    └── test_simulation_study.py
```

`markov_cp_routines.py` contains reusable mathematical routines only. It has no
plotting code, no experiment-specific global setup, and no giant all-method
wrapper.

`quick_meeting_demo.py` is the fastest live-meeting script. Usually edit only
`HISTORY`, `SEEDS`, `ALPHA`, `MAX_PERMUTATIONS`, and `RANDOMIZED_TIES`.

`run_demo.py` is the more detailed diagnostic script. It runs the four fixed
history cases and both horizons requested for comparing the methods.

`simulation_study.py` runs empirical coverage and prediction-set-size
simulations for dense three-state Markov chains.

`plot_simulation_results.py` reads simulation summary CSV files and creates
reliability curves and scaled set-size plots.

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

```math
z = (y, u)
```

The current implementation passes the full extended candidate `(y,u)` into the
original i-block diagnostic. Therefore, the current nonconformity score
evaluates the full extended path `(y,u)`. Do not change this unless the advisor
confirms that `u` should only be used as an anchor while the score evaluates
only `y`.

For each extended candidate, the code computes:

```math
p_{\mathrm{block}}(y,u)
```

The same auxiliary table contains both weights below, so the two weighted
methods can be compared using exactly the same computed p-values.

## Method 1: Permutation-count (D!) Weighted

This is the old cardinality-weighted method, retained as a comparison method.
For fixed `y`, its normalized weight is:

```math
\omega^{\mathrm{perm}}_{y,u}
=
\frac{D_{y,u}!}{\sum_{v} D_{y,v}!}
```

The aggregate is:

```math
\tilde{q}_{\mathrm{perm}}(y)
=
\sum_{u} \omega^{\mathrm{perm}}_{y,u} p_{\mathrm{block}}(y,u)
```

The corresponding exploratory set is:

```math
C_{\mathrm{perm}}
=
\{ y : \tilde{q}_{\mathrm{perm}}(y) > \alpha \}
```

The old public wrapper `cardinality_weighted_auxiliary_cp(...)` remains
available for backward compatibility and uses this D!-weighted rule.

## Method 2: I-block-count (D) Weighted

The new comparison method weights by the number of middle permutable i-blocks:

```math
\omega^{\mathrm{block}}_{y,u}
=
\frac{D_{y,u}}{\sum_{v} D_{y,v}}
```

The aggregate is:

```math
\tilde{q}_{\mathrm{block}}(y)
=
\sum_{u} \omega^{\mathrm{block}}_{y,u} p_{\mathrm{block}}(y,u)
```

The corresponding exploratory set is:

```math
C_{\mathrm{block}}
=
\{ y : \tilde{q}_{\mathrm{block}}(y) > \alpha \}
```

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

```math
p = \frac{n_{>} + n_{=}}{n_{\mathrm{eval}}}
```

When `randomized_ties=True`, the code samples `xi` from `Uniform(0,1)` and uses:

```math
p = \frac{n_{>} + \xi n_{=}}{n_{\mathrm{eval}}}
```

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

## Simulation Study

The simulation study estimates empirical coverage by generating a Markov chain
of length `T + h`, using the first `T` states as the training history, and then
checking whether the true future path `X_{T+1:T+h}` is included in the final CP
set.

The main default case is the dense three-state chain:

```python
P_SIM1 = [
    [0.7, 0.15, 0.15],
    [0.3, 0.6, 0.1],
    [0.2, 0.2, 0.6],
]
PI_SIM1 = [0.4, 0.3, 0.3]
```

The CP code still uses dense adjacency, not the transition probability matrix:

```python
ADJACENCY = np.ones((3, 3), dtype=int)
```

The simulation compares:

1. `original`: original i-block CP.
2. `permutation_count`: the old D!-weighted auxiliary method.
3. `iblock_count`: the new D-weighted auxiliary method.

For each method, horizon, and alpha, the summary reports:

- empirical coverage
- Monte Carlo standard error for coverage
- mean prediction-set size
- mean scaled set size, where the set size is divided by `3**h`

Run a fast smoke test first:

```bash
python simulation_study.py --quick
```

Run the main Sim 1 study:

```bash
python simulation_study.py --sim sim1 --n-sim 500 --T 500
```

Optional stress-test cases are available:

```bash
python simulation_study.py --sim sim2 --n-sim 500 --T 500
python simulation_study.py --sim sim3 --n-sim 500 --T 500
python simulation_study.py --sim all --n-sim 500 --T 500
```

Raw replicate-level output is saved as:

```text
simulation_results/{simulation_name}_raw_results.csv
```

Summary output is saved as:

```text
simulation_results/{simulation_name}_summary.csv
simulation_results/all_summary.csv
```

Create reliability and set-size plots with:

```bash
python plot_simulation_results.py --summary simulation_results/sim1_summary.csv
```

The reliability plots use target coverage `1 - alpha` on the x-axis and
empirical coverage on the y-axis, with a diagonal reference line. The set-size
plots use target coverage on the x-axis and mean scaled set size on the y-axis.

The plots are saved in `simulation_results/`, for example:

```text
simulation_results/sim1_reliability_original.png
simulation_results/sim1_reliability_permutation_count.png
simulation_results/sim1_reliability_iblock_count.png
simulation_results/sim1_scaled_set_size_original.png
simulation_results/sim1_scaled_set_size_permutation_count.png
simulation_results/sim1_scaled_set_size_iblock_count.png
```

For a fair comparison, the simulation computes the auxiliary p-values once per
history and horizon. The D!-weighted and D-weighted methods then aggregate the
same auxiliary rows, so they use the same values of `p_block(y,u)`.

This simulation study is empirical. It does not prove finite-sample validity
for either weighted auxiliary aggregate.

## Tests

Run:

```bash
python -m py_compile \
    markov_cp_routines.py \
    run_demo.py \
    quick_meeting_demo.py \
    simulation_study.py \
    plot_simulation_results.py \
    emmett_tests.py
python -m unittest discover -s tests
python simulation_study.py --quick
python plot_simulation_results.py --summary simulation_results/sim1_summary.csv
python quick_meeting_demo.py
python run_demo.py
git diff --check
```

The tests cover adjacency validation, candidate enumeration, i-block
decomposition, indexed block-order permutations, factorial cardinalities,
tie-handling modes, D! weights, D weights, shared-row aggregation, p-value
ranges, `q_tilde` ranges, seed reproducibility, public signatures without
`random_seed`, final prediction-set path lengths, successful demo execution,
Markov simulation helpers, quick simulation output, summary statistics, and
plot creation.
