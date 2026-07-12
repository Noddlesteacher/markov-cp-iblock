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
├── run_simulation_analysis.py
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

`simulation_study.py` generates and saves Markov-chain input sequences. It does
not run CP and does not create plots.

`run_simulation_analysis.py` loads saved sequences, runs the CP methods, and
saves raw and summarized output.

`plot_simulation_results.py` reads saved summary CSV files and creates
reliability figures plus raw prediction-set-size tables. It does not rerun CP.

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
2. one fixed Markov-chain sequence generated from `P_SIM1` and `PI_SIM1`
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

## Canonical Simulation Workflow

The simulation workflow is split into three steps so that changing a plot does
not require rerunning the expensive CP calculations.

1. `simulation_study.py` generates input sequences and saves them.
2. `run_simulation_analysis.py` loads those sequences, runs CP, and saves CSV
   output.
3. `plot_simulation_results.py` loads the saved CSV output and makes figures
   and tables.

For each stochastic replicate, the code generates one Markov chain of length
`T + max(horizons)`. The first `T` states are the common training history for
all horizons. The true futures are nested prefixes of the same realized future:
`h=1` uses `X_{T+1}`, `h=2` uses `(X_{T+1}, X_{T+2})`, and `h=3` uses
`(X_{T+1}, X_{T+2}, X_{T+3})`.

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

The default simulation analysis compares:

1. `original`: original i-block CP.
2. `iblock_count`: the D-weighted auxiliary method.

The old `permutation_count` D!-weighted method remains available in the core
code and can be included as a legacy comparison, but it is not part of the
default figures and tables.

For each method, horizon, and alpha, the summary reports:

- empirical coverage
- Monte Carlo standard error for coverage
- mean raw prediction-set size
- standard deviation of raw prediction-set size

The target-coverage grid includes `1.00`, corresponding to `alpha = 0`.

Generate quick input data:

```bash
python simulation_study.py --quick
```

Run quick CP analysis on those saved inputs:

```bash
python run_simulation_analysis.py \
    --input simulation_results/input/sim1_sequences_quick.npz \
    --quick
```

Create quick figures and tables from the saved summary:

```bash
python plot_simulation_results.py \
    --summary simulation_results/output/sim1_summary_quick.csv
```

For the full Sim 1 input generation:

```bash
python simulation_study.py \
    --simulation sim1 \
    --n-sim 500 \
    --T 500 \
    --output-dir simulation_results
```

Then run the analysis:

```bash
python run_simulation_analysis.py \
    --input simulation_results/input/sim1_sequences.npz \
    --max-permutations 500
```

Raw replicate-level output is saved as:

```text
simulation_results/output/sim1_raw_results.csv
```

Summary output is saved as:

```text
simulation_results/output/sim1_summary.csv
```

Publication-style reliability figures and raw set-size tables are saved as:

```text
simulation_results/figures/sim1_reliability.png
simulation_results/figures/sim1_reliability.pdf
simulation_results/tables/sim1_raw_set_size_summary.csv
simulation_results/tables/sim1_raw_set_size_target_080.md
simulation_results/tables/sim1_raw_set_size_target_080.tex
```

The reliability plots use target coverage `1 - alpha` on the x-axis and
empirical coverage on the y-axis, with a diagonal reference line. The axis
limits are fixed at 0.50 to 1.00. The set-size tables report raw prediction-set
cardinality, not normalized or scaled size.

Case 2 in the diagnostic workflow is also generated from the same dense Markov
transition matrix and initial distribution. It uses one sequence of length
`T + max(horizons)`, one common training history, and nested true futures:

```bash
python run_simulation_analysis.py --case case2
```

A Sweden-style all-state-1 diagnostic is available as:

```bash
python run_simulation_analysis.py --case sweden
```

This deterministic diagnostic uses `history = [1] * T` and true future
`(1,) * h`. It is a stress-case diagnostic, not a 500-replicate Monte Carlo
coverage simulation.

The deterministic cycle diagnostics are also available:

```bash
python run_simulation_analysis.py --case two_cycle
python run_simulation_analysis.py --case three_cycle
```

For a fair comparison, the analysis computes the auxiliary p-values once per
history and horizon, then thresholds the same candidate values over all alpha
values. The D-weighted aggregate uses the same auxiliary rows across the alpha
grid.

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
    run_simulation_analysis.py \
    plot_simulation_results.py \
    emmett_tests.py
python -m unittest discover -s tests
python simulation_study.py --quick
python run_simulation_analysis.py \
    --input simulation_results/input/sim1_sequences_quick.npz \
    --quick
python plot_simulation_results.py \
    --summary simulation_results/output/sim1_summary_quick.csv
python run_simulation_analysis.py --case case2 --quick
python run_simulation_analysis.py --case sweden --quick
python quick_meeting_demo.py
python run_demo.py
git diff --check
```

The tests cover adjacency validation, candidate enumeration, i-block
decomposition, indexed block-order permutations, factorial cardinalities,
tie-handling modes, D! weights, D weights, shared-row aggregation, p-value
ranges, `q_tilde` ranges, seed reproducibility, public signatures without
`random_seed`, final prediction-set path lengths, successful demo execution,
Markov simulation helpers, same-history and nested-future slicing, quick input
generation, quick analysis output, raw set-size summaries, diagnostic cases,
and plot/table creation from saved CSV files.
