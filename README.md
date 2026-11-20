# hep-simbench

Validation and benchmarking of a fast simulation against its stock (full) simulation. The question it answers for
every commit of a fast simulation: *is it still right, and how much faster is it, in both float precisions?*

| Module | What it does |
|---|---|
| `hepsimbench.score` | scores a fast arm against stock on the same generated events: **validity** (histograms flagged against a reseeded-stock floor, yields with tolerances and a statistical-power rule), **throughput multiplier** (warm rate over the stock rate per single-core grid slot), **cost per event** (private scoring only); both float32 and float64 required. NumPy only, so a Codabench worker runs it unchanged |
| `hepsimbench.bench` | warm-rate timer: three passes in one process, device synchronised before the clock stops; setup, warm rate, N10 (the job size at which setup is 10 % of the time), device label (GPU model and MIG profile, or CPU model), JAX compiles per pass, fresh inputs per pass; experiment hooks through `Extension` |
| `hepsimbench.approximations` | every model that stands in for stock declared once, with its levels, a `default`/`fast`/`fidelity` preset scheme, and a provenance record per run |
| `hepsimbench.stats` | robust residual statistics and most-probable-value fits |
| `hepsimbench.from_root` | 1D ROOT histograms (validation or DQM files) to the scorer's `hists.json`, one shard per file |
| `hepsimbench.codabench` | a Codabench results-submission bundle with leaderboard columns per workflow and dtype |
| `hepsimbench.synthetic` | a synthetic reference and three submissions (good, shifted, slow), used by the tests and the dry run |

Origin: extracted from [mu2e-fastsim](https://github.com/RobMina/mu2e-fastsim), which keeps its task table, its
experiment-specific converters and its GPU launch settings and uses this package for the rest (see NOTICE).

## Scoring

```bash
python -m hepsimbench.synthetic /tmp/example                     # a toy reference and submissions
python -m hepsimbench.score /tmp/example/submissions/good /tmp/example/reference --out scores.json
```

Layout, one directory per workflow:

```text
submission/<W>/<dtype>/hists.json     the fast arm (dtype = float32 and float64, both required)
submission/<W>/<dtype>/bench.json     hepsimbench.bench report of the same workflow and dtype
reference/<W>/stock/hists.json        the stock arm on the same generated events
reference/<W>/floor/hists.json        stock with another seed (optional; without it the flag threshold is 2)
reference/<W>/reference.json          stock_rate_per_slot, boundary, mode, yield_tolerance, ... (below)
```

| Histogram kind | hists.json entry | Statistic |
|---|---|---|
| counts | `edges` + `counts`, or `edges` + `shards` | mode `same_events` (both arms on the same events, no scaling), `shape` (scaled to the reference total), or paired per shard when both arms carry shards (with `shape`: the paired shape test) |
| values | `edges` + `values` + `errors` | sum (v1 - v2)^2 / (e1^2 + e2^2): tables and derived quantities with their own errors |
| efficiency | `edges` + `passed` + `total` | per-bin binomial difference |

A histogram is flagged when chi2/ndf > max(2, 3 x floor) and p < 1e-3. A yield (a count, per-shard counts, or a
measured `{"value", "error"}`) passes when |ratio - 1| - 3 se <= its tolerance. Validity per workflow and dtype is
(1 - flagged fraction) x (fraction of yields that pass); the headline is the mean over workflows and dtypes, the
throughput headline the geometric mean.

Optional keys of `reference.json`:

| Key | What the scorer does |
|---|---|
| `yield_max_se` `{name: se}` | a yield with a larger standard error fails as underpowered, so a small sample cannot pass a tight tolerance |
| `yield_tolerances` `{name: tol}` | per-yield tolerances |
| `classifier` `{"file", "seed"}` | a fixed logistic regression on quadratic features of per-object feature vectors (`features.npz`), AUC on a held-out half; `cls_score = clip(2 (1 - AUC), 0, 1)`, 1 = indistinguishable; reported beside validity |
| `peak_fit` `{name: {"start_above", "start_width"}}` | a Gaussian-core peak fit per arm, the ratio scored as a yield with a paired bootstrap error over shards |
| `track_agreement` `{"file", "threshold_sigma", "min_fraction"}` | fraction of stock tracks whose fast fit lies within threshold x sigma in every parameter; pulls against truth |
| `scaling_batches` | throughput multiplier per batch size and the crossover batch |
| `group` + `part` (`dev`, `place`, `design`) | portability ratio (mean placement validity / dev validity) and adaptation score per group |
| `capture_budget` | an arm over its declared training-data budget is an error |
| `max_memory_mb_per_thread` | `memory_ok` from the bench report |
| `stock_rate_per_slot: null` | validity only, no throughput |

A bench report with `"host": "geant4"` (a model running inside a Geant4 job) is scored against its own
`baseline_rate` (model off, same node and primaries) instead of the per-slot stock rate.

## Timing

```bash
python -m hepsimbench.bench mypkg.chain:run --events 100000 --prepare mypkg.chain:load --fresh-inputs \
    --dtype float32 --workflow W1 --boundary "input -> output" --out bench.json
```

The entry point returns its device outputs so the timer can wait for them (`jax.block_until_ready`, CuPy
synchronise). An experiment wraps `bench.main(argv, prog=..., extensions=[...])` to add its own options (for example
launch flags that must be set before a backend exists) and report fields.

## Approximations

A package declares `mypkg/approximations.py` with one `Workflow` of `Approximation`s (levels, the default level that
is today's behaviour, what each other level sets). Runs select levels with `--approx-preset fast|fidelity|default`
and `--approx NAME=LEVEL`; the timer records the selection; `python -m hepsimbench.approximations mypkg` prints the
documentation table.

## Codabench

```bash
python -m hepsimbench.codabench build                 # synthetic example; --reference DIR for a private reference
```

## Rules this package keeps

- Both float32 and float64 are required for every workflow.
- A reference's stock histograms are an experiment's data: keep real references out of public repositories; the
  synthetic example here has no experiment content.
- Cost per event (SU) is scored only with `--private` and never on a leaderboard.

## Tests

`python -m pytest` (synthetic data only). CI runs Python 3.10-3.14 and a Codabench dry run.

## Citing

Please cite the release you used: `CITATION.cff` holds the metadata (GitHub's "Cite this repository" renders it), and
each tagged release is archived on Zenodo with its own DOI.
