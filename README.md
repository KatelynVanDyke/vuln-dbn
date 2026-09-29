# vuln-dbn

Dynamic Bayesian network (2-slice) structure learning over the natural commit-history
change-type tables produced by [vuln-commit-history](../vuln-commit-history). Adapted from
[vuln-change-direction](../vuln-change-direction)'s static Bayesian network pipeline
(`src/megavul_bn`): same `pgmpy` hill-climbing/checkpoint/bootstrap machinery, with the
"OUTCOME must be a sink" constraint generalized to "no edge from a later (`__t1`) slice
variable back into an earlier (`__t0`) one."

## 1. Setup

```
git clone git@github.com:KatelynVanDyke/vuln-dbn.git && cd vuln-dbn && uv sync
```

That's it — `data/java/wide_two_slice_table.csv` is already committed in the repo, so there's
no separate data-transfer step for the Java dataset. (C and C++ versions will land in
`data/c/` / `data/cpp/` the same way once that extraction finishes upstream.)

Every command below is run from this directory (the one with `pyproject.toml` in it), using
`uv run` so you don't need to separately activate the virtualenv:

```
uv run vuln-dbn <command> ...
```

## 2. The two experiments

Both share the same `learn` command; the only difference is the `--sink` flag.

**Experiment A — pure structure learning** (no sink, only the temporal constraint):
```
uv run vuln-dbn learn --dataset data/java/wide_two_slice_table.csv --model models/java_structure.joblib --summary results/java_structure_summary.json --checkpoint results/java_structure.checkpoint
```
Describes how change types co-occur and evolve across slices. No single prediction target.

**Experiment B — targeted** (`TRANSITION_LABEL__t1` constrained as a pure sink, like the
static repo's `OUTCOME`):
```
uv run vuln-dbn learn --dataset data/java/wide_two_slice_table.csv --model models/java_sink.joblib --summary results/java_sink_summary.json --sink TRANSITION_LABEL__t1 --checkpoint results/java_sink.checkpoint
```
This unlocks the target-dependent commands below.

Run A and B as two separate processes if you want them in parallel — nothing else in this
repo benefits from extra CPUs (see §5), so this is the one place multiple cores actually help.

## 3. Sizing N before committing to a real run

Learning over all 934 feature columns is expensive: `HillClimbSearch` re-enumerates every
candidate edge from scratch each iteration (O(N^2), ~880,000 candidates at N=934), and that
cost is paid fresh every single iteration regardless of score caching. Before burning a long
job on the full variable set, measure how that cost actually scales on your hardware and pick
an N your walltime budget can afford:

```
uv run vuln-dbn benchmark --dataset data/java/wide_two_slice_table.csv --output results/benchmark.csv --sink TRANSITION_LABEL__t1 --n-values 20,50,100,200,400
```

This times exactly one cold hill-climbing iteration at each N (not time-to-convergence --
iteration counts to convergence vary too much across N to compare cleanly; a single
iteration's cost is the actual bottleneck we diagnosed, so it's the clean, comparable number).
Then fit a quadratic to extrapolate to N values you didn't test directly:

```
uv run vuln-dbn fit --report results/benchmark.csv --output results/benchmark_fit.json
```

Gives `elapsed_seconds = a*n^2 + b*n + c` -- plug in any N to estimate per-iteration cost, and
from there a rough total-runtime budget (iteration cost x however many iterations you can
afford). On real Java data measured locally (N=10..80), this fit extrapolated to ~890 seconds
(~15 minutes) per iteration at the full N=934 -- consistent with what was actually observed
on a live Hellbender job stalling on a single iteration for 20+ minutes. Trust the fit over
intuition here.

Once you've picked an N, build the actual reduced dataset (keeps the top-N MI-ranked nodes
against the sink, same schema as the full table so every other command still works on it):

```
uv run vuln-dbn select-features --dataset data/java/wide_two_slice_table.csv --output data/java/wide_two_slice_table_top100.csv --sink TRANSITION_LABEL__t1 --n 100
```

## 4. Downstream analysis (Experiment B's model only)

```
uv run vuln-dbn evaluate --dataset data/java/wide_two_slice_table.csv --output results/java_eval.json --sink TRANSITION_LABEL__t1
```
```
uv run vuln-dbn mi --dataset data/java/wide_two_slice_table.csv --output results/java_mi.csv --sink TRANSITION_LABEL__t1
```
```
uv run vuln-dbn bootstrap --dataset data/java/wide_two_slice_table.csv --output results/java_bootstrap.json --sink TRANSITION_LABEL__t1
```
```
uv run vuln-dbn dot --summary results/java_sink_summary.json --output results/java_sink.dot
```

`predict` takes an evidence JSON file, e.g. `{"evidence": {"CT__SOME_TYPE__t0": 1, "TRANSITION_LABEL__t0": "OTHER"}}` -- unlisted nodes are marginalized, not assumed absent, since not every node is a binary indicator:
```
uv run vuln-dbn predict --model models/java_sink.joblib --evidence evidence.json --output results/prediction.json
```

`bootstrap`/`mi`/`dot` also work on a no-sink (Experiment A) model/dataset; `evaluate` and
`predict` specifically need a sink, since they're inherently about a prediction target.

## 5. Compute sizing (e.g. for a Hellbender / SLURM job)

- **CPU only, no GPU.** `HillClimbSearch`'s candidate-edge scan is a plain single-threaded
  Python generator (verified against the installed pgmpy source) -- no `n_jobs`, nothing a
  GPU would accelerate. A GPU allocation would sit idle.
- **1 node.** Nothing here is distributed across machines.
- **Memory: request ~8GB for two concurrent experiments.** Measured peak for one `learn` call
  on the full 934-column Java dataset was ~454MB locally; running Experiments A and B at once
  on a real SLURM job measured ~941MB combined (`sstat`'s `MaxRSS`) -- 8GB gives comfortable
  headroom for both at once. Memory is cheap to over-request on a shared cluster, unlike
  CPU/GPU, which cost queue priority the way over-asking for cores does.
- **Walltime is the real constraint, and full-N is likely impractical.** A live Hellbender job
  confirmed a single hill-climbing iteration at the full N=934 can take upwards of 15-20
  minutes -- not a "let it run overnight" problem, closer to a "this could take weeks"
  problem if convergence needs hundreds of iterations. Don't just throw more walltime at it:
  use §3's `benchmark`/`fit`/`select-features` workflow to work at a tractable N instead.
  `--checkpoint path --resume` still matters regardless of N, for both timeouts and
  preemption on a backfill/requeue-style partition -- resubmitting the exact same command
  with `--resume` picks up where it left off.
- **CPUs: 2 is enough to run Experiments A and B side by side** as separate processes (a
  single `learn` call won't use more than one core). Only go higher than that if you also want
  `bootstrap` parallelized across its resampling repetitions -- that's not implemented yet, so
  ask before relying on it.

## 6. Data format

`wide_two_slice_table.csv`: one row per consecutive pair of history slices, columns
`anchor_id`, `project`, `slice_offset_t0`, `slice_offset_t1`, `TRANSITION_LABEL__t0`,
`TRANSITION_LABEL__t1`, and a `CT__*__t0`/`CT__*__t1` pair per retained change-type feature.
`long_table_metadata.json` alongside it records how the feature set was built (`min_support`,
transition/anchor counts) in `vuln-commit-history`.
