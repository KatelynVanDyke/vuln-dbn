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
repo benefits from extra CPUs (see §4), so this is the one place multiple cores actually help.

## 3. Downstream analysis (Experiment B's model only)

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

## 4. Compute sizing (e.g. for a Hellbender / SLURM job)

- **CPU only, no GPU.** `HillClimbSearch`'s candidate-edge scan is a plain single-threaded
  Python generator (verified against the installed pgmpy source) -- no `n_jobs`, nothing a
  GPU would accelerate. A GPU allocation would sit idle.
- **1 node.** Nothing here is distributed across machines.
- **Memory: request ~4GB.** Measured peak on the real 940-column Java dataset was ~454MB;
  4GB gives comfortable headroom. Memory is cheap to over-request on a shared cluster --
  unlike CPU/GPU, it doesn't cost queue priority the way over-asking for cores does.
- **Walltime: budget generously, and use `--checkpoint`.** The first iteration alone (scoring
  every candidate edge from scratch, ~880,000 of them for 940 variables) took several minutes
  and hadn't finished in local testing; a full run to convergence is plausibly hours or more.
  `--checkpoint path --resume` makes a run resumable across multiple job submissions if your
  cluster caps individual job walltime -- re-run the exact same command with `--resume` added.
- **CPUs: 2 is enough to run Experiments A and B side by side** as separate processes (a
  single `learn` call won't use more than one core). Only go higher than that if you also want
  `bootstrap` parallelized across its resampling repetitions -- that's not implemented yet, so
  ask before relying on it.

## 5. Data format

`wide_two_slice_table.csv`: one row per consecutive pair of history slices, columns
`anchor_id`, `project`, `slice_offset_t0`, `slice_offset_t1`, `TRANSITION_LABEL__t0`,
`TRANSITION_LABEL__t1`, and a `CT__*__t0`/`CT__*__t1` pair per retained change-type feature.
`long_table_metadata.json` alongside it records how the feature set was built (`min_support`,
transition/anchor counts) in `vuln-commit-history`.
