# vuln-dbn

Dynamic Bayesian network (2-slice) structure learning over the natural commit-history
change-type tables produced by [vuln-commit-history](../vuln-commit-history). Adapted from
[vuln-change-direction](../vuln-change-direction)'s static Bayesian network pipeline
(`src/megavul_bn`): same `pgmpy` hill-climbing/checkpoint/bootstrap machinery, with the
"OUTCOME must be a sink" constraint generalized to "no edge from a later (`__t1`) slice
variable back into an earlier (`__t0`) one."

## Input

A `wide_two_slice_table.csv` from `vuln-commit-history`'s `dbn_encode.build_wide_two_slice_table`:
one row per consecutive pair of history slices, columns `anchor_id`, `project`,
`slice_offset_t0`, `slice_offset_t1`, `TRANSITION_LABEL__t0`, `TRANSITION_LABEL__t1`, and a
`CT__*__t0`/`CT__*__t1` pair per retained change-type feature.

## Two experiments, one `--sink` flag

- **Pure structure learning** (`learn` with no `--sink`): only the temporal constraint
  applies. Describes how change types co-occur and evolve across slices, no single
  prediction target.
- **Targeted** (`learn --sink TRANSITION_LABEL__t1`): additionally constrains that node to be
  a pure sink, matching the static repo's OUTCOME pattern. Unlocks `evaluate`/`predict`/`mi`,
  which all require a designated sink.

`bootstrap` and `dot` work in either mode.

## Setup

```
uv sync   # or: pip install -e .
```

## Usage

```
vuln-dbn learn --dataset wide_two_slice_table.csv --model model.joblib --summary summary.json \
    --checkpoint checkpoint.joblib   # omit --sink for the pure-structure-learning experiment

vuln-dbn learn --dataset wide_two_slice_table.csv --model model.joblib --summary summary.json \
    --sink TRANSITION_LABEL__t1 --checkpoint checkpoint.joblib

vuln-dbn evaluate --dataset wide_two_slice_table.csv --output eval.json --sink TRANSITION_LABEL__t1
vuln-dbn mi --dataset wide_two_slice_table.csv --output mi.csv --sink TRANSITION_LABEL__t1
vuln-dbn bootstrap --dataset wide_two_slice_table.csv --output bootstrap.json --sink TRANSITION_LABEL__t1
vuln-dbn dot --summary summary.json --output graph.dot

# evidence.json: {"evidence": {"CT__SOME_TYPE__t0": 1, "TRANSITION_LABEL__t0": "OTHER"}}
# unlisted nodes are marginalized, not assumed absent -- not every node is a binary indicator
vuln-dbn predict --model model.joblib --evidence evidence.json --output prediction.json
```

Structure learning over the full feature set (hundreds of `__t0`/`__t1` columns) is
CPU-heavy -- `--checkpoint` makes a long run resumable (Ctrl-C-safe), which matters most on a
remote VM you might need to reconnect to.
