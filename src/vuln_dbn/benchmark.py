from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .bn import learn_model
from .io import write_json
from .select import rank_nodes


def benchmark_scaling(
    dataset_csv: str | Path,
    output_csv: str | Path,
    metadata_columns: list[str],
    sink: str,
    n_values: list[int],
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Times exactly one cold-start hill-climbing iteration (max_iter=1) at each requested
    variable count N, ranking once and reusing the top-N slice at each point.

    Deliberately not a time-to-convergence benchmark: how many iterations a search needs
    before hitting its stopping criteria varies a lot and isn't comparable across N (small
    N can exhaust its legal search space in a handful of steps). A single iteration's cost
    is what we actually diagnosed as the bottleneck (candidate-edge enumeration is O(N^2),
    paid fresh every iteration regardless of caching), so it's the clean, comparable number
    to fit a scaling curve against."""
    frame = pd.read_csv(dataset_csv)
    ranking = rank_nodes(dataset_csv, metadata_columns, sink, random_state=random_state)
    ranked_nodes = ranking.sort_values("mutual_information_nats", ascending=False)["node"].tolist()

    rows = []
    for n in sorted(set(n_values)):
        if n > len(ranked_nodes):
            rows.append({"n": n, "elapsed_seconds": None, "status": f"only {len(ranked_nodes)} nodes available"})
            continue
        nodes = ranked_nodes[:n]
        subset = frame[nodes + [sink]].copy()
        start = time.perf_counter()
        try:
            learn_model(
                subset,
                sink=sink,
                score=score,
                max_indegree=max_indegree,
                equivalent_sample_size=equivalent_sample_size,
                show_progress=False,
                max_iter=1,
            )
            elapsed = time.perf_counter() - start
            status = "ok"
        except Exception as exc:
            elapsed = time.perf_counter() - start
            status = f"error: {exc}"
        rows.append({"n": n, "elapsed_seconds": elapsed, "status": status})
        print(f"n={n}: {elapsed:.2f}s ({status})", flush=True)

    report = pd.DataFrame(rows)
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    return report


def fit_quadratic(report: pd.DataFrame) -> dict:
    """Fits elapsed = a*n^2 + b*n + c to the ok rows, for extrapolating to untested N."""
    ok = report[report["status"] == "ok"].dropna(subset=["elapsed_seconds"])
    if len(ok) < 3:
        raise ValueError("Need at least 3 successful data points to fit a quadratic")
    coeffs = np.polyfit(ok["n"], ok["elapsed_seconds"], 2)
    a, b, c = (float(x) for x in coeffs)
    return {"a": a, "b": b, "c": c, "formula": "elapsed_seconds = a*n^2 + b*n + c", "fit_points": len(ok)}


def write_fit(report_csv: str | Path, output_json: str | Path) -> dict:
    report = pd.read_csv(report_csv)
    fit = fit_quadratic(report)
    write_json(output_json, fit)
    return fit
