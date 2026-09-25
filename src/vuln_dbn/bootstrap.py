from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .bn import learn_model
from .constants import ANCHOR_ID
from .io import write_json


def bootstrap_edges(
    dataset_csv: str | Path,
    output_json: str | Path,
    metadata_columns: list[str],
    sink: str | None = None,
    repetitions: int = 100,
    random_state: int = 42,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
) -> dict:
    """Anchor-preserving bootstrap frequencies for directed edges and adjacencies.

    Resampling by anchor_id (not by row) keeps every transition from the same commit-history
    window together, since transitions within one anchor are not independent observations."""
    frame = pd.read_csv(dataset_csv)
    node_columns = [column for column in frame.columns if column not in metadata_columns]
    anchor_groups = {anchor_id: group for anchor_id, group in frame.groupby(ANCHOR_ID, sort=False)}
    anchor_ids = np.array(list(anchor_groups))
    if not len(anchor_ids):
        raise ValueError("Dataset has no anchors")
    rng = np.random.default_rng(random_state)
    directed: Counter[tuple[str, str]] = Counter()
    adjacency: Counter[tuple[str, str]] = Counter()
    failures: list[dict] = []
    completed = 0

    for repetition in range(repetitions):
        sampled_ids = rng.choice(anchor_ids, size=len(anchor_ids), replace=True)
        sampled = pd.concat([anchor_groups[anchor_id] for anchor_id in sampled_ids], ignore_index=True)
        active_nodes = [node for node in node_columns if sampled[node].nunique(dropna=False) > 1]
        try:
            model = learn_model(
                sampled[active_nodes],
                sink=sink,
                score=score,
                max_indegree=max_indegree,
                equivalent_sample_size=equivalent_sample_size,
                show_progress=False,
            )
            completed += 1
            for source, target in model.edges():
                directed[(source, target)] += 1
                adjacency[tuple(sorted((source, target)))] += 1
        except Exception as exc:
            failures.append({"repetition": repetition, "error": str(exc)})

    if completed == 0:
        raise RuntimeError("Every bootstrap structure-learning run failed")
    report = {
        "requested_repetitions": repetitions,
        "completed_repetitions": completed,
        "sink": sink,
        "directed_edges": [
            {"source": source, "target": target, "frequency": count / completed, "count": count}
            for (source, target), count in directed.most_common()
        ],
        "adjacencies": [
            {"node_a": a, "node_b": b, "frequency": count / completed, "count": count}
            for (a, b), count in adjacency.most_common()
        ],
        "failures": failures,
        "note": "Adjacency stability is more robust than direction when structures are Markov equivalent.",
    }
    write_json(output_json, report)
    return report
