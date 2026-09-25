from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from .constants import ANCHOR_ID


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    count = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(count, dtype=float)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def mutual_information_report(
    dataset_csv: str | Path,
    output_csv: str | Path,
    metadata_columns: list[str],
    sink: str,
    permutations: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank nodes by anchor-aware permutation p-values against a chosen sink/target.

    Anchor membership (not row identity) is permuted, since transitions from the same
    anchor are not independent -- shuffling individual rows would overstate significance."""
    frame = pd.read_csv(dataset_csv)
    node_columns = [column for column in frame.columns if column not in metadata_columns and column != sink]
    truth = frame[sink].astype("category").cat.codes.to_numpy()
    observed = np.array([mutual_info_score(frame[node], truth) for node in node_columns])
    exceedances = np.ones(len(node_columns), dtype=int)
    rng = np.random.default_rng(random_state)
    anchor_indices = [group.index.to_numpy() for _, group in frame.groupby(ANCHOR_ID, sort=False)]

    for _ in range(permutations):
        permuted = truth.copy()
        for indices in anchor_indices:
            if len(indices) > 1:
                permuted[indices] = rng.permutation(permuted[indices])
        scores = np.array([mutual_info_score(frame[node], permuted) for node in node_columns])
        exceedances += scores >= observed

    p_values = exceedances / (permutations + 1)
    q_values = _benjamini_hochberg(p_values)
    support = np.array([(frame[node] != frame[node].mode().iloc[0]).sum() for node in node_columns])
    report = pd.DataFrame(
        {
            "node": node_columns,
            "mutual_information_nats": observed,
            "support": support,
            "permutation_p": p_values,
            "bh_q": q_values,
        }
    ).sort_values(["mutual_information_nats", "support"], ascending=[False, False])
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)
    return report
