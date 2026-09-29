from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from .significance import mutual_information_report


def rank_nodes(
    dataset_csv: str | Path, metadata_columns: list[str], sink: str, random_state: int = 42
) -> pd.DataFrame:
    """MI ranking only (permutations=0 skips the p-value loop entirely -- the observed MI
    score itself doesn't depend on permutations, only its significance test does, so this
    is the cheap way to get a ranking without paying for 1000 permutation runs)."""
    with tempfile.TemporaryDirectory() as tmp:
        discard_csv = Path(tmp) / "ranking.csv"
        return mutual_information_report(
            dataset_csv, discard_csv, metadata_columns, sink, permutations=0, random_state=random_state
        )


def select_top_features(
    dataset_csv: str | Path,
    output_csv: str | Path,
    metadata_columns: list[str],
    sink: str,
    n: int,
    random_state: int = 42,
) -> list[str]:
    """Writes a reduced dataset (same schema, just fewer feature columns) keeping the
    top-n MI-ranked nodes plus metadata and the sink -- still a valid input to every other
    vuln-dbn command, just over a smaller, more tractable variable set."""
    frame = pd.read_csv(dataset_csv)
    ranking = rank_nodes(dataset_csv, metadata_columns, sink, random_state=random_state)
    top_nodes = ranking.sort_values("mutual_information_nats", ascending=False)["node"].head(n).tolist()
    reduced = frame[metadata_columns + [sink] + top_nodes]
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_csv(destination, index=False)
    return top_nodes
