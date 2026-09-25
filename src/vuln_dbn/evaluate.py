from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import GroupShuffleSplit

from .bn import learn_model
from .io import write_json


def _probability_of(inference, row: pd.Series, evidence_nodes: list[str], sink: str, positive_state: str) -> float:
    evidence = {node: int(row[node]) for node in evidence_nodes}
    query = inference.query([sink], evidence=evidence, show_progress=False)
    states = list(query.state_names[sink])
    return float(query.values[states.index(positive_state)])


def evaluate_holdout(
    dataset_csv: str | Path,
    output_json: str | Path,
    metadata_columns: list[str],
    sink: str,
    group_column: str = "project",
    positive_state: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
) -> dict:
    frame = pd.read_csv(dataset_csv)
    if group_column not in frame:
        raise ValueError(f"Unknown split group {group_column!r}")
    all_nodes = [column for column in frame.columns if column not in metadata_columns]
    if sink not in all_nodes:
        raise ValueError(f"Sink column {sink!r} not found in dataset")
    sink_states = sorted(frame[sink].astype(str).unique())
    if positive_state is None:
        if len(sink_states) != 2:
            raise ValueError(f"Sink {sink!r} has {len(sink_states)} states; pass positive_state explicitly")
        positive_state = sink_states[-1]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_index, test_index = next(splitter.split(frame, groups=frame[group_column]))
    train, test = frame.iloc[train_index], frame.iloc[test_index]
    evidence_nodes = [n for n in all_nodes if n != sink and train[n].nunique(dropna=False) > 1]
    if not evidence_nodes:
        raise ValueError("No nonconstant nodes remain in the training partition")
    model = learn_model(
        train[evidence_nodes + [sink]],
        sink=sink,
        score=score,
        max_indegree=max_indegree,
        equivalent_sample_size=equivalent_sample_size,
        show_progress=False,
    )
    from pgmpy.inference import VariableElimination

    inference = VariableElimination(model)
    cache: dict[tuple[int, ...], float] = {}
    probabilities_list = []
    for _, row in test.iterrows():
        key = tuple(int(row[node]) for node in evidence_nodes)
        if key not in cache:
            cache[key] = _probability_of(inference, row, evidence_nodes, sink, positive_state)
        probabilities_list.append(cache[key])
    probabilities = np.array(probabilities_list)
    truth = (test[sink].astype(str) == positive_state).astype(int).to_numpy()
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    observed, predicted = calibration_curve(truth, probabilities, n_bins=10, strategy="uniform")
    result = {
        "sink": sink,
        "positive_state": positive_state,
        "split_group": group_column,
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "train_groups": int(train[group_column].nunique()),
        "test_groups": int(test[group_column].nunique()),
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "brier": float(brier_score_loss(truth, probabilities)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "pr_auc": float(average_precision_score(truth, probabilities)),
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "calibration": [
            {"mean_predicted": float(pred), "observed_fraction": float(obs)}
            for pred, obs in zip(predicted, observed)
        ],
        "nodes_used": len(evidence_nodes),
        "sink_parents": sorted(model.get_parents(sink)),
        "isolated_nodes_with_marginal_cpds": model.graph.get("isolated_cpd_nodes", []),
    }
    write_json(output_json, result)
    return result
