from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from .checkpoint_hillclimb import CheckpointHillClimbSearch
from .constants import T0_SUFFIX, T1_SUFFIX
from .io import write_json


def forbidden_temporal_edges(
    columns: Iterable[str], t0_suffix: str = T0_SUFFIX, t1_suffix: str = T1_SUFFIX
) -> set[tuple[str, str]]:
    """No edge may run from a later (t1) slice variable back into an earlier (t0) one."""
    t0_vars = [c for c in columns if c.endswith(t0_suffix)]
    t1_vars = [c for c in columns if c.endswith(t1_suffix)]
    return {(t1, t0) for t1 in t1_vars for t0 in t0_vars}


def forbidden_sink_edges(columns: Iterable[str], sink: str) -> set[tuple[str, str]]:
    return {(sink, column) for column in columns if column != sink}


def _load_pgmpy():
    try:
        from pgmpy.estimators import BDeu, BayesianEstimator, ExpertKnowledge, HillClimbSearch
        from pgmpy.models import DiscreteBayesianNetwork
    except ImportError as exc:
        raise RuntimeError("Bayesian-network commands require the project dependencies") from exc
    return BDeu, BayesianEstimator, ExpertKnowledge, HillClimbSearch, DiscreteBayesianNetwork


def _bdeu_marginal(series: pd.Series, equivalent_sample_size: float) -> tuple[list, list[float]]:
    """Estimate a BDeu-smoothed marginal distribution for an isolated node."""
    if equivalent_sample_size <= 0:
        raise ValueError("equivalent_sample_size must be greater than zero")
    if series.isna().any():
        raise ValueError(f"Missing values are not supported for isolated node {series.name!r}")
    if isinstance(series.dtype, pd.CategoricalDtype):
        states = list(series.cat.categories)
    else:
        states = sorted(series.unique().tolist(), key=str)
    if not states:
        raise ValueError(f"Cannot estimate a CPD for empty node {series.name!r}")
    counts = series.value_counts(sort=False).reindex(states, fill_value=0).astype(float)
    alpha = equivalent_sample_size / len(states)
    probabilities = (counts.to_numpy() + alpha) / (float(counts.sum()) + equivalent_sample_size)
    return states, probabilities.tolist()


def _add_missing_isolated_cpds(model, data: pd.DataFrame, equivalent_sample_size: float) -> list[str]:
    """pgmpy 1.0 drops isolated nodes during Bayesian estimation; refill them with their marginal."""
    from pgmpy.factors.discrete import TabularCPD

    missing = [node for node in model.nodes() if model.get_cpds(node) is None]
    for node in missing:
        if model.degree(node) != 0:
            raise ValueError(f"Connected node {node!r} is missing its CPD; this is not an isolated-node case")
        states, probabilities = _bdeu_marginal(data[node], equivalent_sample_size)
        cpd = TabularCPD(
            variable=node,
            variable_card=len(states),
            values=np.asarray(probabilities, dtype=float).reshape(len(states), 1),
            state_names={node: states},
        )
        model.add_cpds(cpd)
    model.graph["isolated_cpd_nodes"] = sorted(missing)
    return missing


def learn_model(
    data: pd.DataFrame,
    sink: str | None = None,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
    show_progress: bool = True,
    epsilon: float = 1e-4,
    max_iter: int = 1_000_000,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
):
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    BDeu, BayesianEstimator, ExpertKnowledge, HillClimbSearch, DiscreteBayesianNetwork = _load_pgmpy()
    model_data = data.copy()
    for column in model_data.columns:
        model_data[column] = model_data[column].astype("category")

    forbidden = forbidden_temporal_edges(model_data.columns)
    if sink is not None:
        forbidden |= forbidden_sink_edges(model_data.columns, sink)
    knowledge = ExpertKnowledge(forbidden_edges=forbidden)

    if score == "bdeu":
        structure_score = BDeu(model_data, equivalent_sample_size=equivalent_sample_size)
    else:
        structure_score = score

    dag = CheckpointHillClimbSearch(model_data).estimate(
        scoring_method=structure_score,
        expert_knowledge=knowledge,
        max_indegree=max_indegree,
        show_progress=show_progress,
        epsilon=epsilon,
        max_iter=max_iter,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )

    for source, target in dag.edges():
        if source.endswith(T1_SUFFIX) and target.endswith(T0_SUFFIX):
            raise AssertionError(f"Temporal constraint was violated by structure learning: {source} -> {target}")
    if sink is not None and any(source == sink for source, _ in dag.edges()):
        raise AssertionError("Sink constraint was violated by structure learning")

    model = DiscreteBayesianNetwork(dag.edges())
    model.graph.update(dag.graph)
    model.add_nodes_from(model_data.columns)
    estimator = BayesianEstimator(model, model_data)
    model.add_cpds(*estimator.get_parameters(prior_type="BDeu", equivalent_sample_size=equivalent_sample_size))
    _add_missing_isolated_cpds(model, model_data, equivalent_sample_size)
    model.check_model()
    return model


def train_from_csv(
    dataset_csv: str | Path,
    model_path: str | Path,
    summary_path: str | Path,
    metadata_columns: list[str],
    sink: str | None = None,
    score: str = "bic-d",
    max_indegree: int = 4,
    equivalent_sample_size: float = 5.0,
    epsilon: float = 1e-4,
    max_iter: int = 1_000_000,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> dict:
    frame = pd.read_csv(dataset_csv)
    node_columns = [column for column in frame.columns if column not in metadata_columns]
    if not node_columns:
        raise ValueError("Dataset has no modelable columns")
    if sink is not None and sink not in node_columns:
        raise ValueError(f"Sink column {sink!r} not found in dataset")
    model_data = frame[node_columns].copy()
    model = learn_model(
        model_data,
        sink=sink,
        score=score,
        max_indegree=max_indegree,
        equivalent_sample_size=equivalent_sample_size,
        epsilon=epsilon,
        max_iter=max_iter,
        checkpoint_path=checkpoint_path,
        resume=resume,
    )
    bundle = {
        "model": model,
        "nodes": node_columns,
        "sink": sink,
        "score": score,
        "max_indegree": max_indegree,
        "equivalent_sample_size": equivalent_sample_size,
        "epsilon": epsilon,
        "max_iter": max_iter,
    }
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    edges = sorted([list(edge) for edge in model.edges()])
    summary = {
        "nodes": sorted(model.nodes()),
        "edges": edges,
        "sink": sink,
        "sink_parents": sorted(model.get_parents(sink)) if sink else None,
        "sink_children": sorted(model.get_children(sink)) if sink else None,
        "sink_constraint_satisfied": (len(model.get_children(sink)) == 0) if sink else None,
        "isolated_nodes_with_marginal_cpds": model.graph.get("isolated_cpd_nodes", []),
        "interpretation": "Learned directions are statistical, subject to the temporal (and, if set, sink) "
        "constraints -- not independent causal claims.",
        "epsilon": epsilon,
        "max_iter": max_iter,
    }
    write_json(summary_path, summary)
    return summary


def predict(model_path: str | Path, evidence: dict[str, object]) -> dict:
    """evidence maps a subset of node names to their observed state (e.g. 1/0 for a CT__
    presence node, or "FIX"/"OTHER" for TRANSITION_LABEL__t0) -- unlisted nodes are marginalized
    out, not assumed absent, since not every node here is a binary presence indicator."""
    from pgmpy.inference import VariableElimination

    bundle = joblib.load(model_path)
    model = bundle["model"]
    sink = bundle["sink"]
    if sink is None:
        raise ValueError("This model was learned without a sink; there is no target to predict")
    valid_nodes = set(bundle["nodes"]) - {sink}
    unknown = sorted(set(evidence) - valid_nodes)
    known_evidence = {node: state for node, state in evidence.items() if node in valid_nodes}
    query = VariableElimination(model).query([sink], evidence=known_evidence, show_progress=False)
    states = query.state_names[sink]
    probabilities = {str(state): float(query.values[index]) for index, state in enumerate(states)}
    return {"probabilities": probabilities, "unknown_nodes": unknown, "evidence_nodes": len(known_evidence)}


def write_prediction(model_path: str | Path, evidence_json: str | Path, output_json: str | Path) -> dict:
    from .io import read_json

    payload = read_json(evidence_json)
    result = predict(model_path, payload.get("evidence", {}))
    write_json(output_json, result)
    return result
