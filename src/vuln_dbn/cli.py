from __future__ import annotations

import argparse
import json

from .bn import train_from_csv, write_prediction
from .bootstrap import bootstrap_edges
from .constants import METADATA_COLUMNS
from .evaluate import evaluate_holdout
from .graph import write_dot
from .significance import mutual_information_report


def _metadata_columns(raw: str | None) -> list[str]:
    if raw is None:
        return list(METADATA_COLUMNS)
    return [c.strip() for c in raw.split(",") if c.strip()]


def _add_metadata_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata-columns",
        help=f"Comma-separated non-modeled columns (default: {','.join(METADATA_COLUMNS)})",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vuln-dbn")
    commands = parser.add_subparsers(dest="command", required=True)

    learn = commands.add_parser("learn", help="Learn a temporally-constrained (optionally sink-constrained) DBN")
    learn.add_argument("--dataset", required=True, help="wide_two_slice_table.csv from vuln-commit-history")
    learn.add_argument("--model", required=True)
    learn.add_argument("--summary", required=True)
    learn.add_argument("--sink", help="Optional node to also constrain as a pure sink, e.g. TRANSITION_LABEL__t1")
    _add_metadata_arg(learn)
    learn.add_argument("--score", choices=["bic-d", "bdeu"], default="bic-d")
    learn.add_argument("--max-indegree", type=int, default=4)
    learn.add_argument("--equivalent-sample-size", type=float, default=5.0)
    learn.add_argument("--epsilon", type=float, default=1e-4)
    learn.add_argument("--max-iterations", type=int, default=1_000_000)
    learn.add_argument("--checkpoint", help="Path for the resumable structure-learning checkpoint")
    learn.add_argument("--resume", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Grouped holdout evaluation against a sink")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--sink", required=True)
    evaluate.add_argument("--positive-state", help="Sink state treated as positive (default: only valid for a binary sink)")
    _add_metadata_arg(evaluate)
    evaluate.add_argument("--group", default="project", help="Column to group-split on, e.g. project or anchor_id")
    evaluate.add_argument("--test-size", type=float, default=0.2)
    evaluate.add_argument("--random-state", type=int, default=42)
    evaluate.add_argument("--score", choices=["bic-d", "bdeu"], default="bic-d")
    evaluate.add_argument("--max-indegree", type=int, default=4)
    evaluate.add_argument("--equivalent-sample-size", type=float, default=5.0)

    predict = commands.add_parser("predict", help="Infer sink-state probabilities for present nodes")
    predict.add_argument("--model", required=True)
    predict.add_argument("--evidence", required=True)
    predict.add_argument("--output", required=True)

    significance = commands.add_parser("mi", help="Anchor-aware mutual-information significance analysis")
    significance.add_argument("--dataset", required=True)
    significance.add_argument("--output", required=True)
    significance.add_argument("--sink", required=True)
    _add_metadata_arg(significance)
    significance.add_argument("--permutations", type=int, default=1000)
    significance.add_argument("--random-state", type=int, default=42)

    bootstrap = commands.add_parser("bootstrap", help="Anchor-preserving edge-stability bootstrap")
    bootstrap.add_argument("--dataset", required=True)
    bootstrap.add_argument("--output", required=True)
    bootstrap.add_argument("--sink", help="Optional sink constraint, same semantics as `learn`")
    _add_metadata_arg(bootstrap)
    bootstrap.add_argument("--repetitions", type=int, default=100)
    bootstrap.add_argument("--random-state", type=int, default=42)
    bootstrap.add_argument("--score", choices=["bic-d", "bdeu"], default="bic-d")
    bootstrap.add_argument("--max-indegree", type=int, default=4)
    bootstrap.add_argument("--equivalent-sample-size", type=float, default=5.0)

    dot = commands.add_parser("dot", help="Export a learned structure summary as Graphviz DOT")
    dot.add_argument("--summary", required=True)
    dot.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "learn":
        message = train_from_csv(
            args.dataset,
            args.model,
            args.summary,
            _metadata_columns(args.metadata_columns),
            sink=args.sink,
            score=args.score,
            max_indegree=args.max_indegree,
            equivalent_sample_size=args.equivalent_sample_size,
            epsilon=args.epsilon,
            max_iter=args.max_iterations,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
        )
    elif args.command == "evaluate":
        message = evaluate_holdout(
            args.dataset,
            args.output,
            _metadata_columns(args.metadata_columns),
            args.sink,
            group_column=args.group,
            positive_state=args.positive_state,
            test_size=args.test_size,
            random_state=args.random_state,
            score=args.score,
            max_indegree=args.max_indegree,
            equivalent_sample_size=args.equivalent_sample_size,
        )
    elif args.command == "predict":
        message = write_prediction(args.model, args.evidence, args.output)
    elif args.command == "mi":
        report = mutual_information_report(
            args.dataset,
            args.output,
            _metadata_columns(args.metadata_columns),
            args.sink,
            permutations=args.permutations,
            random_state=args.random_state,
        )
        message = {"nodes": len(report), "output": args.output}
    elif args.command == "bootstrap":
        message = bootstrap_edges(
            args.dataset,
            args.output,
            _metadata_columns(args.metadata_columns),
            sink=args.sink,
            repetitions=args.repetitions,
            random_state=args.random_state,
            score=args.score,
            max_indegree=args.max_indegree,
            equivalent_sample_size=args.equivalent_sample_size,
        )
    elif args.command == "dot":
        write_dot(args.summary, args.output)
        message = {"output": args.output}
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(message, indent=2, default=str))


if __name__ == "__main__":
    main()
