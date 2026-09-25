from __future__ import annotations

from pathlib import Path

from .constants import T0_SUFFIX
from .io import read_json


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_dot(summary_json: str | Path, output_dot: str | Path) -> None:
    summary = read_json(summary_json)
    sink = summary.get("sink")
    lines = ["digraph DynamicBayesianNetwork {", "  rankdir=LR;", "  node [shape=box];"]
    for node in summary["nodes"]:
        if node == sink:
            lines.append(f"  {_quote(node)} [shape=doublecircle, style=filled, fillcolor=lightgoldenrod1];")
        elif node.endswith(T0_SUFFIX):
            lines.append(f"  {_quote(node)} [style=filled, fillcolor=lightblue1];")
        else:
            lines.append(f"  {_quote(node)} [style=filled, fillcolor=honeydew2];")
    for source, target in summary["edges"]:
        lines.append(f"  {_quote(source)} -> {_quote(target)};")
    lines.append("}")
    destination = Path(output_dot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
