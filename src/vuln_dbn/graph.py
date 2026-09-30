from __future__ import annotations

from pathlib import Path

from .constants import T0_SUFFIX
from .io import read_json


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _ancestors_within_hops(edges: list[list[str]], root: str, hops: int) -> set[str]:
    """Nodes reachable by walking edges backward (parent, parent-of-parent, ...) from root."""
    parents_of: dict[str, list[str]] = {}
    for source, target in edges:
        parents_of.setdefault(target, []).append(source)
    visited = {root}
    frontier = {root}
    for _ in range(hops):
        frontier = {p for n in frontier for p in parents_of.get(n, []) if p not in visited}
        if not frontier:
            break
        visited |= frontier
    return visited


def write_dot(
    summary_json: str | Path, output_dot: str | Path, focus_sink_hops: int | None = None
) -> None:
    summary = read_json(summary_json)
    sink = summary.get("sink")
    nodes, edges = summary["nodes"], summary["edges"]

    if focus_sink_hops is not None:
        if not sink:
            raise ValueError("--focus-sink-hops needs a summary learned with a sink")
        keep = _ancestors_within_hops(edges, sink, focus_sink_hops)
        nodes = [n for n in nodes if n in keep]
        edges = [e for e in edges if e[0] in keep and e[1] in keep]

    lines = ["digraph DynamicBayesianNetwork {", "  rankdir=LR;", "  node [shape=box];"]
    for node in nodes:
        if node == sink:
            lines.append(f"  {_quote(node)} [shape=doublecircle, style=filled, fillcolor=lightgoldenrod1];")
        elif node.endswith(T0_SUFFIX):
            lines.append(f"  {_quote(node)} [style=filled, fillcolor=lightblue1];")
        else:
            lines.append(f"  {_quote(node)} [style=filled, fillcolor=honeydew2];")
    for source, target in edges:
        lines.append(f"  {_quote(source)} -> {_quote(target)};")
    lines.append("}")
    destination = Path(output_dot)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
