"""The dependency arrow between `eval` and the runtime packages only goes one way.

`eval` is allowed to import `graph`, `rag`, `mcp_server` to build fixtures and
call assertions against real state shapes. The reverse must never happen: if
runtime code started importing the eval harness, a production deploy could
crash from a missing `ragas` install (an optional dependency, see
`pyproject.toml`).
"""

from __future__ import annotations

import ast
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGES = ["graph", "rag", "mcp_server"]


def _imports_eval(py_file: Path) -> bool:
    """Whether this file has a top-level import naming the `eval` package."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "eval" or alias.name.startswith("eval.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "eval" or node.module.startswith("eval.")):
                return True
    return False


def test_no_runtime_package_imports_eval() -> None:
    offenders = []
    for package in RUNTIME_PACKAGES:
        for py_file in (AGENT_ROOT / package).rglob("*.py"):
            if _imports_eval(py_file):
                offenders.append(str(py_file.relative_to(AGENT_ROOT)))

    assert not offenders, (
        f"runtime packages must never import `eval`, found imports in: {offenders}"
    )
