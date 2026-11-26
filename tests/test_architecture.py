"""MVC dependency rules: the shared package and the models layer never import streamlit."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHARED_PACKAGE = ROOT / "src" / "employee_attrition"
DASHBOARD_MODELS = ROOT / "dashboard" / "models"


def _imported_modules(path: Path) -> set[str]:
    """Return the full module names imported by a Python file."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize("path", sorted(DASHBOARD_MODELS.rglob("*.py")), ids=lambda p: p.name)
def test_models_do_not_import_streamlit_or_views(path: Path) -> None:
    for module in _imported_modules(path):
        assert module.split(".")[0] != "streamlit", (path, module)
        assert not module.startswith("dashboard.views"), (path, module)
        assert not module.startswith("dashboard.controllers"), (path, module)


@pytest.mark.parametrize("path", sorted(SHARED_PACKAGE.rglob("*.py")), ids=lambda p: p.name)
def test_shared_package_is_app_independent(path: Path) -> None:
    for module in _imported_modules(path):
        assert module.split(".")[0] not in {"streamlit", "dashboard"}, (path, module)
