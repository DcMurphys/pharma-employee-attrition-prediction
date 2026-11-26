"""Every third-party library the app imports is pinned in dashboard/requirements.txt.

Streamlit Community Cloud installs only that file, so an unpinned import could arrive in a
different version than the one tested (scope §8).
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "dashboard" / "requirements.txt"
DEV = ROOT / "requirements-dev.txt"
FIRST_PARTY = {"dashboard", "employee_attrition"}
# Import name -> distribution name, where the two differ.
DISTRIBUTIONS = {"sklearn": "scikit-learn"}


def pins(path: Path) -> dict[str, str]:
    """Exact pins ``name==version`` in a requirements file (comments and options skipped)."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def third_party_imports(*folders: Path) -> set[str]:
    """Top-level third-party modules imported by the Python files in ``folders``."""
    modules = set()
    for folder in folders:
        for path in folder.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    modules |= {alias.name.split(".")[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules.add(node.module.split(".")[0])
    return {m for m in modules if m not in sys.stdlib_module_names and m not in FIRST_PARTY}


def test_app_imports_are_pinned() -> None:
    imported = third_party_imports(ROOT / "dashboard", ROOT / "src")
    missing = {DISTRIBUTIONS.get(m, m) for m in imported} - set(pins(RUNTIME))
    assert not missing, f"Pin these in dashboard/requirements.txt: {sorted(missing)}"


def test_dev_requirements_are_all_pinned() -> None:
    for line in DEV.read_text(encoding="utf-8").splitlines():
        entry = line.split("#")[0].strip()
        if entry and not entry.startswith("-"):
            assert "==" in entry, f"Unpinned in requirements-dev.txt: {entry}"
