"""CI installs the project as a package, so a module missing from ``py-modules``
imports fine on a dev machine (cwd) and then breaks in CI/Docker."""

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_imported_root_module_is_packaged():
    listed = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]["py-modules"]
    )
    root_modules = {p.stem for p in ROOT.glob("*.py")}
    # Only modules the packaged code actually imports can break an installed copy.
    needed: set[str] = set()
    for name in listed:
        tree = ast.parse((ROOT / f"{name}.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                needed |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                needed.add(node.module.split(".")[0])
    missing = sorted((needed & root_modules) - listed)
    assert not missing, f"add to [tool.setuptools] py-modules in pyproject.toml: {missing}"
