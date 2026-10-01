"""What a cached artifact was made from, as a digest.

A build depends on the modules that make it and every module of this
package they import at top level, transitively (imports inside functions
are lazy and optional).  Keying each cache by just that code means an edit
elsewhere, to the app or the PPU say, keeps it.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import os

PACKAGE = "transformer_nes"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _path(module: str) -> str | None:
    base = os.path.join(ROOT, *module.split("."))
    for path in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(path):
            return path
    return None


def _imports(path: str) -> set[str]:
    """The package's modules imported at the top of the file at `path`."""
    found = set()
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.startswith(PACKAGE)}
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PACKAGE):
            for alias in node.names:  # `from package import module` names a module
                submodule = f"{node.module}.{alias.name}"
                found.add(submodule if _path(submodule) else node.module)
    return found


@functools.cache
def source_digest(*modules: str) -> str:
    """A digest of `modules`' sources and their imports' (see above)."""
    seen: dict[str, str] = {}
    pending = list(modules)
    while pending:
        module = pending.pop()
        path = _path(module)
        if module in seen or path is None:
            continue
        seen[module] = path
        pending.extend(_imports(path))
    digest = hashlib.sha256()
    for module in sorted(seen):
        digest.update(module.encode())
        with open(seen[module], "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()
