"""What a cached artifact was made from, as a digest.

A build depends on the modules that make it and every module of this
package they import at module scope, transitively. Conditional and relative
imports and package initializers count; imports inside functions are lazy
and optional. Native source files can be included explicitly with ``files``.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import os
from importlib.util import resolve_name

PACKAGE = "transformer_nes"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _path(module: str) -> str | None:
    base = os.path.join(ROOT, *module.split("."))
    for path in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(path):
            return path
    return None


def _module_imports(nodes):
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from _module_imports(ast.iter_child_nodes(node))


def _imports(path: str) -> set[str]:
    """Package imports executed at module scope, without importing code."""
    found = set()
    relative = os.path.relpath(path, ROOT)
    module = relative.removesuffix(".py").replace(os.sep, ".")
    package = module.rpartition(".")[0]
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in _module_imports(tree.body):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        else:
            origin = node.module or ""
            if node.level:
                origin = resolve_name("." * node.level + origin, package)
            found.add(origin)
            found.update(
                submodule for alias in node.names
                if _path(submodule := f"{origin}.{alias.name}") is not None
            )
    return {name for name in found if name == PACKAGE or name.startswith(PACKAGE + ".")}


@functools.cache
def source_digest(*modules: str, files: tuple[str, ...] = ()) -> str:
    """Hash construction modules, their imports, and explicit native files."""
    seen: dict[str, str] = {}
    pending = list(modules)
    while pending:
        module = pending.pop()
        path = _path(module)
        if module in seen or path is None:
            continue
        seen[module] = path
        pending.extend(_imports(path))
        if "." in module:
            pending.append(module.rpartition(".")[0])
    digest = hashlib.sha256()
    for module in sorted(seen):
        digest.update(module.encode())
        with open(seen[module], "rb") as f:
            digest.update(f.read())
    for path in sorted(set(files)):
        digest.update(os.path.relpath(path, ROOT).encode())
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()
