"""Build infrastructure: paths, source identities, the native console
extensions, and reuse of individual scalar kernels across builds (published
atomically under process locks).
"""

from __future__ import annotations

import ast
import fcntl
import functools
import hashlib
import importlib.util
import json
import logging
import marshal
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import types
from contextlib import contextmanager
from importlib.util import resolve_name
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
PROJECT_DIR = PACKAGE_DIR.parent
FIXTURES_DIR = PROJECT_DIR / "fixtures"
# The scalar compiler's reusable kernels (and their locks), beside the project.
CACHE_DIR = str(PROJECT_DIR / ".cache" / "compiler")
PACKAGE = "transformer_nes"
MAX_KERNEL_BYTES = 256 * 1024**2
KERNEL_CACHE_SCHEMA = 1
# Limit concurrent compiler frontends independently of CPU source sharding.


# Source files and build settings


def module_file(module: str) -> str | None:
    base = os.path.join(PROJECT_DIR, *module.split("."))
    for path in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.exists(path):
            return path
    return None


def scope_imports(nodes):
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from scope_imports(ast.iter_child_nodes(node))


def module_imports(path: str) -> set[str]:
    """Package imports executed at module scope, without importing code."""
    found = set()
    relative = os.path.relpath(path, PROJECT_DIR)
    module = relative.removesuffix(".py").replace(os.sep, ".")
    package = module.rpartition(".")[0]
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    for node in scope_imports(tree.body):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        else:
            origin = node.module or ""
            if node.level:
                origin = resolve_name("." * node.level + origin, package)
            found.add(origin)
            found.update(
                submodule for alias in node.names
                if module_file(submodule := f"{origin}.{alias.name}") is not None
            )
    return {name for name in found if name == PACKAGE or name.startswith(PACKAGE + ".")}


@functools.cache
def source_digest(*modules: str, files: tuple[str, ...] = ()) -> str:
    """Hash construction modules, their imports, and explicit native files."""
    seen: dict[str, str] = {}
    pending = list(modules)
    while pending:
        module = pending.pop()
        path = module_file(module)
        if module in seen or path is None:
            continue
        seen[module] = path
        pending.extend(module_imports(path))
        if "." in module:
            pending.append(module.rpartition(".")[0])
    digest = hashlib.sha256()
    for module in sorted(seen):
        digest.update(module.encode())
        with open(seen[module], "rb") as f:
            digest.update(f.read())
    for path in sorted(set(files)):
        digest.update(os.path.relpath(path, PROJECT_DIR).encode())
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


# Atomic artifact ownership and publication


def write_atomic(path: str, data: bytes) -> None:
    """Write `path` so no reader sees it half written, and writers racing
    for it each replace
    it whole: through a temporary file of the writer's own, then a rename."""
    directory = os.path.dirname(path) or "."
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


@contextmanager
def artifact_lock(name: str, *, blocking: bool = True):
    # Stable, separate inodes: unlinking a lock after use lets a waiter and a
    # new caller lock different files. The OS releases flock on process exit.
    directory = os.path.join(CACHE_DIR, ".locks")
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, name), "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


# Native extensions


def make(target: str) -> None:
    """Build a repository Makefile target using this Python environment."""
    subprocess.run(
        ["make", "-s", "-C", str(PROJECT_DIR), target, f"PYTHON={sys.executable}"], check=True
    )


def load_nes_extension(target: str):
    """Build and load hardware, frontend, or the test-only reference hardware."""
    name = {
        "hardware": "nes_hardware",
        "frontend": "nes_app",
        "reference-hardware": "nes_hardware_reference",
    }[target]
    make(target)
    path = PROJECT_DIR / "bin" / f"{name}.so"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse of individual scalar kernels


@functools.cache
def compiler_identity(extractor):
    import torch

    def normalized(code):
        return code.replace(
            co_filename="", co_firstlineno=1, co_linetable=b"",
            co_consts=tuple(normalized(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
        )

    # The extractor's own code plus everything its module defines and
    # imports: a hit must reproduce exactly what a cold lowering would emit.
    return (
        KERNEL_CACHE_SCHEMA, torch.__version__, torch.version.git_version,
        platform.system(), platform.machine(), sys.version_info[:3],
        sys.byteorder, marshal.dumps(normalized(extractor.__code__)),
        extractor.__defaults__, extractor.__kwdefaults__,
        source_digest(extractor.__module__),
    )


def kernel_fingerprint(graph_key, config, *, memory_size, scalar, extractor):
    # Pickle is used only as deterministic hash input; cache files are JSON.
    # FX targets here are Torch/operator globals, not generated closures.
    identity = (compiler_identity(extractor), graph_key, tuple(sorted(config.items())),
                memory_size, scalar)
    return hashlib.sha256(pickle.dumps(identity, protocol=4)).hexdigest()


def read_kernel(path, digest):
    try:
        entry = json.loads(path.read_bytes())
        body, headers, meta = entry["body"], entry["headers"], entry["meta"]
        checksum = hashlib.sha256(json.dumps([body, headers, meta], sort_keys=True).encode()).hexdigest()
        if entry["key"] != digest or entry["checksum"] != checksum:
            return None
        if not isinstance(body, str) or not isinstance(headers, list) or not all(isinstance(h, str) for h in headers):
            return None
        if not all(type(meta[n]) is int and meta[n] >= 0 for n in ("width", "reads", "arguments")):
            return None
        if type(meta["checks_division"]) is not bool:
            return None
        os.utime(path, None)
        return body, headers, meta
    except (OSError, ValueError, KeyError, TypeError):
        return None


def cached_kernel(digest, lower):
    """Return (body, headers, metadata, hit), compiling once across processes."""
    directory = Path(CACHE_DIR) / "kernels"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    with artifact_lock(f"kernel-{digest}"):
        result = read_kernel(path, digest)
        if result is not None:
            return *result, True
        body, headers, meta = lower()
        checksum = hashlib.sha256(json.dumps([body, headers, meta], sort_keys=True).encode()).hexdigest()
        entry = dict(key=digest, checksum=checksum, body=body, headers=headers, meta=meta)
        write_atomic(path, json.dumps(entry).encode())
    prune_kernels(directory)
    return body, headers, meta, False


def prune_kernels(directory):
    # Keep source reuse bounded independently of bank/native-artifact pruning.
    # Never wait on a kernel while holding the eviction lock.
    with artifact_lock("kernel-eviction"):
        files = []
        for path in directory.glob("*.json"):
            try:
                stat = path.stat()
                files.append((stat.st_mtime_ns, stat.st_size, path))
            except FileNotFoundError:
                pass
        size = sum(n for _, n, _ in files)
        for _, n, path in sorted(files):
            if size <= MAX_KERNEL_BYTES:
                break
            with artifact_lock(f"kernel-{path.stem}", blocking=False) as acquired:
                if acquired:
                    try:
                        path.unlink()
                        size -= n
                    except FileNotFoundError:
                        pass
