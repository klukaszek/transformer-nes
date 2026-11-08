"""Build infrastructure shared by the CPU compiler and NES frontend.

Paths and source identities come first, followed by artifact ownership and
native loading, then reuse of individual scalar kernels. Cached artifacts
are published atomically under process locks; no build work runs during play.
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
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
from contextlib import contextmanager
from importlib.util import resolve_name
from pathlib import Path

logger = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).parent
PROJECT_DIR = PACKAGE_DIR.parent
# Generated kernels and native banks live beside the project, never in the package.
CACHE_DIR = str(PROJECT_DIR / ".cache" / "engine")
PACKAGE = "transformer_nes"
ARTIFACT_RE = re.compile(r"^(?:engine|graph|build)_([0-9a-f]{16})(?:[_.].*)?$")
MAX_KERNEL_BYTES = 256 * 1024**2
KERNEL_CACHE_SCHEMA = 1
# Limit concurrent compiler frontends independently of CPU source sharding.
DEFAULT_COMPILE_JOBS = 2


# Source files and build settings


def positive_env_int(name: str, default: int) -> int:
    """Read a positive integer build setting without leaking stringly typed
    environment handling through the engine/cache paths."""
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


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


def manifest_read() -> dict:
    try:
        with (Path(CACHE_DIR) / "manifest.json").open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def manifest_put(key: str, entry: dict, *, code: str | None = None) -> None:
    """Record an artifact, replacing obsolete builds of the same kind.

    A machine compiler supplies its source digest as ``code``. Graphs use
    their own construction identities and are replaced by their tag.
    """
    with artifact_lock("manifest"):
        manifest = manifest_read()
        manifest[key] = {**entry, "code": code}
        kind = entry.get("kind")
        manifest = {
            k: e for k, e in manifest.items() if e.get("kind") != kind or e.get("code") == code
        }
        write_atomic(Path(CACHE_DIR) / "manifest.json", json.dumps(manifest, indent=1, sort_keys=True).encode())
        prune_artifacts({e["digest"] for e in manifest.values()})


def prune_artifacts(live_digests: set[str]) -> None:
    """Under the manifest lock, remove obsolete, inactive artifact groups."""
    obsolete: dict[str, list[str]] = {}
    for name in os.listdir(CACHE_DIR):
        m = ARTIFACT_RE.match(name)
        if m is None or m.group(1) in live_digests:
            continue
        obsolete.setdefault(m.group(1), []).append(os.path.join(CACHE_DIR, name))
    for digest, paths in obsolete.items():
        # Writers acquire artifact -> manifest. Never wait in the opposite
        # order: an active writer may be about to publish its completed build.
        with artifact_lock(digest, blocking=False) as acquired:
            if not acquired:
                continue
            for path in paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)


# Native extension compilation and loading


def load_module(
    name: str,
    digest: str,
    sources: list[str],
    *,
    jobs: int | None = None,
    extra_cflags: tuple[str, ...] = (),
):
    """Load a cached extension or compile it from ``sources``, `jobs`
    compilers at once (default: the bound for core shards).

    The caller holds ``artifact_lock(digest)`` through source writing, this
    load, and manifest publication.
    """
    if name in sys.modules:
        return sys.modules[name]
    build_dir = os.path.join(CACHE_DIR, f"build_{digest}")
    os.makedirs(build_dir, exist_ok=True)
    so_path = os.path.join(build_dir, f"{name}.so")
    if os.path.exists(so_path):
        try:
            spec = importlib.util.spec_from_file_location(name, so_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
            return module
        except ImportError as exc:
            # A cached extension can outlive the process that provided its
            # dynamic-library search path (notably torch's libc10 on macOS).
            logger.info("rebuilding unavailable cached engine %s: %s", name, exc)
            os.unlink(so_path)

    import torch
    from torch.utils.cpp_extension import load

    # Torch's file baton survives a killed compiler process. Our OS lock
    # proves no other project builder owns this directory, so reclaim it.
    baton = os.path.join(build_dir, "lock")
    if os.path.exists(baton):
        os.unlink(baton)
    build_start = time.perf_counter()
    cflags = ["-O2", "-std=c++17", "-fwrapv", "-ffp-contract=off", *extra_cflags]
    old_max_jobs = os.environ.get("MAX_JOBS")
    if old_max_jobs is None:
        os.environ["MAX_JOBS"] = str(
            jobs or positive_env_int("TRANSFORMER_NES_COMPILE_JOBS", DEFAULT_COMPILE_JOBS)
        )
    try:
        module = load(
            name=name,
            sources=sources,
            extra_cflags=cflags,
            extra_ldflags=[f"-Wl,-rpath,{os.path.join(os.path.dirname(torch.__file__), 'lib')}"],
            extra_include_paths=[CACHE_DIR],
            build_directory=build_dir,
            verbose=False,
        )
    finally:
        if old_max_jobs is None:
            del os.environ["MAX_JOBS"]
    logger.info("compiled %s in %.1fs", name, time.perf_counter() - build_start)
    return module


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


def exit_with_parent(interval: float = 0.5) -> None:
    """Exit this process, at once and quietly, when its parent does."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(interval)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


# Reuse of individual scalar kernels


@functools.cache
def compiler_identity(extractor):
    import torch

    def normalized(code):
        return code.replace(
            co_filename="", co_firstlineno=1, co_linetable=b"",
            co_consts=tuple(normalized(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
        )

    return (
        KERNEL_CACHE_SCHEMA, torch.__version__, torch.version.git_version,
        platform.system(), platform.machine(), sys.version_info[:3],
        sys.byteorder, marshal.dumps(normalized(extractor.__code__)),
        extractor.__defaults__, extractor.__kwdefaults__,
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
