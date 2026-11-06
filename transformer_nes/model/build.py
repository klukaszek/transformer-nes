"""Build and cache generated native extensions, independently of a machine.

Callers provide their construction identity. This module owns artifact
manifests, build locks, and extension loading, not graph semantics.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager

from transformer_nes.files import write_atomic

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_engine_cache")
MANIFEST_PATH = os.path.join(CACHE_DIR, "manifest.json")
# Limit concurrent compiler frontends independently of CPU source sharding.
DEFAULT_COMPILE_JOBS = 2


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


def manifest_read() -> dict:
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


@contextmanager
def _cache_lock(name: str, *, blocking: bool = True):
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


def artifact_lock(digest: str):
    """Own a digest from source creation through module loading/publication.

    Cache-hit callers must check files again after acquiring this lock: a
    manifest snapshot alone cannot keep its artifacts alive. Loaded native
    modules need no lease; their code remains mapped if disk files are pruned.
    """
    return _cache_lock(digest)


def manifest_put(key: str, entry: dict, *, code: str | None = None) -> None:
    """Record an artifact, replacing obsolete builds of the same kind.

    A machine compiler supplies its source digest as ``code``. Graphs use
    their own construction identities and are replaced by their tag.
    """
    with _cache_lock("manifest"):
        manifest = manifest_read()
        manifest[key] = {**entry, "code": code}
        kind = entry.get("kind")
        manifest = {
            k: e for k, e in manifest.items() if e.get("kind") != kind or e.get("code") == code
        }
        write_atomic(MANIFEST_PATH, json.dumps(manifest, indent=1, sort_keys=True).encode())
        _prune_cache({e["digest"] for e in manifest.values()})


ARTIFACT_RE = re.compile(r"^(?:engine|graph|build)_([0-9a-f]{16})(?:[_.].*)?$")


def _prune_cache(live_digests: set[str]) -> None:
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
        with _cache_lock(digest, blocking=False) as acquired:
            if not acquired:
                continue
            for path in paths:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.unlink(path)


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


def exit_with_parent(interval: float = 0.5) -> None:
    """Exit this process, at once and quietly, when its parent does."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(interval)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
