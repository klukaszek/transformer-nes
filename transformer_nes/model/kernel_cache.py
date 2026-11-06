"""Reuse scalar circuits by their actual arithmetic and compiler ABI.

Construction-source changes invalidate a bank, but do not invalidate identical
circuits inside it. Cached bodies use model index zero; assembly relocates only
the circuit/phase function names. No arithmetic or native execution changes.
"""

from __future__ import annotations

import functools
import hashlib
import json
import marshal
import os
import pickle
import platform
import sys
import types
from pathlib import Path

import torch

from transformer_nes.files import write_atomic
from transformer_nes.model import build

MAX_BYTES = 256 * 1024**2
SCHEMA = 1


@functools.cache
def _compiler_identity(extractor):
    def normalized(code):
        return code.replace(
            co_filename="", co_firstlineno=1, co_linetable=b"",
            co_consts=tuple(normalized(c) if isinstance(c, types.CodeType) else c for c in code.co_consts),
        )

    return (
        SCHEMA, torch.__version__, torch.version.git_version,
        platform.system(), platform.machine(), sys.version_info[:3],
        sys.byteorder, marshal.dumps(normalized(extractor.__code__)),
        extractor.__defaults__, extractor.__kwdefaults__,
    )


def fingerprint(graph_key, config, *, memory_size, scalar, extractor):
    # Pickle is used only as deterministic hash input; cache files are JSON.
    # FX targets here are Torch/operator globals, not generated closures.
    identity = (_compiler_identity(extractor), graph_key, tuple(sorted(config.items())),
                memory_size, scalar)
    return hashlib.sha256(pickle.dumps(identity, protocol=4)).hexdigest()


def _read(path, digest):
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


def get_kernel(digest, lower):
    """Return (body, headers, metadata, hit), compiling once across processes."""
    directory = Path(build.CACHE_DIR) / "kernels"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    with build.artifact_lock(f"kernel-{digest}"):
        result = _read(path, digest)
        if result is not None:
            return *result, True
        body, headers, meta = lower()
        checksum = hashlib.sha256(json.dumps([body, headers, meta], sort_keys=True).encode()).hexdigest()
        entry = dict(key=digest, checksum=checksum, body=body, headers=headers, meta=meta)
        write_atomic(path, json.dumps(entry).encode())
    _prune(directory)
    return body, headers, meta, False


def _prune(directory):
    # Keep source reuse bounded independently of bank/native-artifact pruning.
    # Never wait on a kernel while holding the eviction lock.
    with build.artifact_lock("kernel-eviction"):
        files = []
        for path in directory.glob("*.json"):
            try:
                stat = path.stat()
                files.append((stat.st_mtime_ns, stat.st_size, path))
            except FileNotFoundError:
                pass
        size = sum(n for _, n, _ in files)
        for _, n, path in sorted(files):
            if size <= MAX_BYTES:
                break
            with build._cache_lock(f"kernel-{path.stem}", blocking=False) as acquired:
                if acquired:
                    try:
                        path.unlink()
                        size -= n
                    except FileNotFoundError:
                        pass
