"""CPU runtime source and sharding for the generated scheduling shell."""

import os
import re
from pathlib import Path

from transformer_nes.files import write_atomic
from transformer_nes.model.build import positive_env_int

ROOT = Path(__file__).parent
ENGINE_FILES = tuple(str(path) for path in sorted((ROOT / "engine.cpp", ROOT / "machine_abi.h")))


def source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


# Bound translation-unit size separately from the generic build job limit.
DEFAULT_COMPILE_SHARDS = 12


def compile_shards() -> int:
    return positive_env_int("TRANSFORMER_NES_COMPILE_SHARDS", DEFAULT_COMPILE_SHARDS)


FUNC_RE = re.compile(r"^    (void|bool|int|long long) (step_core\w*)\((.*)\) \{$")


def shard_source(source: str, struct_name: str, n_shards: int) -> tuple[str, list[str]]:
    """Split a single-TU engine into a header + n_shards bodies so ninja
    compiles them in parallel.  The generated step_core* methods become out-of-class definitions, greedily balanced by line
    count; everything else stays in-class (implicitly inline) in the
    header.  The PYBIND module block lands in shard 0.  Purely a deterministic post-processing of the emitted
    text -- the digest covers the original."""
    lines = source.split("\n")
    header: list[str] = ["#pragma once"]
    chunks: dict[str, list[str]] = {}
    pybind: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = FUNC_RE.match(line)
        if m is not None:
            returns, name, args = m.groups()
            depth = line.count("{") - line.count("}")
            chunk = [f"{returns} {struct_name}::{name}({args}) {{"]
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                chunk.append(lines[i])
                i += 1
            header.append(f"    {returns} {name}({args});")
            chunks.setdefault(name, []).extend([*chunk, ""])
            continue
        if line.startswith("PYBIND11_MODULE("):
            depth = line.count("{") - line.count("}")
            pybind.append(line)
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                pybind.append(lines[i])
                i += 1
            continue
        header.append(line)
        i += 1

    shards: list[list[str]] = [[] for _ in range(max(1, n_shards))]
    sizes = [0] * len(shards)
    for chunk in sorted(chunks.values(), key=len, reverse=True):
        k = sizes.index(min(sizes))
        shards[k].extend(chunk)
        sizes[k] += len(chunk)
    shards[0] = pybind + [""] + shards[0]
    return "\n".join(header) + "\n", ["\n".join(s) + "\n" for s in shards]


def write_shards(cache_dir: str, stem: str, struct_name: str, source: str, n_shards: int) -> list[str]:
    """`source` as cache_dir/<stem>.hpp and <stem>_s<k>.cpp (shard_source),
    unless already written; the shards' paths."""
    hpp_path = os.path.join(cache_dir, f"{stem}.hpp")
    shard_paths = [os.path.join(cache_dir, f"{stem}_s{k}.cpp") for k in range(n_shards)]
    existing = [
        os.path.join(cache_dir, path)
        for path in os.listdir(cache_dir)
        if path.startswith(f"{stem}_s") and path.endswith(".cpp")
    ]
    if not os.path.exists(hpp_path) or set(existing) != set(shard_paths):
        header, shards = shard_source(source, struct_name, n_shards)
        for path, body in zip(shard_paths, shards, strict=True):
            write_atomic(path, f'#include "{stem}.hpp"\n\n{body}'.encode())
        write_atomic(hpp_path, header.encode())
        for path in set(existing) - set(shard_paths):
            os.unlink(path)
    return shard_paths


