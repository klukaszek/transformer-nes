"""Cache and data files written whole or not at all."""

from __future__ import annotations

import os
import tempfile


def write_atomic(path: str, data: bytes) -> None:
    """Write `path` so no reader sees it half written, and writers racing
    for it (the game and a learner folding the same window) each replace
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
