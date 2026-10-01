"""Helper processes (model compilers, window learners) end with the
process that started them: whatever they were doing is only for it."""

from __future__ import annotations

import os
import threading
import time


def exit_with_parent(interval: float = 0.5) -> None:
    """Exit this process, at once and quietly, when its parent does."""
    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(interval)
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
