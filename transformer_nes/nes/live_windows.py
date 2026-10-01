"""Learning a ROM's windows while it plays.

The engine profiles the code blocks it enters by single steps.  Every so
often the console hands that profile and a snapshot of memory to a learner
process, which folds the hottest blocks into windows (runner.learn_windows),
merges them into the ROM's saved windows and builds their module; each
fold is cached as it is made, so a learner stopped by quitting keeps its
work.  The console then attaches the windows between frames, from the
cache.  Windows change which forward retires an instruction, never what it
computes, so they can be attached mid-game.  Saved windows whose module
is not built yet are built the same way, so a game never waits on a fold.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys

import numpy as np

logger = logging.getLogger(__name__)


class WindowLearner:
    """Runs learner processes for `runtime` (a Cpu6502TransformerMachine),
    saving to `path`, one at a time: a couple of seconds after boot when the
    ROM has no windows yet, then every `every` frames.  A learner ends
    with the game."""

    def __init__(self, runtime, path: str, *, every: int = 1800):
        self.runtime, self.path, self.every = runtime, path, every
        self.first = every if os.path.exists(path) else 120
        self.process: subprocess.Popen | None = None

    def prepare(self) -> None:
        """Fold the saved windows and build their module, learning nothing."""
        self._run({"path": self.path, "forward_hook": self.runtime.forward_hook})
        logger.info("building the saved windows in the background")

    def tick(self, frame: int) -> bool:
        """Once per frame: whether new windows are ready to attach."""
        ready = self._finished()
        if frame == self.first or (frame > self.first and frame % self.every == 0):
            self._start()
        return ready

    def _start(self) -> None:
        """Learn from the profile since the last start (taking it)."""
        if self.process is not None:
            return  # one at a time
        profile = np.asarray(self.runtime.engine.engine.take_profile())
        if not profile[0].any():
            return  # everything ran through windows
        self._run({
            "profile": profile, "memory": self.runtime.dump_ram(), "path": self.path,
            "forward_hook": self.runtime.forward_hook,
        })
        logger.info("learning windows in the background")

    def _run(self, spec: dict) -> None:
        self.process = subprocess.Popen([sys.executable, "-m", __name__], stdin=subprocess.PIPE)
        pickle.dump(spec, self.process.stdin)
        self.process.stdin.close()

    def _finished(self) -> bool:
        if self.process is None or self.process.poll() is None:
            return False
        ok, self.process = self.process.returncode == 0, None
        if not ok:
            logger.warning("window learner failed")
        return ok


def learn(spec: dict) -> None:
    from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine
    from transformer_nes.nes.app import load_windows, save_windows

    machine = Cpu6502TransformerMachine(
        bcd=False, mirror_cpu_ram=True, forward_hook=spec["forward_hook"])
    if "profile" in spec:
        windows = machine.learn_windows(profile=spec["profile"], memory=spec["memory"])
        save_windows(spec["path"], windows)
    machine.attach_sequence_module(load_windows(spec["path"]))  # folds, builds the module


if __name__ == "__main__":
    from transformer_nes.children import exit_with_parent

    exit_with_parent()
    logging.basicConfig(level=logging.INFO, format="learner: %(message)s")
    learn(pickle.load(sys.stdin.buffer))
