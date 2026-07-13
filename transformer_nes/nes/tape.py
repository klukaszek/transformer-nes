"""Replay recorded input tapes headless: benchmark, profile, compare.

A tape (app --record) is a step function of controller state: lines of
"<frame> <mask hex> <names>".  This tool replays one against a backend
and reports per-segment fps and instructions/frame (menu phases and
gameplay phases separate cleanly), the fallback-pc profile over the
whole run, optional frame snapshots, and -- as a debugging mode -- a
frame-by-frame lockstep comparison against a second backend under the
same input.

    python -m transformer_nes.nes.tape ROM TAPE [--backend weights]
        [--segment 600] [--tail 120] [--profile MIN] [--compare BACKEND]
        [--ppm-every N --out DIR]
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np

from transformer_nes.nes.console import (
    BUTTONS,
    Console,
    profile_fallbacks,
    write_ppm,
)

logger = logging.getLogger(__name__)


def load_tape(path: str) -> list[tuple[int, int]]:
    """Tape file -> [(frame, button mask)] transitions, sorted."""
    events = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                events.append((int(parts[0]), int(parts[1], 16)))
    return sorted(events)


def mask_buttons(mask: int) -> dict:
    return {name: bool(mask >> i & 1) for i, name in enumerate(BUTTONS)}


class TapeDeck:
    """Applies a tape's controller state to a console, frame by frame."""

    def __init__(self, events: list[tuple[int, int]]):
        self.events = events
        self.next_event = 0
        self.mask = 0

    def apply(self, console: Console, frame: int) -> None:
        while (self.next_event < len(self.events)
               and self.events[self.next_event][0] <= frame):
            self.mask = self.events[self.next_event][1]
            self.next_event += 1
        console.controller.set_buttons(**mask_buttons(self.mask))


def instr_count(console: Console) -> int | None:
    """Executed-instruction counter for native backends (None otherwise)."""
    if console.uses_engine_bus:
        return int(console.rt.get_state()["steps"])
    return None


def replay(rom: str, events, *, backend: str, total: int, segment: int,
           ppm_every: int = 0, out_dir: str | None = None) -> Console:
    console = Console(rom, backend=backend)
    deck = TapeDeck(events)
    seg_start = time.perf_counter()
    seg_instr = instr_count(console)
    for frame in range(total):
        deck.apply(console, frame)
        rendered = console.run_frame()
        if console.halted:
            logger.info("halted at frame %d", frame)
            break
        if ppm_every and out_dir and (frame + 1) % ppm_every == 0:
            write_ppm(os.path.join(out_dir, f"frame_{frame + 1:06d}.ppm"),
                      rendered)
        if (frame + 1) % segment == 0:
            elapsed = time.perf_counter() - seg_start
            instr = instr_count(console)
            per_frame = ((instr - seg_instr) / segment
                         if instr is not None and seg_instr is not None else 0)
            logger.info(
                "frames %6d-%6d  %7.1f fps  %5.2f ms/frame  %5.0f instr/frame",
                frame + 1 - segment, frame + 1, segment / elapsed,
                elapsed / segment * 1e3, per_frame,
            )
            seg_start = time.perf_counter()
            seg_instr = instr
    return console


def compare(rom: str, events, *, backend: str, against: str,
            total: int) -> None:
    """Frame-by-frame lockstep of two backends under the same tape."""
    a = Console(rom, backend=backend)
    b = Console(rom, backend=against)
    deck_a, deck_b = TapeDeck(events), TapeDeck(events)
    for frame in range(total):
        deck_a.apply(a, frame)
        deck_b.apply(b, frame)
        fa, fb = a.run_frame(), b.run_frame()
        if not np.array_equal(fa, fb):
            diff = np.argwhere(fa != fb)
            logger.error(
                "FRAME %d DIVERGED: %d pixels differ, first at row %d col %d",
                frame, len(diff), diff[0][0], diff[0][1],
            )
            return
        if a.halted or b.halted:
            logger.info("halted at frame %d (%s=%s %s=%s)",
                        frame, backend, a.halted, against, b.halted)
            break
    logger.info("%d frames identical %s vs %s", frame + 1, backend, against)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Replay a recorded input tape headless.")
    parser.add_argument("rom", help="path to an iNES .nes file")
    parser.add_argument("tape", help="input tape from app --record")
    parser.add_argument(
        "--backend", choices=["reference", "graph", "weights"],
        default="weights")
    parser.add_argument("--segment", type=int, default=600,
                        help="report fps per this many frames")
    parser.add_argument("--tail", type=int, default=120,
                        help="frames to run past the tape's last event")
    parser.add_argument("--profile", type=int, default=0, metavar="MIN",
                        help="after the run, list fallback pcs >= MIN steps")
    parser.add_argument("--compare", metavar="BACKEND",
                        help="lockstep against this backend instead of "
                             "benchmarking; reports the first divergent frame")
    parser.add_argument("--ppm-every", type=int, default=0, metavar="N",
                        help="write a frame snapshot every N frames")
    parser.add_argument("--out", help="directory for frame snapshots")
    args = parser.parse_args()

    events = load_tape(args.tape)
    if not events:
        raise SystemExit(f"no input events in {args.tape}")
    total = events[-1][0] + args.tail
    logger.info("tape: %d events over %d frames (+%d tail)",
                len(events), events[-1][0], args.tail)
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    if args.compare:
        compare(args.rom, events, backend=args.backend,
                against=args.compare, total=total)
        return
    run_start = time.perf_counter()
    console = replay(args.rom, events, backend=args.backend, total=total,
                     segment=args.segment, ppm_every=args.ppm_every,
                     out_dir=args.out)
    elapsed = time.perf_counter() - run_start
    logger.info("total: %d frames in %.1fs (%.1f fps)%s",
                console.frame_index, elapsed,
                console.frame_index / elapsed,
                " [HALTED]" if console.halted else "")
    if args.profile:
        profile_fallbacks(console, args.profile)
    console.save_jit_proposals()


if __name__ == "__main__":
    main()
