"""Launch the NES: pick a CPU backend, open the Sokol window or replay a tape."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from transformer_nes.build import load_nes_extension
from transformer_nes.frontend.debug import DebugBridge
from transformer_nes.nes.backends import backend_factory

logger = logging.getLogger(__name__)


def run_app(
    console,
    frontend,
    *,
    title: str,
    scale: int,
    max_frames: int,
    record=None,
    debug=None,
    on_quit=None,
    overlay=False,
):
    elapsed = 0.0
    recent_frames = deque(maxlen=120)
    last_buttons = 0
    last_frame = None

    def frame(buttons: int):
        nonlocal elapsed, last_buttons, last_frame
        if console.halted or (max_frames and console.frame_index >= max_frames):
            return None
        if debug is not None and debug.pause and last_frame is not None:
            if debug.step_pending:
                debug.step_pending -= 1
            else:
                return last_frame  # hold the picture, no emulation
        if buttons != last_buttons:
            last_buttons = buttons
            if record is not None:
                names = (
                    "+".join(
                        name
                        for i, name in enumerate(
                            ("a", "b", "select", "start", "up", "down", "left", "right")
                        )
                        if buttons >> i & 1
                    )
                    or "(none)"
                )
                record.write(f"{console.frame_index} {buttons & 0xFF:02x} {names}\n")
                record.flush()
        console.set_buttons(buttons & 0xFF)
        start = time.perf_counter()
        rendered = console.run_frame()
        seconds = time.perf_counter() - start
        elapsed += seconds
        recent_frames.append(seconds)
        if debug is not None:
            debug.note_frame(seconds)
        if debug is None or not debug.mute:
            frontend.push_audio(console.last_audio)
        last_frame = rendered
        return rendered

    def quit_cb():
        if console.frame_index and recent_frames:
            logger.info(
                "%d frames, %.1f ms/frame average; %.1f ms/frame over last %d frames",
                console.frame_index,
                1e3 * elapsed / console.frame_index,
                1e3 * sum(recent_frames) / len(recent_frames),
                len(recent_frames),
            )
        if on_quit is not None:
            on_quit()

    frontend.run(title, scale, frame, quit_cb, debug, overlay)


def main():
    parser = argparse.ArgumentParser(description="Play an NES ROM")
    parser.add_argument("rom")
    parser.add_argument(
        "--backend",
        choices=("reference", "weights"),
        default="weights",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--record")
    parser.add_argument("--tape")
    parser.add_argument(
        "--benchmark-json", help="write tape timings, including individual frame samples, to JSON"
    )
    parser.add_argument(
        "--tail", type=int, default=120, help="frames to run after the final tape event"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--xray",
        action="store_true",
        help="open the overlay (~) on its transformer X-ray at startup",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    if args.tape and args.record:
        parser.error("--tape and --record cannot be used together")
    if args.benchmark_json and not args.tape:
        parser.error("--benchmark-json requires --tape")
    setup_start = time.perf_counter()
    hardware = load_nes_extension("hardware")
    cart = hardware.Cartridge.load(args.rom)

    runtimes = []  # the CPU runtime the console drives

    def build_backend(image, start_pc):
        runtime, is_engine = backend_factory(args.backend)(image, start_pc)
        runtimes.append(runtime)
        return runtime, is_engine

    if args.tape:
        console = hardware.Console(cart, build_backend, 44100.0)
        setup_seconds = time.perf_counter() - setup_start
        result = console.run_tape(args.tape, args.tail, args.frames)
        result.update(setup_seconds=setup_seconds, rom=args.rom, tape=args.tape, backend=args.backend)
        logger.info("%(frames)d frames in %(seconds).3fs: %(fps).1f fps, %(cycles)d cycles", result)
        for field, label in (("frame_seconds", "wall"), ("frame_cpu_seconds", "thread CPU")):
            samples = result[field]
            if samples:
                p1, p50, p99 = np.percentile(samples, [1, 50, 99]) * 1e3
                result[f"{field}_percentiles_ms"] = dict(p1=p1, p50=p50, p99=p99)
                logger.info("%s ms/frame: p1 %.3f, p50 %.3f, p99 %.3f", label, p1, p50, p99)
        if args.benchmark_json:
            Path(args.benchmark_json).write_text(json.dumps(result, indent=2) + "\n")
        return
    frontend = load_nes_extension("frontend")
    rate = frontend.audio_setup()

    console = hardware.Console(cart, build_backend, float(rate) if rate else 44100.0)

    # Battery-backed saves live next to the ROM; loaded before the first
    # frame so the game's boot sees them, written back on quit.  Tape
    # runs stay save-blind: a .sav at boot would desync recorded inputs.
    sav_path = args.rom + ".sav"
    if cart.battery and os.path.exists(sav_path):
        with open(sav_path, "rb") as sav:
            console.load_prg_ram(sav.read())
        logger.info("loaded battery save: %s", sav_path)

    def save_battery():
        if not cart.battery:
            return
        with open(sav_path, "wb") as sav:
            sav.write(console.prg_ram())
        logger.info("wrote battery save: %s", sav_path)

    debug = DebugBridge(
        console,
        args.backend,
        cpu_runtime=runtimes[0] if args.backend == "weights" else None,
        ppu_engine=console.ppu_engine,
    )

    def launch(record=None):
        run_app(
            console,
            frontend,
            title=f"transformer-nes: {os.path.basename(args.rom)} ({args.backend})",
            scale=args.scale,
            max_frames=args.frames,
            record=record,
            overlay=args.xray,
            debug=debug,
            on_quit=save_battery,
        )

    if args.record:
        with open(args.record, "w") as record:
            launch(record)
    else:
        launch()


if __name__ == "__main__":
    main()
