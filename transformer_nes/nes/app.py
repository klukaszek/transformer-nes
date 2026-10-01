"""Launch the NES: pick a CPU backend, open the Sokol window or replay a tape."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time

from transformer_nes.files import write_atomic
from transformer_nes.nes.backends import backend_factory
from transformer_nes.nes.build import load_frontend, load_hardware
from transformer_nes.nes.debug import DebugBridge

logger = logging.getLogger(__name__)


# ── Learned windows (profile-guided, per ROM) ──
# A weights run profiles which code blocks the CPU enters; the hottest
# become per-ROM sequence windows (runner.learn_windows), saved in windows/
# and attached at boot on later runs.  Interactive runs learn
# them live (nes/live_windows.py); a --tape run learns after the tape.


# Learned windows are data, not a build: they live outside the engine
# cache, which can be cleared at will.
WINDOWS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "windows")


def windows_path(rom: str) -> str:
    with open(rom, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:16]
    os.makedirs(WINDOWS_DIR, exist_ok=True)
    return os.path.join(WINDOWS_DIR, f"{digest}.json")


def load_windows(path: str) -> list[tuple[int, bytes, int, bool, tuple[bool, ...]]]:
    """(head, code, 1, store forwarding, branch predictions) per learned
    window (a file from before predictions has none)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [(head, bytes.fromhex(code), 1, forward, tuple(taken))
                for head, code, forward, *rest in json.load(f)
                for taken in [rest[0] if rest else ()]]


def save_windows(path: str, windows) -> None:
    known = {(head, code, forward, taken) for head, code, _, forward, taken in load_windows(path)}
    known |= {(head, bytes(code), forward, tuple(taken))
              for head, code, _, forward, taken in windows}
    write_atomic(path, json.dumps(sorted(
        [head, code.hex(), forward, list(taken)] for head, code, forward, taken in known)).encode())
    logger.info("saved %d learned windows to %s", len(known), path)


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
    on_frame=None,
    overlay=False,
):
    elapsed = 0.0
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
        if debug is not None:
            debug.note_frame(seconds)
        if debug is None or not debug.mute:
            frontend.push_audio(console.last_audio)
        last_frame = rendered
        if on_frame is not None:
            on_frame(console.frame_index)
        return rendered

    def quit_cb():
        if console.frame_index:
            logger.info(
                "%d frames, %.1f ms/frame", console.frame_index, 1e3 * elapsed / console.frame_index
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
    parser.add_argument(
        "--ppu",
        choices=("native", "weights"),
        default="native",
        help="run the PPU register file, background rendering and sprite unit as "
        "transformer weights (transformer_nes/ppu2c02)",
    )
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--record")
    parser.add_argument("--tape")
    parser.add_argument(
        "--tail", type=int, default=120, help="frames to run after the final tape event"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cpu-forwards",
        choices=("native", "torch"),
        default="native",
        help="where the 6502 experts' forward passes run: the generated C++ engine, or "
        "PyTorch, each expert AOTInductor-compiled and called from the native run loop "
        "(cached; missing ones compile in the background while the engine runs their "
        "forwards, and a --tape run waits for them)",
    )
    parser.add_argument(
        "--learn-windows",
        action="store_true",
        help="start without the ROM's learned sequence windows and learn them again "
        "(interactive runs always learn live; a --tape run learns after the tape, "
        "automatically on a ROM's first one)",
    )
    parser.add_argument(
        "--xray",
        action="store_true",
        help="open the overlay (~) on its transformer X-ray at startup",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    if args.tape and args.record:
        parser.error("--tape and --record cannot be used together")
    hardware = load_hardware()
    cart = hardware.Cartridge.load(args.rom)

    ppu_engines = []  # the PPU transformer, for the X-ray
    runtimes = []  # the CPU runtime the console drives
    torch_forwards = []  # its forwards in PyTorch (--cpu-forwards torch)
    learned = windows_path(args.rom)
    learn = args.backend == "weights" and (args.learn_windows or not os.path.exists(learned))

    def build_backend(image, start_pc):
        hooked = args.backend == "weights" and args.cpu_forwards != "native"
        runtime, is_engine = backend_factory(args.backend, forward_hook=hooked)(
            image, start_pc
        )
        runtimes.append(runtime)
        if args.backend == "weights" and not args.learn_windows:
            # A tape run builds missing windows first; a game starts without
            # them and builds them in the background (below).
            runtime.attach_sequence_module(load_windows(learned), build=bool(args.tape))
        if hooked:
            from transformer_nes.cpu6502.backends.weights.torch_forward import TorchForwards

            torch_forwards.append(TorchForwards(runtime, wait=bool(args.tape)))
        return runtime, is_engine

    def learn_windows():
        if learn and runtimes:
            save_windows(learned, runtimes[0].learn_windows())

    def attach_ppu(console):
        if args.ppu == "weights":
            from transformer_nes.ppu2c02.machine import ppu_engine

            ppu_engines[:] = [ppu_engine(console.ppu_graph_chr_full)]
            console.set_ppu_engine(ppu_engines[0])

    if args.tape:
        console = hardware.Console(cart, build_backend, 44100.0)
        attach_ppu(console)
        result = console.run_tape(args.tape, args.tail, args.frames)
        logger.info("%(frames)d frames in %(seconds).3fs: %(fps).1f fps, %(cycles)d cycles", result)
        learn_windows()
        return
    frontend = load_frontend()
    rate = frontend.audio_setup()

    console = hardware.Console(
        cart, build_backend, float(rate) if rate else 44100.0
    )
    attach_ppu(console)

    # Battery-backed saves live next to the ROM; loaded before the first
    # frame so the game's boot sees them, written back on quit.  Tape
    # runs stay save-blind: a .sav at boot would desync recorded inputs.
    sav_path = args.rom + ".sav"
    if cart.battery and os.path.exists(sav_path):
        with open(sav_path, "rb") as sav:
            console.load_prg_ram(sav.read())
        logger.info("loaded battery save: %s", sav_path)

    # Windows are learned while the game runs (nes/live_windows.py) and
    # attach between frames.
    learner = None
    if args.backend == "weights":
        from transformer_nes.nes.live_windows import WindowLearner

        learner = WindowLearner(runtimes[0], learned)
        saved = os.path.exists(learned) and not args.learn_windows
        built = runtimes[0].seq_key and runtimes[0].windows_folded()
        if saved and not built:
            learner.prepare()

    def on_frame(frame: int):
        if learner is not None and learner.tick(frame):
            runtimes[0].attach_sequence_module(load_windows(learned))
            for forwards in torch_forwards:
                forwards.attach_windows()
            logger.info("attached %d learned windows", len(runtimes[0].seq_windows))

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
        ppu_engine=ppu_engines[0] if ppu_engines else None,
    )

    def launch(record=None):
        run_app(
            console,
            frontend,
            title=(
                f"transformer-nes: {os.path.basename(args.rom)} ({args.backend}"
                f"{', PPU as weights' if args.ppu == 'weights' else ''})"
            ),
            scale=args.scale,
            max_frames=args.frames,
            record=record,
            overlay=args.xray,
            debug=debug,
            on_quit=save_battery,
            on_frame=on_frame,
        )

    if args.record:
        with open(args.record, "w") as record:
            launch(record)
    else:
        launch()


if __name__ == "__main__":
    main()
