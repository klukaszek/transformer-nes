"""Native NES app: the Sokol window over the console frame loop.

The window, GPU blit, and keyboard live in `nes/app.cpp` (vendor/sokol
headers, macOS/Metal), compiled once and cached by content digest like
the PPU.  Python stays the orchestrator: the app calls back here once
per NTSC frame with the controller bitmask, and the callback runs
Console.run_frame() on the chosen backend and returns the RGB frame.

Keys: arrows = d-pad, X = A, Z = B, Enter = Start, Tab = Select,
Esc = quit.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time

from transformer_nes.nes.console import BUTTONS, Console
from transformer_nes.nes.ppu import to_rgb

logger = logging.getLogger(__name__)

NES_DIR = os.path.dirname(os.path.abspath(__file__))
APP_CPP = os.path.join(NES_DIR, "app.cpp")
SOKOL_DIR = os.path.normpath(os.path.join(NES_DIR, "..", "..", "vendor", "sokol"))
CACHE_DIR = os.path.join(NES_DIR, "_app_cache")

SOKOL_HEADERS = ("sokol_app.h", "sokol_gfx.h", "sokol_glue.h", "sokol_log.h")
MACOS_FRAMEWORKS = ("Cocoa", "QuartzCore", "Metal", "MetalKit")

app_module = None


def load_app_module():
    """Compile-or-load the native app, cached by source+header digest."""
    global app_module
    if app_module is None:
        from torch.utils.cpp_extension import load

        digest = hashlib.sha256()
        for path in [APP_CPP] + [os.path.join(SOKOL_DIR, h) for h in SOKOL_HEADERS]:
            with open(path, "rb") as f:
                digest.update(f.read())
        digest = digest.hexdigest()[:16]
        build_dir = os.path.join(CACHE_DIR, f"build_{digest}")
        os.makedirs(build_dir, exist_ok=True)
        ldflags = [flag for fw in MACOS_FRAMEWORKS for flag in ("-framework", fw)]
        app_module = load(
            name=f"nes_app_{digest}",
            sources=[APP_CPP],
            extra_cflags=["-O2", "-std=c++17", "-x", "objective-c++"],
            extra_include_paths=[SOKOL_DIR],
            extra_ldflags=ldflags,
            build_directory=build_dir,
            verbose=False,
        )
    return app_module


def run_app(console: Console, *, title: str = "transformer-nes", scale: int = 3,
            max_frames: int = 0) -> None:
    """Open the window and play the console until quit.

    Does not return on macOS: quitting exits the process inside the
    native run loop ([NSApp terminate:]), so the run stats are logged
    from the quit callback -- the last Python to execute.
    """
    module = load_app_module()
    emu_seconds = 0.0
    last_buttons = 0

    def frame(buttons: int):
        nonlocal emu_seconds, last_buttons
        if console.halted or (max_frames and console.frame_index >= max_frames):
            return None
        if buttons != last_buttons:
            last_buttons = buttons
            pressed = [name for i, name in enumerate(BUTTONS) if buttons >> i & 1]
            logger.debug("buttons: %s", "+".join(pressed) or "(none)")
        console.controller.set_buttons(
            **{name: bool(buttons >> i & 1) for i, name in enumerate(BUTTONS)}
        )
        start = time.perf_counter()
        frame = console.run_frame()
        emu_seconds += time.perf_counter() - start
        return to_rgb(frame)

    def quit_cb():
        if console.frame_index and emu_seconds > 0:
            logger.info(
                "%d frames, emulation %.1f ms/frame (headroom: %.0f fps uncapped)%s",
                console.frame_index, 1e3 * emu_seconds / console.frame_index,
                console.frame_index / emu_seconds,
                " [HALTED]" if console.halted else "",
            )

    module.run(title, scale, frame, quit_cb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play an NES ROM (NROM/MMC1) in a native window.")
    parser.add_argument("rom", help="path to an iNES .nes file")
    parser.add_argument(
        "--backend", choices=["kernel", "reference", "graph", "weights"], default="weights"
    )
    parser.add_argument("--scale", type=int, default=3, help="window size in framebuffers")
    parser.add_argument("--frames", type=int, default=0,
                        help="quit after this many frames (0: play until closed)")
    parser.add_argument("--debug", action="store_true", help="log button changes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(message)s")

    build_start = time.perf_counter()
    console = Console(args.rom, backend=args.backend)
    load_app_module()
    logger.info("machine + app build %.2fs", time.perf_counter() - build_start)

    logger.info("keys: arrows = d-pad, X = A, Z = B, Enter = Start, "
                "Tab = Select, Esc = quit")
    title = f"transformer-nes: {os.path.basename(args.rom)} ({args.backend})"
    stats = run_app(console, title=title, scale=args.scale, max_frames=args.frames)

    frames, emu = stats["frames"], stats["emu_seconds"]
    if frames and emu > 0:
        logger.info(
            "%d frames, emulation %.1f ms/frame (headroom: %.0f fps uncapped)%s",
            frames, 1e3 * emu / frames, frames / emu,
            " [HALTED]" if console.halted else "",
        )


if __name__ == "__main__":
    main()
