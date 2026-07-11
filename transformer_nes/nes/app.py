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


DUMP_DIR = os.path.join(NES_DIR, "_debug_dumps")


def debug_dump(console: Console) -> None:
    """F1: snapshot the frame + PPU/mapper state for offline analysis."""
    import numpy as np

    os.makedirs(DUMP_DIR, exist_ok=True)
    ppu = console.ppu
    n = console.frame_index
    path = os.path.join(DUMP_DIR, f"dump_{n:06d}.npz")
    cpu = console.snapshot()[1]
    state = {
        "pc": f"{int(cpu['pc']):04X}", "p": f"{int(cpu['p']):02X}",
        "i_flag": bool(int(cpu["p"]) & 4),
        "live_ctrl": ppu.ctrl, "live_mask": ppu.mask,
        "live_mirroring": ppu.mirroring, "live_t": ppu.t_addr,
        "live_fine_x": ppu.fine_x, "live_banks": list(ppu.chr_banks),
        "frame_ctrl": ppu.frame_ctrl, "frame_mask": ppu.frame_mask,
        "frame_mirroring": ppu.frame_mirroring, "frame_t": ppu.frame_t_addr,
        "frame_fine_x": ppu.frame_fine_x,
        "mid_ctrl": ppu.mid_ctrl, "mid_mask": ppu.mid_mask,
        "mid_mirroring": ppu.mid_mirroring, "mid_t": ppu.mid_t_addr,
        "mid_fine_x": ppu.mid_fine_x,
        "sprite0_scanline": ppu.sprite0_scanline,
    }
    if console.mapper.irq_scanline() is not None:
        state.update(irq_scanline=console.mapper.irq_scanline(),
                     irq_enabled=console.mapper.irq_enabled,
                     irq_latch=console.mapper.irq_latch,
                     prg_banks=list(console.mapper.prg_banks()),
                     mapper_chr=list(console.mapper.chr_banks()))
    np.savez(path,
             frame=np.asarray(console.run_frame()),
             nametables=np.frombuffer(bytes(ppu.nametables), dtype=np.uint8),
             palette=np.frombuffer(bytes(ppu.palette), dtype=np.uint8),
             oam=np.frombuffer(bytes(ppu.oam), dtype=np.uint8),
             **{k: np.asarray(v) for k, v in state.items()})
    logger.info("debug dump: %s | %s", path,
                " ".join(f"{k}={v}" for k, v in state.items()))


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
        if buttons & 0x100 and not last_buttons & 0x100:
            debug_dump(console)
        if buttons != last_buttons:
            last_buttons = buttons
            pressed = [name for i, name in enumerate(BUTTONS) if buttons >> i & 1]
            logger.debug("buttons: %s", "+".join(pressed) or "(none)")
        console.controller.set_buttons(
            **{name: bool(buttons >> i & 1) for i, name in enumerate(BUTTONS)}
        )
        start = time.perf_counter()
        frame = console.run_frame()
        elapsed = time.perf_counter() - start
        emu_seconds += elapsed
        if elapsed > 0.1:
            logger.warning(
                "slow frame %d: %.0f ms (checkpoints=%d, log=%s tokens)",
                console.frame_index, 1e3 * elapsed, console.checkpoint_count,
                getattr(console.bus, "log_tokens", "?"),
            )
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
