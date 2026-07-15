"""Build the native NES, adapt a CPU runtime backend, and launch Sokol."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import time

import numpy as np

from transformer_nes.cpu6502.backends.reference import irq_deliverable
from transformer_nes.cpu6502.spec import (
    MEMORY_MODES,
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
    READS_MEMORY,
)

logger = logging.getLogger(__name__)
NES_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(NES_DIR, "_nes_cache")
SOKOL_DIR = os.path.normpath(os.path.join(NES_DIR, "..", "..", "vendor", "sokol"))
SOKOL_HEADERS = ("sokol_app.h", "sokol_gfx.h", "sokol_glue.h", "sokol_log.h", "sokol_audio.h")
SOKOL_IMPL = os.path.join(NES_DIR, "sokol_impl.m")
MACOS_FRAMEWORKS = ("Cocoa", "QuartzCore", "Metal", "MetalKit", "AudioToolbox")
HARDWARE_SOURCES = ("bindings.cpp", "bus.cpp", "ppu.cpp", "apu.cpp", "console.cpp", "tape.cpp")
HARDWARE_HEADERS = (
    "cartridge.h",
    "mapper.h",
    "mappers.h",
    "bus.h",
    "ppu.h",
    "apu.h",
    "console.h",
    "tape.h",
    "mmc1.h",
    "mmc2.h",
    "mmc3.h",
    "mmc4.h",
    "mmc5.h",
)
CXX_FLAGS = ("-O2", "-std=c++26", "-Wall", "-Wextra", "-Wpedantic")

NES_PALETTE = np.array(
    [
        (84, 84, 84),
        (0, 30, 116),
        (8, 16, 144),
        (48, 0, 136),
        (68, 0, 100),
        (92, 0, 48),
        (84, 4, 0),
        (60, 24, 0),
        (32, 42, 0),
        (8, 58, 0),
        (0, 64, 0),
        (0, 60, 0),
        (0, 50, 60),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (152, 150, 152),
        (8, 76, 196),
        (48, 50, 236),
        (92, 30, 228),
        (136, 20, 176),
        (160, 20, 100),
        (152, 34, 32),
        (120, 60, 0),
        (84, 90, 0),
        (40, 114, 0),
        (8, 124, 0),
        (0, 118, 40),
        (0, 102, 120),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (236, 238, 236),
        (76, 154, 236),
        (120, 124, 236),
        (176, 98, 236),
        (228, 84, 236),
        (236, 88, 180),
        (236, 106, 100),
        (212, 136, 32),
        (160, 170, 0),
        (116, 196, 0),
        (76, 208, 32),
        (56, 204, 108),
        (56, 180, 204),
        (60, 60, 60),
        (0, 0, 0),
        (0, 0, 0),
        (236, 238, 236),
        (168, 204, 236),
        (188, 188, 236),
        (212, 178, 236),
        (236, 174, 236),
        (236, 174, 212),
        (236, 180, 176),
        (228, 196, 144),
        (204, 210, 120),
        (180, 222, 120),
        (168, 226, 144),
        (152, 226, 180),
        (160, 214, 228),
        (160, 162, 160),
        (0, 0, 0),
        (0, 0, 0),
    ],
    dtype=np.uint8,
)


def _build_dir(component: str, digest: str) -> str:
    path = os.path.join(CACHE_DIR, component, f"build_{digest}")
    os.makedirs(path, exist_ok=True)
    return path


def _digest(paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as source:
            digest.update(source.read())
    return digest.hexdigest()[:16]


def _load(name: str, sources: tuple[str, ...], headers: tuple[str, ...], **kwargs):
    from torch.utils.cpp_extension import load

    tag = _digest(sources + headers)
    return load(
        name=f"{name}_{tag}",
        sources=list(sources),
        build_directory=_build_dir(name, tag),
        verbose=False,
        **kwargs,
    )


def _build_sokol() -> str:
    headers = tuple(os.path.join(SOKOL_DIR, name) for name in SOKOL_HEADERS)
    tag = _digest((SOKOL_IMPL, *headers))
    build_dir = _build_dir("sokol", tag)
    object_path = os.path.join(build_dir, "sokol_impl.o")
    if os.path.exists(object_path):
        return object_path

    temporary = f"{object_path}.{os.getpid()}.tmp"
    compiler = os.environ.get("CC", "cc")
    try:
        subprocess.run(
            [
                compiler,
                "-std=c17",
                "-O2",
                "-fPIC",
                "-x",
                "objective-c",
                "-I",
                SOKOL_DIR,
                "-c",
                SOKOL_IMPL,
                "-o",
                temporary,
            ],
            check=True,
        )
        os.replace(temporary, object_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return object_path


def load_hardware():
    return _load(
        "nes_hardware",
        tuple(os.path.join(NES_DIR, x) for x in HARDWARE_SOURCES),
        tuple(os.path.join(NES_DIR, x) for x in HARDWARE_HEADERS),
        extra_cflags=list(CXX_FLAGS),
    )


def load_frontend():
    sources = (os.path.join(NES_DIR, "app.cpp"),)
    headers = tuple(os.path.join(SOKOL_DIR, x) for x in SOKOL_HEADERS)
    ldflags = [_build_sokol()]
    ldflags.extend(flag for framework in MACOS_FRAMEWORKS for flag in ("-framework", framework))
    return _load(
        "nes_app",
        sources,
        (SOKOL_IMPL, *headers),
        extra_cflags=list(CXX_FLAGS),
        extra_include_paths=[SOKOL_DIR],
        extra_ldflags=ldflags,
    )


class ReferenceCore:
    def __init__(self, cpu):
        self.cpu = cpu

    pc = property(lambda s: s.cpu.pc)
    x = property(lambda s: s.cpu.x)
    y = property(lambda s: s.cpu.y)
    halted = property(lambda s: s.cpu.halted)
    nmi_pending = property(lambda s: s.cpu.nmi_pending)
    irq_ready = property(lambda s: irq_deliverable(s.cpu.p, s.cpu.irq_pending))

    def set_nmi(self):
        self.cpu.nmi_pending = True

    def set_irq(self):
        self.cpu.irq_pending = True

    def clear_irq(self):
        self.cpu.irq_pending = False

    def poke(self, addr, value):
        self.cpu.memory[addr & 0xFFFF] = value & 0xFF

    def step(self):
        return self.cpu.step()


class GraphCore:
    def __init__(self, runtime):
        self.runtime = runtime

    pc = property(lambda s: s.runtime.mstate["pc"])
    x = property(lambda s: s.runtime.mstate["x"])
    y = property(lambda s: s.runtime.mstate["y"])
    halted = property(lambda s: s.runtime.machine_halted)
    nmi_pending = property(lambda s: s.runtime.nmi_pending)
    irq_ready = property(lambda s: irq_deliverable(s.runtime.mstate["p"], s.runtime.irq_pending))
    log_tokens = property(lambda s: s.runtime.pos)

    def set_nmi(self):
        self.runtime.set_nmi()

    def set_irq(self):
        self.runtime.set_irq()

    def clear_irq(self):
        self.runtime.clear_irq()

    def poke(self, addr, value):
        self.runtime.inject_write(addr, value)

    def step(self):
        return self.runtime.machine_step()

    def checkpoint(self, _ram):
        self.runtime.checkpoint()


def backend_factory(name: str):
    def build(image: bytes, start_pc: int):
        if name == "weights":
            from transformer_nes.cpu6502.backends.weights.runner import Cpu6502TransformerMachine

            return Cpu6502TransformerMachine(bcd=False), True
        if name == "reference":
            from transformer_nes.cpu6502.backends.reference import Cpu6502

            cpu = Cpu6502(memory=bytearray(image), bcd_enabled=False)
            cpu.pc = start_pc
            return ReferenceCore(cpu), False
        if name == "graph":
            from transformer_nes.cpu6502.backends.graph import Cpu6502Runtime

            runtime = Cpu6502Runtime(bcd=False)
            runtime.boot_image(list(image), start_pc=start_pc)
            return GraphCore(runtime), False
        raise ValueError(f"unknown backend {name!r}")

    return build


MODE_CODES = {
    MODE_ZERO_PAGE: 0,
    MODE_ZERO_PAGE_X: 1,
    MODE_ZERO_PAGE_Y: 2,
    MODE_ABSOLUTE: 3,
    MODE_ABSOLUTE_X: 4,
    MODE_ABSOLUTE_Y: 5,
    MODE_INDIRECT_X: 6,
    MODE_INDIRECT_Y: 7,
}
OPCODES_NATIVE = [(0, -1)] * 256
for byte, op in OPCODES.items():
    mode = MODE_CODES[op.mode] if op.operation in READS_MEMORY and op.mode in MEMORY_MODES else -1
    OPCODES_NATIVE[byte] = (op.cycles, mode)


def run_app(console, frontend, *, title: str, scale: int, max_frames: int, record=None):
    elapsed = 0.0
    last_buttons = 0

    def frame(buttons: int):
        nonlocal elapsed, last_buttons
        if console.halted or (max_frames and console.frame_index >= max_frames):
            return None
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
        elapsed += time.perf_counter() - start
        frontend.push_audio(console.last_audio)
        return rendered

    def quit_cb():
        if console.frame_index:
            logger.info(
                "%d frames, %.1f ms/frame", console.frame_index, 1e3 * elapsed / console.frame_index
            )

    frontend.run(title, scale, frame, NES_PALETTE, quit_cb)


def main():
    parser = argparse.ArgumentParser(description="Play an NES ROM")
    parser.add_argument("rom")
    parser.add_argument("--backend", choices=("reference", "graph", "weights"), default="weights")
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--record")
    parser.add_argument("--tape")
    parser.add_argument(
        "--tail", type=int, default=120, help="frames to run after the final tape event"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s")
    if args.tape and args.record:
        parser.error("--tape and --record cannot be used together")
    hardware = load_hardware()
    cart = hardware.Cartridge.load(args.rom)
    if args.tape:
        console = hardware.Console(cart, backend_factory(args.backend), OPCODES_NATIVE, 44100.0)
        result = console.run_tape(args.tape, args.tail, args.frames)
        logger.info("%(frames)d frames in %(seconds).3fs: %(fps).1f fps, %(cycles)d cycles", result)
        return
    frontend = load_frontend()
    rate = frontend.audio_setup()
    console = hardware.Console(
        cart, backend_factory(args.backend), OPCODES_NATIVE, float(rate) if rate else 44100.0
    )

    def launch(record=None):
        run_app(
            console,
            frontend,
            title=f"transformer-nes: {os.path.basename(args.rom)} ({args.backend})",
            scale=args.scale,
            max_frames=args.frames,
            record=record,
        )

    if args.record:
        with open(args.record, "w") as record:
            launch(record)
    else:
        launch()


if __name__ == "__main__":
    main()
