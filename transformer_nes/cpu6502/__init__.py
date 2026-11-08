"""6502 state types and an explicit reference-emulator API."""

from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow

__all__ = ["Cpu6502", "Cpu6502State", "MemWrite", "Trace", "TraceRow", "run"]


def __getattr__(name):
    if name in {"Cpu6502", "run"}:
        from transformer_nes.cpu6502 import reference

        return getattr(reference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
