"""Bounded scanline clock and ordered event descriptors.

The host materializes the five distinct timelines once from these weights.
Line transitions and frame parity effects use the same circuit's readouts;
absolute timestamps remain host integers.
"""

from transformer_nes.graph.core import InputDimension, construction, current_graph, persist, stepglu
from transformer_nes.graph.ops import both, either, equals, gate, select

INPUTS = ("line", "index")
OUTPUTS = ("next_line", "wrap", "table", "count", "dot", "event", "slot", "run")
EVENTS = ("slot", "scanline", "sprites", "prefetch", "vertical_copy", "odd_skip",
          "frame_done", "vblank", "clear", "evaluate")


@construction
def clock_graph():
    x = {name: InputDimension(name) for name in INPUTS}
    line, index = x["line"], x["index"]
    visible = 1 - stepglu(1, line - 240)
    post, vblank, pre = (equals(line, n) for n in (240, 241, 261))
    idle = 1 - visible - post - vblank - pre

    def between(lo, hi):
        return both(stepglu(1, index - lo), 1 - stepglu(1, index - hi))

    slot = either(both(visible, either(equals(index, 0), either(between(2, 9), between(10, 34)))),
                  both(pre, between(1, 33)))
    slot_index = select(visible, index - stepglu(1, index - 1) - stepglu(1, index - 9), index - 1)
    ev = {
        "slot": slot,
        "scanline": both(visible, equals(index, 1)),
        "evaluate": both(visible, equals(index, 9)),
        "sprites": either(both(visible, equals(index, 34)), both(pre, equals(index, 33))),
        "prefetch": both(either(visible, pre), equals(index, 35)),
        "vertical_copy": both(pre, equals(index, 34)),
        "odd_skip": both(pre, equals(index, 36)),
        "frame_done": both(post, equals(index, 0)),
        "vblank": both(vblank, equals(index, 0)),
        "clear": both(pre, equals(index, 0)),
    }
    dot = gate(slot, 1 + 8 * slot_index)
    for name, at in (("scanline", 3), ("evaluate", 65), ("sprites", 257),
                     ("prefetch", 321), ("vertical_copy", 280), ("odd_skip", 339),
                     ("vblank", 1), ("clear", 1)):
        dot += at * ev[name]
    end = select(pre, 33, select(equals(index, 0), 1, select(stepglu(1, index - 10), 34, 9)))
    values = (
        line + 1 - 262 * pre,
        pre,
        post + 2 * vblank + 3 * idle + 4 * pre,
        36 * visible + post + vblank + 37 * pre,
        dot,
        sum(i * ev[name] for i, name in enumerate(EVENTS)),
        gate(slot, slot_index),
        1 + gate(slot, end - index - 1),
    )
    outputs = {name: persist(value, name=name) for name, value in zip(OUTPUTS, values, strict=True)}
    return current_graph().capture({"CLOCK": sum(x.values())}, outputs)
