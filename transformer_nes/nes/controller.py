"""Controller state transitions and bounded serial reply packets as weights."""

from functools import cache

from transformer_nes.graph.core import InputDimension, construction, current_graph, persist, reglu
from transformer_nes.graph.ops import both, either, equals, remainder, select
from transformer_nes.model.graph_engine import CompiledGraph, TokenGraph

EVENTS = ("write", "read", "lease", "consume")
INPUTS = ("strobe", "index", *(f"button{i}" for i in range(8)), "event", "value", "count")
OUTPUTS = ("new_strobe", "new_index", "reply", "length", *(f"byte{i}" for i in range(8)))


@construction
def controller_graph():
    x = {name: InputDimension(name) for name in INPUTS}
    event = {name: equals(x["event"], i) for i, name in enumerate(EVENTS)}
    strobe, index = x["strobe"], x["index"]
    next_strobe = remainder(x["value"], 2, 255)
    new_strobe = select(event["write"], next_strobe, strobe)
    reset = both(event["write"], either(strobe, next_strobe))
    count = reglu(1, x["count"])
    advance = reglu(event["read"] + reglu(count, event["consume"]), 1 - strobe)
    new_index = select(reset, 0, index + advance - reglu(1, index + advance - 8))

    def reply(at):
        button = sum(reglu(x[f"button{i}"], equals(at, i)) for i in range(8))
        return 64 + button + reglu(1, at - 7) - reglu(1, at - 8)

    direct = select(strobe, 64 + x["button0"], reply(index))
    length = reglu(8 - index, 1 - strobe)
    values = (new_strobe, new_index, direct, length, *(reply(index + i) for i in range(8)))
    outputs = {name: persist(value, name=name) for name, value in zip(OUTPUTS, values, strict=True)}
    return current_graph().capture({"STATE": sum(x.values())}, outputs)


@cache
def controller_engine():
    return CompiledGraph(
        lambda: {
            "controller": TokenGraph(
                controller_graph(),
                INPUTS,
                OUTPUTS,
                {name: {"event": i} for i, name in enumerate(EVENTS)},
            )
        },
        tag="controller",
        sources=(__name__,),
        memory_size=1,
    )
