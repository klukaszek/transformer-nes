"""The 6502 as a chip of the NES model (nes/model.py).

Three token types over the CPU's memory:

- ``union``: the whole instruction circuit (any byte, either interrupt),
  reading out the instruction's effects (graph.py RUNTIME_READ_DIMS). The
  host falls back to it, and the X-ray traces it.
- ``step``: one specialization per compiled execution -- an opcode, an
  interrupt, a shape of several instructions, a counter loop -- each its
  own folded circuit (experts.py) with the same named outputs: the new
  registers and flags, halting, cycles, loop iterations and up to three
  ordered writes. Values a fold proved constant or left untouched are
  outputs too, so the host commits every step the same way.
- ``request``: the address and timing of an instruction's device read,
  before it runs.
"""

from __future__ import annotations

from transformer_nes.cpu6502.execution import (
    INPUT_SLOT_NAMES,
    OPTIONAL_INPUT_SLOT_NAMES,
    P_FLAG_NAMES,
    STATE_FIELDS,
    ExecutionGraph,
)
from transformer_nes.cpu6502.graph import RUNTIME_READ_DIMS, Cpu6502Graph
from transformer_nes.graph.core import (
    Expression,
    GraphContext,
    InputDimension,
    ProgramGraph,
    persist,
)
from transformer_nes.model.construct import TokenFamily, TokenGraph, construct_chip
from transformer_nes.model.transformer import Chip

# Every token's inputs: the architectural state, the instruction and read
# context, interrupt lines and loop bounds, as the host embeds them.
INPUTS = (
    *INPUT_SLOT_NAMES.values(),
    *(name for name, _mask in P_FLAG_NAMES),
    *OPTIONAL_INPUT_SLOT_NAMES.values(),
)
WRITES = 3
STEP_OUTPUTS = (
    *(f"step_{name.removeprefix('new_')}" for name, _final, _input in STATE_FIELDS),
    "step_halted",
    "step_cycles",
    "step_iterations",
    "step_writes",
    *(f"step_w{k}_{field}" for k in range(WRITES) for field in ("addr", "value", "original")),
)
REQUEST_OUTPUTS = ("request_address", "request_cycles", "request_oplo", "request_ophi")
UNION_OUTPUTS = RUNTIME_READ_DIMS
MEMORY_SIZE = 0x10000


def _named_outputs(graph: ProgramGraph, values: dict[str, Expression]) -> ProgramGraph:
    """`graph` with each value persisted under its name as a readout, and an
    input token for any input dim the values add."""
    context = GraphContext.extending(graph)
    with context.activate():
        readout = Expression()
        for name, value in values.items():
            readout = readout + 1e-6 * persist(value, name=name)
        added = [d for d in context.dims[len(graph.all_dims):] if isinstance(d, InputDimension)]
        inputs = dict(graph.input_tokens)
        if added:
            inputs["PASSTHROUGH"] = sum((d.as_expr() for d in added), Expression())
        return context.capture(inputs, {**graph.output_tokens, "STEP": readout})


def step_circuit(execution: ExecutionGraph, iterations=None) -> ProgramGraph:
    """An execution's circuit with the step token's outputs."""
    graph = execution.program_graph
    one = graph.one
    inputs = {d.name: d for d in graph.all_dims if isinstance(d, InputDimension)}

    def input_dim(name):
        # An input the circuit never read: it passes straight through.
        return inputs.get(name) or InputDimension(name)

    context = GraphContext.extending(graph)
    values = {}
    with context.activate():
        by_name = {out.name: out for out in execution.outputs}
        for name, _final, input_name in STATE_FIELDS:
            out = by_name[name]
            key = f"step_{name.removeprefix('new_')}"
            if out.source == "model":
                values[key] = out.dimension.as_expr()
            elif out.source == "constant":
                values[key] = out.constant * one
            elif input_name is None:  # the PC, untouched: its two bytes
                values[key] = input_dim("in_pc_lo") + 256 * input_dim("in_pc_hi")
            else:
                values[key] = input_dim(input_name).as_expr()
        values["step_halted"] = (
            execution.halt_dimension.as_expr() if execution.halt_dimension is not None
            else input_dim("in_halted").as_expr()
        )
        cycles = Expression()
        for c in execution.cycles:
            cycles = cycles + (c * one if isinstance(c, int) else c.as_expr())
        values["step_cycles"] = cycles
        values["step_iterations"] = iterations.as_expr() if iterations is not None else 0 * one
        writes = sorted(execution.memory_writes, key=lambda w: (w.step, w.order))
        if len(writes) > WRITES:
            raise ValueError("a step makes at most three writes")
        values["step_writes"] = len(writes) * one
        for k in range(WRITES):
            write = writes[k] if k < len(writes) else None
            values[f"step_w{k}_addr"] = write.address.as_expr() if write else 0 * one
            values[f"step_w{k}_value"] = write.value.as_expr() if write else 0 * one
            values[f"step_w{k}_original"] = (
                write.original.as_expr() if write and write.original is not None else -1 * one
            )
        # Extend once: the passthrough inputs above joined this context.
        graph = ProgramGraph(
            graph.input_tokens, graph.output_tokens, tuple(context.dims),
            tuple(context.lookups), tuple(context.memory_reads), one,
        )
    return _named_outputs(graph, values)


def request_circuit(graph: ProgramGraph) -> ProgramGraph:
    by_name = {}
    for d in graph.all_dims:
        by_name.setdefault(d.name, d)
    fields = ("ea", "cycles", "fetched_operand_lo", "fetched_operand_hi")
    return _named_outputs(
        graph, {out: by_name[f].as_expr() for out, f in zip(REQUEST_OUTPUTS, fields, strict=True)}
    )


def experts(bcd: bool = False) -> tuple[ProgramGraph, list[dict], list[dict]]:
    """The instruction graph (construction folds it into the union circuit),
    the compiled executions (experts.py) and the shapes' metadata."""
    from transformer_nes.cpu6502.experts import build_core_experts
    from transformer_nes.cpu6502.shapes import with_loops

    pg = Cpu6502Graph(bcd=bcd, wide_bits=True).program_graph()
    core, meta = build_core_experts(pg)
    return pg, with_loops(core), meta


def facts(e: dict) -> dict:
    """What the run loop needs to know of an execution to dispatch it."""
    kind = e["kind"]
    if kind == "exec_opcode":
        return {"kind": kind, "byte": e["byte"], "read_resume": bool(e.get("read_resume"))}
    if kind == "exec_int":
        return {"kind": kind, "nmi": e["nmi"]}
    if kind == "exec_loop":
        return {"kind": kind, "reg": e["reg"], "delta": e["delta"]}
    if kind == "read_request":
        return {"kind": kind, "bytes": list(e["bytes"])}
    return {
        "kind": kind, "shape_id": e["shape_id"], "length": e["length"], "steps": e["n_steps"],
        "prefix": e["cycles_prefix"], "spin": e["spin"], "loop": list(e["loop"]),
        "checks": [list(c) for c in e["opcode_checks"]],
    }


def chip(bcd: bool = False) -> Chip:
    """The CPU, without decimal mode on the NES (bcd builds the 6502's)."""
    union, core, windows = experts(bcd)
    steps, requests = {}, {}
    for e in core:
        if e["kind"] == "read_request":
            requests[e["name"]] = request_circuit(e["graph"])
        else:
            iterations = None
            if e["kind"] == "exec_loop":
                iterations = e["execution"].dim_of_name["loop_iterations"]
            steps[e["name"]] = step_circuit(e["execution"], iterations)
    return construct_chip(
        {
            "union": TokenGraph(union, INPUTS, UNION_OUTPUTS),
            "step": TokenFamily(INPUTS, STEP_OUTPUTS, steps),
            "request": TokenFamily(INPUTS, REQUEST_OUTPUTS, requests),
        },
        memory_size=MEMORY_SIZE,
        constants={
            "executions": {e["name"]: facts(e) for e in core},
            # Shape windows to find in newly mapped code (shapes.py).
            "windows": [dict(w) for w in windows],
        },
    )


def tables(cpu, *, union_only: bool = False) -> dict:
    """The run loop's dispatch tables (native/include/nes/cpu.h CpuTables)
    for a compiled CPU chip (model/backend.py:CompiledChip). `union_only`
    runs every instruction through the union circuit except the operand
    continuation of a served read (a diagnostic of the union)."""
    from transformer_nes.cpu6502.trace import trace_data_table, trace_length_table, trace_mode_table

    none = -1
    t = {
        "union": cpu.model_id("union"), "exec": [none] * 256, "exec_after_read": [False] * 256,
        "request": [none] * 256, "nmi": none, "irq": none, "loops": [], "shapes": [],
        "trace_mode": list(trace_mode_table()), "trace_length": list(trace_length_table()),
        "trace_data": list(trace_data_table()),
    }
    shapes = {}
    for name, f in cpu.chip.constants["executions"].items():
        kind = f["kind"]
        if kind == "read_request":
            for byte in f["bytes"]:
                t["request"][byte] = cpu.model_id("request", name)
            continue
        model = cpu.model_id("step", name)
        if kind == "exec_opcode":
            if union_only and not f["read_resume"]:
                continue
            t["exec"][f["byte"]] = model
            t["exec_after_read"][f["byte"]] = union_only
        elif union_only:
            continue
        elif kind == "exec_int":
            t["nmi" if f["nmi"] else "irq"] = model
        elif kind == "exec_loop":
            t["loops"].append((f["reg"], f["delta"], model))
        else:
            shapes[f["shape_id"]] = {
                "model": model, "length": f["length"], "steps": f["steps"], "prefix": f["prefix"],
                "spin": f["spin"], "checks": f["checks"], "self_target": f["spin"] != 0, "loop": f["loop"],
            }
    t["shapes"] = [shapes[i] for i in sorted(shapes)]
    # Whether each circuit reads the interrupt lines: inputs it never reads
    # have no residual slot.
    lines = [INPUTS.index("in_nmi"), INPUTS.index("in_irq")]
    t["reads_interrupts"] = [
        bool((circuit.input_slots[lines] >= 0).any()) for _t, _s, circuit, _o in cpu.chip.programs()
    ]
    return t
