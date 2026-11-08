"""Emit and load the native CPU scheduling shell.

Source templates live in native/. Sharding keeps native compilation bounded;
the emitter specializes embedding, readout and dispatch tables. Neural forwards
are compiled separately by bank.py; native/engine.cpp owns device arbitration.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path

from transformer_nes import build
from transformer_nes.cpu6502.execution import readout_slots
from transformer_nes.cpu6502.spec import (
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_ACCUMULATOR,
    MODE_IMMEDIATE,
    MODE_INDIRECT,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_RELATIVE,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
)

NATIVE_DIR = Path(__file__).with_name("native")
ENGINE_FILES = tuple(str(path) for path in sorted((NATIVE_DIR / "engine.cpp", NATIVE_DIR / "machine_abi.h")))


def native_source(name: str) -> str:
    return (NATIVE_DIR / name).read_text(encoding="utf-8")


# Bound translation-unit size separately from the generic build job limit.
DEFAULT_COMPILE_SHARDS = 12


def compile_shards() -> int:
    return build.positive_env_int("TRANSFORMER_NES_COMPILE_SHARDS", DEFAULT_COMPILE_SHARDS)


FUNC_RE = re.compile(r"^    (void|bool|int|long long) (step_core\w*)\((.*)\) \{$")


def shard_source(source: str, struct_name: str, n_shards: int) -> tuple[str, list[str]]:
    """Split a single-TU engine into a header + n_shards bodies so ninja
    compiles them in parallel.  The generated step_core* methods become out-of-class definitions, greedily balanced by line
    count; everything else stays in-class (implicitly inline) in the
    header.  The PYBIND module block lands in shard 0.  Purely a deterministic post-processing of the emitted
    text -- the digest covers the original."""
    lines = source.split("\n")
    header: list[str] = ["#pragma once"]
    chunks: dict[str, list[str]] = {}
    pybind: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = FUNC_RE.match(line)
        if m is not None:
            returns, name, args = m.groups()
            depth = line.count("{") - line.count("}")
            chunk = [f"{returns} {struct_name}::{name}({args}) {{"]
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                chunk.append(lines[i])
                i += 1
            header.append(f"    {returns} {name}({args});")
            chunks.setdefault(name, []).extend([*chunk, ""])
            continue
        if line.startswith("PYBIND11_MODULE("):
            depth = line.count("{") - line.count("}")
            pybind.append(line)
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                pybind.append(lines[i])
                i += 1
            continue
        header.append(line)
        i += 1

    shards: list[list[str]] = [[] for _ in range(max(1, n_shards))]
    sizes = [0] * len(shards)
    for chunk in sorted(chunks.values(), key=len, reverse=True):
        k = sizes.index(min(sizes))
        shards[k].extend(chunk)
        sizes[k] += len(chunk)
    shards[0] = pybind + [""] + shards[0]
    return "\n".join(header) + "\n", ["\n".join(s) + "\n" for s in shards]


def write_shards(cache_dir: str, stem: str, struct_name: str, source: str, n_shards: int) -> list[str]:
    """`source` as cache_dir/<stem>.hpp and <stem>_s<k>.cpp (shard_source),
    unless already written; the shards' paths."""
    hpp_path = os.path.join(cache_dir, f"{stem}.hpp")
    shard_paths = [os.path.join(cache_dir, f"{stem}_s{k}.cpp") for k in range(n_shards)]
    existing = [
        os.path.join(cache_dir, path)
        for path in os.listdir(cache_dir)
        if path.startswith(f"{stem}_s") and path.endswith(".cpp")
    ]
    if not os.path.exists(hpp_path) or set(existing) != set(shard_paths):
        header, shards = shard_source(source, struct_name, n_shards)
        for path, body in zip(shard_paths, shards, strict=True):
            build.write_atomic(path, f'#include "{stem}.hpp"\n\n{body}'.encode())
        build.write_atomic(hpp_path, header.encode())
        for path in set(existing) - set(shard_paths):
            os.unlink(path)
    return shard_paths


# What shapes the emitted engine: the runner
# that builds them and everything it imports (transformer_nes/build.py).
ENGINE_SOURCES = ("transformer_nes.cpu6502.runner",)


def engine_code_digest() -> str:
    return build.source_digest(*ENGINE_SOURCES, files=ENGINE_FILES)


def engine_config_key(*, bcd: bool, union_only: bool = False, mirror_cpu_ram: bool = False) -> str:
    """Cache key for a built engine, computable without folding or emitting:
    decimal mode plus every source file that shapes the fixed production core.
    A union-only engine holds no experts, so it keys separately."""
    raw = (
        f"{engine_code_digest()}|bcd={int(bcd)}|union_only={int(union_only)}"
        f"|mirror_cpu_ram={int(mirror_cpu_ram)}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


ADDRESS_MODE_CODE = {
    MODE_ZERO_PAGE: 1,
    MODE_ZERO_PAGE_X: 2,
    MODE_ZERO_PAGE_Y: 3,
    MODE_ABSOLUTE: 4,
    MODE_ABSOLUTE_X: 5,
    MODE_ABSOLUTE_Y: 6,
    MODE_INDIRECT_X: 7,
    MODE_INDIRECT_Y: 8,
}


TRACE_MODE_CODE = ADDRESS_MODE_CODE | {
    MODE_IMMEDIATE: 11,
    MODE_INDIRECT: 9,
    MODE_RELATIVE: 10,
}


def trace_mode_table() -> tuple[int, ...]:
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = TRACE_MODE_CODE.get(op.mode, 0)
    return tuple(table)


def trace_length_table() -> tuple[int, ...]:
    table = [1] * 256
    for byte, op in OPCODES.items():
        table[byte] = op.length
    return tuple(table)


def trace_data_table() -> tuple[int, ...]:
    """Trace source codes consumed by the runtime template's trace_data()."""
    kinds = {
        1: ("store_a", "pha"),
        2: ("store_x",),
        3: ("store_y",),
        4: ("php",),
        5: ("inc", "isb", "dec", "dcp", "sax", "slo", "rla", "sre", "rra"),
        6: ("load_a", "lax", "adc", "sbc", "and", "ora", "eor", "pla", "txa", "tya"),
        7: ("load_x", "inx", "dex", "tax", "tsx"),
        8: ("load_y", "iny", "dey", "tay"),
        9: ("txs",),
        10: ("plp",),
        11: ("cmp",),
        12: ("cpx",),
        13: ("cpy",),
        14: ("bit",),
    }
    codes = {operation: kind for kind, operations in kinds.items() for operation in operations}
    table = [0] * 256
    for byte, op in OPCODES.items():
        table[byte] = (
            (6 if op.mode == MODE_ACCUMULATOR else 5)
            if op.operation in {"asl", "lsr", "rol", "ror"}
            else codes.get(op.operation, 0)
        )
    return tuple(table)


def shape_run_cpp(exp) -> str:
    """A fixed shape's budget and recurrence descriptor."""
    loop = ", ".join(map(str, exp.get("loop", [0] * 5)))
    return (
        f"static constexpr int loop[] = {{{loop}}}; shape_runs.push_back(ShapeRun{{"
        f"{exp['n_steps']}, {exp['cycles_prefix']}, {exp['spin']}, read_loop(loop)}});"
    )


def generate_source(
    model, step_spec: dict, experts=None, mirror_cpu_ram: bool = False, *, union_id: int = 0
) -> str:
    """Emit a CPU engine with the native machine protocol.

    Core ``experts`` are compact opcode, interrupt or canonical shape models.
    """
    experts = experts or []
    if any(
        e.get("kind") not in {"exec_opcode", "shape", "exec_int", "exec_loop", "read_request"}
        for e in experts
    ):
        raise RuntimeError("core expert has an unsupported execution contract")

    d_model = model.tok.weight.shape[1]
    # The step_core variants share one residual buffer; an expert model may
    # schedule wider than the union (e.g. PLP), so size it for the widest.
    xbuf_size = max([d_model] + [e["model"].tok.weight.shape[1] for e in experts])
    readout_size = max(
        len(readout_slots(spec)) for spec in [step_spec] + [e["step_spec"] for e in experts]
    )

    lines: list[str] = []
    emit = lines.append

    elem = "int"

    # ── Step-core emission helpers ──────────────────────────────────
    one_e, zero_l = "SCALE", "0"

    def sc(v, e=elem):
        return f"(({e})({v}) << F)"

    def emit_step_core(
        fn_name,
        mdl,
        spec,
        *,
        stage_writes=False,
        embed_only=False,
        request=False,
        hook_id=None,
    ):
        # One model forward over persistent st_* state. The union emits one
        # diagnostic receipt; execution models emit only final state, control
        # and ordered effects. embed_only prepares the diagnostic input;
        # request=True emits only the address/timing packet and commits nothing.
        ins = spec["inputs"]
        positions = {slot: index for index, slot in enumerate(readout_slots(spec))}

        def rdv(slot):
            return f"shr2(readouts[{positions[slot]}])"

        sequence_final = spec.get("sequence_final")
        if stage_writes and sum(map(len, spec["sequence_writes"])) > 3:
            raise RuntimeError("compact execution exceeds three staged writes")
        needs_x = (
            not sequence_final
            or bool(mdl.attn)
            or bool(sequence_final["model"])
            or any(spec["sequence_writes"])
            or spec["sequence_halt"] is not None
        )
        if not needs_x and (ins or spec["p_flags"]):
            raise RuntimeError("zero-work execution model retained live inputs")

        args = ""
        if not sequence_final and not request and not embed_only:
            args += (", " if args else "") + "long long* out"
        returns = "ReadRequest" if request else "void"
        emit(f"    {returns} {fn_name}({args}) {{")
        if needs_x:
            emit(f"        {elem}* x = xbuf;")
            if embed_only:
                emit(f"        std::fill_n(x, {mdl.tok.weight.shape[1]}, 0);")
            emit(f"        x[{mdl.one_slot}] = {one_e};")

        def embed(key, value):
            if key in ins:
                emit(f"        x[{ins[key]}] = {value};")

        embed("is_state", one_e)
        embed("pc_lo", sc("st_pc & 0xFF"))
        embed("pc_hi", sc("(st_pc >> 8) & 0xFF"))
        embed("a", sc("st_a"))
        embed("x", sc("st_x"))
        embed("y", sc("st_y"))
        embed("s", sc("st_s"))
        for slot, mask in spec["p_flags"]:
            emit(f"        x[{slot}] = (st_p & {mask}) ? {one_e} : {zero_l};")
        embed("ir", sc("st_ir"))
        embed("oplo", sc("st_oplo"))
        embed("ophi", sc("st_ophi"))
        embed("ea_lo", sc("st_ea & 0xFF"))
        embed("ea_hi", sc("(st_ea >> 8) & 0xFF"))
        embed("data", sc("st_data"))
        embed("halted", sc("st_halted ? 1 : 0"))
        embed("read_addr_lo", sc("pending_read.address & 0xFF"))
        embed("read_addr_hi", sc("(pending_read.address >> 8) & 0xFF"))
        embed("read_cycles", sc("pending_read.cycles"))
        embed("loop_limit", sc("loop_limit"))
        embed("loop_cycles", sc("loop_cycles"))
        for bit in range(8):
            embed(f"a_bit{bit}", sc(f"(st_a >> {bit}) & 1"))
        if "nmi" in ins:
            emit("        const bool take_nmi = nmi_pending && pending_read.address < 0;")
            embed("nmi", sc("take_nmi ? 1 : 0"))
            if not embed_only:
                emit("        if (take_nmi) nmi_pending = false;")
        if "irq" in ins:
            emit("        const bool take_irq = irq_ready() && pending_read.address < 0;")
            embed("irq", sc("take_irq ? 1 : 0"))
            if not embed_only:
                emit("        if (take_irq) irq_pending = false;")
        if embed_only:
            emit("    }")
            return
        if not request:
            emit("        ++seq;")
        if needs_x:
            emit(
                f"        if (!forward_hook || !forward_hook(forward_hook_ctx, {hook_id}, x, readouts))"
            )
            emit('            throw std::runtime_error("Torch transformer bank is not attached");')
        if request:
            outs = spec["outputs"]
            emit(
                f"        return {{(int){rdv(outs['address'])}, (int){rdv(outs['cycles'])}, "
                f"(int){rdv(outs['oplo'])}, (int){rdv(outs['ophi'])}}};"
            )
            emit("    }")
            return
        if sequence_final:
            outs = sequence_final["model"]
            constants = sequence_final["constant"]
            fields = {
                "new_pc": ("st_pc", 0xFFFF),
                "new_a": ("st_a", 0xFF),
                "new_x": ("st_x", 0xFF),
                "new_y": ("st_y", 0xFF),
                "new_s": ("st_s", 0xFF),
            }
            for name, (field, mask) in fields.items():
                if name in outs:
                    emit(f"        {field} = (int)({rdv(outs[name])} & 0x{mask:X});")
                elif name in constants:
                    emit(f"        {field} = {int(constants[name]) & mask};")
            flag_masks = {
                "p_c": 0x01,
                "p_z": 0x02,
                "p_i": 0x04,
                "p_d": 0x08,
                "p_b": 0x10,
                "p_v": 0x40,
                "p_n": 0x80,
            }
            for name, mask in flag_masks.items():
                if name in outs:
                    value = f"({rdv(outs[name])} & 1) ? 0x{mask:X} : 0"
                elif name in constants:
                    op = "|=" if constants[name] else "&="
                    value = f"0x{mask:X}" if constants[name] else f"~0x{mask:X}"
                    emit(f"        st_p {op} {value};")
                    continue
                else:
                    continue
                emit(f"        st_p = (st_p & ~0x{mask:X}) | ({value});")
            cycles = spec["sequence_cycles"]
            emit(
                "        seq_cycles = "
                + " + ".join([str(cycles["constant"])] + [rdv(slot) for slot in cycles["model"]])
                + ";"
            )
            if "iterations" in spec.get("outputs", {}):
                emit(f"        loop_iterations = (int){rdv(spec['outputs']['iterations'])};")
            if spec["sequence_halt"] is not None:
                emit(f"        st_halted = ({rdv(spec['sequence_halt'])} & 1) != 0;")
            for step_writes in spec["sequence_writes"]:
                for write in step_writes:
                    addr_slot, value_slot = write["address"], write["value"]
                    if stage_writes:
                        emit(
                            f"        exec_write_addr[exec_n_writes] = "
                            f"(int)({rdv(addr_slot)} & 0xFFFF);"
                        )
                        emit(
                            f"        exec_write_value[exec_n_writes] = "
                            f"(int)({rdv(value_slot)} & 0xFF);"
                        )
                        original = "-1" if write["original"] is None else f"(int){rdv(write['original'])}"
                        emit(f"        exec_write_original[exec_n_writes] = {original};")
                        emit("        ++exec_n_writes;")
                    else:
                        emit(
                            f"        insert_write((int)({rdv(addr_slot)} & 0xFFFF), "
                            f"(int)({rdv(value_slot)} & 0xFF), seq + 1);"
                        )
            emit("    }")
            return
        outs = spec["outputs"]
        for i, (name, mask) in enumerate(
            (
                ("current_pc", 0xFFFF),
                ("new_a", 0xFF),
                ("new_x", 0xFF),
                ("new_y", 0xFF),
                ("new_s", 0xFF),
                ("new_p", 0xFF),
                ("fetched_opcode", 0xFF),
                ("fetched_operand_lo", 0xFF),
                ("fetched_operand_hi", 0xFF),
                ("ea", 0xFFFF),
                ("data_out", 0xFF),
                ("new_halted", 0xFF),
                ("next_pc", 0xFFFF),
            )
        ):
            emit(f"        out[{i}] = {rdv(outs[name])} & 0x{mask:X};")
        emit("        { long long n_writes = 0;")
        for store, addr, value in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            emit(f"        if (({rdv(outs[store])} & 0xFF) != 0) {{")
            emit(f"            out[15 + 2 * n_writes] = {rdv(outs[addr])} & 0xFFFF;")
            emit(f"            out[16 + 2 * n_writes] = {rdv(outs[value])} & 0xFF;")
            emit("            ++n_writes;")
            emit("        }")
        emit("        out[14] = n_writes;")
        emit(f"        out[21] = {rdv(outs['cycles'])} & 0xFF;")
        emit(f"        out[22] = {rdv(outs['write_original_out'])};")
        emit("        for (long long i = 0; i < n_writes; ++i)")
        emit("            insert_write((int)out[15 + 2 * i], (int)out[16 + 2 * i], seq + 1);")
        emit("        }")
        emit("        out[13] = out[11] != 0;")
        emit("        // Advance the architectural state (inputs are fully")
        emit("        // consumed by the embed above before this update).")
        emit("        st_pc = (int)out[12];")
        emit("        st_a = (int)out[1]; st_x = (int)out[2]; st_y = (int)out[3];")
        emit("        st_s = (int)out[4]; st_p = (int)out[5];")
        emit("        st_ir = (int)out[6]; st_oplo = (int)out[7]; st_ophi = (int)out[8];")
        emit("        st_ea = (int)out[9]; st_data = (int)out[10];")
        emit("        st_halted = out[11] != 0;")
        emit("    }")

    emit_step_core("step_core", model, step_spec, hook_id=union_id)
    emit("")
    emit_step_core("embed_union", model, step_spec, embed_only=True)
    emit("")
    for j, exp in enumerate(experts):
        emit("")
        emit_step_core(
            f"step_core__{exp['name']}",
            exp["model"],
            exp["step_spec"],
            stage_writes=exp.get("kind") in ("exec_opcode", "exec_int"),
            request=exp.get("kind") == "read_request",
            hook_id=exp.get("hook_id", j),
        )

    forwards = "\n".join(lines) + "\n\n"
    lines.clear()

    # Compact opcode execution is selected by the byte at st_pc. Pending
    # interrupts and tracing bypass it through step_core's union receipt.
    exec_op_experts = [e for e in experts if e.get("kind") == "exec_opcode"]
    shape_experts = [e for e in experts if e.get("kind") == "shape"]
    emit("")
    emit("    // Ordered writes staged for the device boundary.")
    emit("    int exec_n_writes = 0;")
    emit("    int exec_write_addr[3], exec_write_value[3], exec_write_original[3];")
    # Compact per-opcode dispatch; an opcode with no case returns false
    # and the run loop takes the union step_core.
    emit("    bool execution_step(unsigned char opcode, [[maybe_unused]] bool read_serviced) {")
    emit("        exec_n_writes = 0;")
    if exec_op_experts:
        emit("        switch (opcode) {")
        for exp in exec_op_experts:
            emit(f"            case 0x{exp['byte']:02X}:")
            if exp.get("external_only"):
                emit("                if (!read_serviced) return false;")
            emit(f"                step_core__{exp['name']}(); return true;")
        emit("        }")
    emit("        return false;")
    emit("    }")
    emit("")
    emit("    ReadRequest request_read() {")
    emit("        read_ir = ram[st_pc];")
    emit("        switch (read_ir) {")
    for exp in experts:
        if exp.get("kind") == "read_request":
            for byte in exp["bytes"]:
                emit(f"        case 0x{byte:02X}:")
            emit(f"            return step_core__{exp['name']}();")
    emit("        }")
    emit("        return {-1, 0, 0, 0};")
    emit("    }")

    int_experts = {e["nmi"]: e for e in experts if e.get("kind") == "exec_int"}
    emit("")
    emit("    bool interrupt_step(bool nmi) {")
    emit("        exec_n_writes = 0;")
    for is_nmi in (True, False):
        exp = int_experts.get(is_nmi)
        if exp is None:
            continue
        cond = "nmi" if is_nmi else "!nmi"
        emit(f"        if ({cond}) {{ step_core__{exp['name']}(); return true; }}")
    emit("        return false;")
    emit("    }")

    emit("    void counter_recurrence(int reg, int delta, int limit, int cycles) {")
    emit("        loop_limit = limit; loop_cycles = cycles;")
    for exp in experts:
        if exp.get("kind") == "exec_loop":
            emit(f"        if (reg == {exp['reg']} && delta == {exp['delta']}) {{")
            emit(f"            step_core__{exp['name']}(); return;")
            emit("        }")
    emit('        throw std::runtime_error("counter recurrence transformer is missing");')
    emit("    }")

    # Canonical shape windows are compiled once and revalidated against RAM.
    emit("")
    emit("    unsigned char shape_of_pc[65536];")
    n_shapes = len(shape_experts)
    if n_shapes:
        lens = ", ".join(str(e["length"]) for e in shape_experts)
        emit(f"    static constexpr int SHAPE_LEN[{n_shapes}] = {{ {lens} }};")
    emit("    void add_window(int shape, int head) {")
    if n_shapes:
        emit(f"        if (shape < 0 || shape >= {n_shapes})")
        emit('            throw std::runtime_error("unknown shape id");')
        emit("        if (head < 0 || head + SHAPE_LEN[shape] > 0x10000)")
        emit('            throw std::runtime_error("shape window would wrap $FFFF");')
        emit("        shape_of_pc[head] = (unsigned char)(shape + 1);")
    else:
        emit("        (void)shape; (void)head;")
        emit('        throw std::runtime_error("engine has no shape experts");')
    emit("    }")
    emit("    void clear_windows() { std::memset(shape_of_pc, 0, sizeof(shape_of_pc)); }")
    emit("")
    emit("    int shape_dispatch(long i, long max_steps, long cycles_this,")
    emit("                       long max_cycles, bool stop_on_trap) {")
    if n_shapes:
        emit("        const int sid = shape_of_pc[st_pc];")
        emit("        if (!sid || nmi_pending || irq_ready() || stop_on_trap) return 0;")
        emit("        // A device reply may have changed instruction bytes. The")
        emit("        // first instruction must still match its retained prefix.")
        emit("        if (pending_read.address >= 0 && (ram[st_pc] != read_ir ||")
        emit("            ram[st_pc + 1] != pending_read.oplo ||")
        emit(
            "            (TRACE_LENGTH[read_ir] > 2 && ram[st_pc + 2] != pending_read.ophi))) return 0;"
        )
        emit("        switch (sid) {")
        for j, exp in enumerate(shape_experts):
            # add_window guarantees head + length <= 0x10000: no wrap.
            checks = " && ".join(
                f"ram[st_pc + {off}] == 0x{b:02X}" if off else f"ram[st_pc] == 0x{b:02X}"
                for off, b in exp["opcode_checks"]
            )
            if exp["spin"]:
                checks += (
                    " && ram[st_pc + 1] == (st_pc & 0xFF)"
                    " && ram[st_pc + 2] == ((st_pc >> 8) & 0xFF)"
                )
            forward = f"step_core__{exp['name']}();"
            emit(f"        case {j + 1}:  // {exp['name']}")
            emit(f"            if ({checks})")
            emit(
                f"                return run_shape(shape_runs[{j}], i, max_steps, cycles_this, max_cycles,"
            )
            emit(f"                                  [&] {{ {forward} }});")
            emit("            return 0;")
        emit("        }")
    emit("        return 0;")
    emit("    }")
    dispatch = "\n".join(lines).lstrip("\n") + "\n"
    source = native_source("engine.cpp")
    replacements = {
        "@MACHINE_ABI@": native_source("machine_abi.h").removeprefix("#pragma once\n"),
        "@MIRROR_CPU_RAM@": "true" if mirror_cpu_ram else "false",
        "@D_MODEL@": str(d_model),
        "@XBUF_SIZE@": str(xbuf_size),
        "@READOUT_SIZE@": str(readout_size),
        "@FORWARDS@": forwards,
        "@DISPATCH@": dispatch,
        "@SHAPE_RUNS@": "".join(f"        {{ {shape_run_cpp(exp)} }}\n" for exp in shape_experts),
        "@TRACE_MODE@": ", ".join(map(str, trace_mode_table())),
        "@TRACE_LENGTH@": ", ".join(map(str, trace_length_table())),
        "@TRACE_DATA@": ", ".join(map(str, trace_data_table())),
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    return source


class GeneratedWeightsEngine:
    """The 6502 transformer as generated exact-int32 C++."""

    # Native methods bound straight off the pybind module (bind()): the hot
    # loops call them with no wrapper frame in between.
    run: Callable
    set_run_state: Callable
    get_run_state: Callable
    machine_abi: Callable
    watch_read: Callable
    watch_write: Callable
    set_nmi: Callable
    set_irq: Callable
    clear_irq: Callable
    seed_image: Callable
    map_region: Callable
    read_ram: Callable
    dump_ram: Callable
    add_window: Callable
    clear_windows: Callable
    digest: str

    def __init__(
        self,
        model,
        *,
        step_spec: dict,
        experts=None,
        cache_key: str | None = None,
        shapes_meta=None,
        mirror_cpu_ram: bool = False,
        union_id: int = 0,
    ):
        self.d_model = int(model.tok.weight.shape[1])
        self.n_experts = len(experts or [])
        self.mirror_cpu_ram = bool(mirror_cpu_ram)
        self.union_id = union_id
        # Shape-window scan metadata (id/signature/length/close rule) so a
        # cache-hit load can populate the runtime table without refolding.
        self.shapes_meta = list(shapes_meta or [])
        source = generate_source(
            model,
            step_spec,
            experts=experts,
            mirror_cpu_ram=self.mirror_cpu_ram,
            union_id=union_id,
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.digest = digest
        source = source.replace("ENGINE_T", f"Engine_{digest}")
        n_shards = compile_shards()
        with build.artifact_lock(digest):
            sources = write_shards(
                build.CACHE_DIR, f"engine_{digest}", f"Engine_{digest}", source, n_shards
            )
            module = build.load_module(f"weights_gen_{digest}", digest, sources)
            if cache_key is not None:
                build.manifest_put(
                    cache_key,
                    {
                        "digest": digest,
                        "n_experts": self.n_experts,
                        "mirror_cpu_ram": self.mirror_cpu_ram,
                        "union_id": union_id,
                        "shapes": self.shapes_meta,
                        "compile_shards": n_shards,
                        "d_model": self.d_model,
                    },
                    code=engine_code_digest(),
                )
        self.bind(module)

    @classmethod
    def from_cache(cls, cache_key: str) -> GeneratedWeightsEngine | None:
        """Load an engine by config key alone, skipping the expert folds and
        the source emission entirely; None on any miss."""
        entry = build.manifest_read().get(cache_key)
        if entry is None:
            return None
        with build.artifact_lock(entry["digest"]):
            # A publisher may replace the entry or prune its files between
            # the initial snapshot and acquiring the artifact lock.
            if build.manifest_read().get(cache_key) != entry:
                return None
            if entry.get("compile_shards") != compile_shards():
                return None
            if not os.path.exists(os.path.join(build.CACHE_DIR, f"engine_{entry['digest']}.hpp")):
                return None
            digest = entry["digest"]
            sources = [
                os.path.join(build.CACHE_DIR, f"engine_{digest}_s{k}.cpp")
                for k in range(entry["compile_shards"])
            ]
            if not all(map(os.path.exists, sources)):
                return None
            module = build.load_module(f"weights_gen_{digest}", digest, sources)
        self = cls.__new__(cls)
        self.digest = entry["digest"]
        self.n_experts = entry["n_experts"]
        self.mirror_cpu_ram = entry["mirror_cpu_ram"]
        self.union_id = entry["union_id"]
        self.shapes_meta = entry["shapes"]
        self.d_model = entry["d_model"]
        self.bind(module)
        return self

    def bind(self, module) -> None:
        self.engine = module.Engine()
        # Bound methods exposed directly: the hot loops call them with no
        # wrapper frame in between.
        self.reset = self.engine.reset
        self.inject_write = self.engine.inject_write
        self.run = self.engine.run
        self.set_run_state = self.engine.set_run_state
        self.get_run_state = self.engine.get_run_state
        self.watch_read = self.engine.watch_read
        self.watch_write = self.engine.watch_write
        self.set_nmi = self.engine.set_nmi
        self.set_irq = self.engine.set_irq
        self.clear_irq = self.engine.clear_irq
        self.seed_image = self.engine.seed_image
        self.map_region = self.engine.map_region
        self.read_ram = self.engine.read_ram
        self.dump_ram = self.engine.dump_ram
        self.add_window = self.engine.add_window
        self.clear_windows = self.engine.clear_windows
        self.machine_abi = self.engine.machine_abi
