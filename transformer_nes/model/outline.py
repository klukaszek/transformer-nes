"""Share repeated SSA fragments in the compiler input.

Fragments are always inlined by the native compiler. Their scalar bindings
and output buffers disappear during optimization; no interpreter or dynamic
instruction dispatch is introduced into a neural forward.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from transformer_nes.model import scalar_ir as ir


def _fragment(program, instructions, defined):
    code, terms, bindings, outputs = [], [], [], []
    local = {}

    def operand(slot):
        if slot in local:
            return local[slot]
        arg = len(code)
        code.append((ir.INPUT, arg, len(bindings), 0, 0))
        # Give each use its own formal argument. Different alias patterns
        # and coefficient values can share a fragment; the call binds them
        # exactly and native constant propagation recovers their identities.
        bindings.append(("value", int(slot)) if slot in defined
                        else ("constant", program["registers"][slot].item()))
        return arg

    for op, dst, a, b, c in instructions:
        if op == ir.LINEAR:
            row = [(operand(s), int(k)) for s, k in program["terms"][a:a + b]]
            a, b = len(terms), len(row)
            terms.extend(row)
        else:
            a = operand(a)
            if op not in (ir.LOAD, ir.RELU):
                b = operand(b)
            if op in (ir.WHERE, ir.CLAMP):
                c = operand(c)
        result = len(code)
        code.append((op, result, a, b, c))
        local[dst] = result
        outputs.append(result)
    fragment = dict(
        code=np.asarray(code, dtype=np.int32).reshape(-1, 5),
        terms=np.asarray(terms, dtype=np.int32).reshape(-1, 2),
        registers=np.zeros(len(code), dtype=program["registers"].dtype),
        outputs=np.asarray(outputs, dtype=np.int32), reads=np.empty(0, dtype=np.int32),
        inputs=len(bindings),
    )
    key = tuple(fragment[name].tobytes() for name in ("code", "terms", "outputs"))
    return key, fragment, bindings


def share(programs, *, scalar="int32_t"):
    """Return helper source and per-program nonoverlapping fragment calls.

    Only fragments with a net source saving across actual unique circuits
    are selected. Multiple window lengths recover repeated hardware units
    even when preceding circuits have different sizes or slot layouts.
    """
    groups = defaultdict(list)
    fragments = {}
    for pi, program in enumerate(programs):
        defined = {int(row[1]) for row in program["code"]}
        rows = program["code"].tolist()
        last_use = dict.fromkeys([*program["outputs"], *program["reads"]], len(rows))
        for i in range(len(rows) - 1, -1, -1):
            op, _dst, a, b, c = rows[i]
            if op == ir.INPUT:
                continue
            if op == ir.LINEAR:
                operands = [s for s, _k in program["terms"][a:a + b]]
            else:
                operands = [a] if op in (ir.LOAD, ir.RELU) else [a, b]
                if op in (ir.WHERE, ir.CLAMP):
                    operands.append(c)
            for slot in operands:
                last_use.setdefault(slot, i)
        instructions = [(i, row) for i, row in enumerate(rows)
                        if row[0] != ir.INPUT]
        lines = ir.emit_cpp(program, scalar=scalar)[0].splitlines()[1:1 + len(program["code"])]
        for size, stride in ((8, 4), (16, 8), (32, 8), (64, 16)):
            for start in range(0, len(instructions) - size + 1, stride):
                run = instructions[start:start + size]
                key, fragment, bindings = _fragment(program, [row for _, row in run], defined)
                raw = sum(len(lines[i]) + 1 for i, _ in run)
                fragments.setdefault(key, fragment)
                end = run[-1][0] + 1
                needed = sum(1 << j for j, (_i, row) in enumerate(run)
                             if last_use.get(row[1], -1) >= end)
                if needed:
                    groups[key].append((pi, start, size, run[0][0], end, bindings, raw, needed))

    def available(entries, occupied):
        # Also reject overlaps between occurrences of the same fragment.
        ends = {}
        kept = []
        for entry in sorted(entries, key=lambda e: (e[0], e[1])):
            pi, start, size = entry[:3]
            if start < ends.get(pi, 0) or occupied[pi][start:start + size].any():
                continue
            kept.append(entry)
            ends[pi] = start + size
        return kept

    occupied = [np.zeros(len(p["code"]), dtype=bool) for p in programs]

    def body_for(key, entries):
        needed = 0
        for entry in entries:
            needed |= entry[-1]
        positions = [i for i in range(len(fragments[key]["outputs"])) if needed & (1 << i)]
        original = fragments[key]
        registers = original["registers"].copy()
        # Intersect equality across all calls. Shared constants become
        # literals, and arguments equal at every call need only one binding.
        signatures, columns, remap, rows = {}, [], {}, []
        for op, dst, a, b, c in original["code"].tolist():
            if op != ir.INPUT:
                rows.append((op, dst, a, b, c))
                continue
            values = [entry[5][a] for entry in entries]
            signature = tuple((kind, value if kind == "value" or registers.dtype.kind == "i" else
                               np.asarray(value, dtype=registers.dtype).tobytes())
                              for kind, value in values)
            if all(kind == "constant" for kind, _ in values) and len(set(signature)) == 1:
                registers[dst] = values[0][1]
            elif signature in signatures:
                remap[dst] = signatures[signature]
            else:
                signatures[signature] = dst
                rows.append((op, dst, len(columns), 0, 0))
                columns.append(a)
        code = np.asarray(rows, dtype=np.int32).reshape(-1, 5)
        terms = original["terms"].copy()
        for row in code:
            op = row[0]
            if op in (ir.INPUT, ir.LINEAR):
                continue
            row[2] = remap.get(row[2], row[2])
            if op not in (ir.LOAD, ir.RELU):
                row[3] = remap.get(row[3], row[3])
            if op in (ir.WHERE, ir.CLAMP):
                row[4] = remap.get(row[4], row[4])
        for row in terms:
            row[0] = remap.get(row[0], row[0])
        fragment = original | dict(
            code=code, terms=terms, registers=registers,
            outputs=original["outputs"][positions], inputs=len(columns),
        )
        body = ir.emit_cpp(fragment, scalar=scalar)[0]
        gain = 0
        for entry in entries:
            # Allow for names becoming buffer references in earlier calls.
            call = 90 + sum(12 if entry[5][i][0] == "value" else len(str(entry[5][i][1])) + 2
                            for i in columns)
            gain += entry[-2] - call
        return body, positions, columns, gain - len(body) - 70

    ranked = []
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        kept = available(entries, occupied)
        _body, _positions, _columns, saving = body_for(key, kept)
        if saving > 0:
            ranked.append((saving, key))

    helpers, plans = [], [{} for _ in programs]
    for _, key in sorted(ranked, key=lambda x: -x[0]):
        kept = available(groups[key], occupied)
        if len(kept) < 2:
            continue
        body, positions, columns, saving = body_for(key, kept)
        if saving <= 0:
            continue
        name = f"block_{len(helpers)}"
        helpers.append(body.replace(
            "void circuit_0(", f"static inline __attribute__((always_inline)) void {name}(", 1,
        ))
        for pi, start, size, begin, end, bindings, _saving, _needed in kept:
            occupied[pi][start:start + size] = True
            plans[pi][begin] = (end, name, [bindings[i] for i in columns], positions)
    return "\n".join(helpers), plans
