"""Scalar SSA encoding, native emission and repeated-fragment sharing.

encode converts FX operations to scalar values; emit_cpp writes each value
once. share_fragments finds repeated instruction windows across circuits,
specializes common arguments and emits helpers that the native compiler inlines.
Integer wraparound, half-unit shifts and float32 operation order remain explicit.
"""

from __future__ import annotations

import operator
from collections import defaultdict

import numpy as np
import torch

INPUT, ADD, MUL, AND, SHL, SHR, EQ, GT, GE, LT, WHERE, CLAMP, DIV, LOAD, RELU, REGLU = range(16)
LINEAR = 16
BINARY = {
    operator.add: ADD, operator.mul: MUL, operator.and_: AND,
    operator.lshift: SHL, operator.rshift: SHR, operator.eq: EQ,
    operator.gt: GT, operator.ge: GE, operator.lt: LT,
}


def encode(module):
    """Build SSA values, preserving int32 overflow and float32 boundaries."""
    floating = module.dtype == torch.float32
    dtype = np.float32 if floating else np.int32
    constants, values, registers, code, terms = {}, {}, [], [], []
    emitted, forms, linears = {}, {}, {}

    def wrap(value):
        return (int(value) + 2**31) % 2**32 - 2**31

    def affine(node):
        if not isinstance(node, torch.fx.Node):
            return {None: int(node)}
        if node not in forms:
            if node.op == "call_function" and node.target is torch.full:
                form = {None: int(node.args[1])}
            elif node.op == "call_function" and node.target is operator.add:
                form = dict(affine(node.args[0]))
                for term, coefficient in affine(node.args[1]).items():
                    form[term] = form.get(term, 0) + coefficient
            elif node.op == "call_function" and node.target is operator.mul:
                a, b = (affine(arg) for arg in node.args)
                if set(a) <= {None}:
                    form = {term: a.get(None, 0) * c for term, c in b.items()}
                elif set(b) <= {None}:
                    form = {term: b.get(None, 0) * c for term, c in a.items()}
                else:
                    form = {node: 1}
            else:
                form = {node: 1}
            forms[node] = {term: wrap(c) for term, c in form.items() if wrap(c)}
        return forms[node]

    def linear(form):
        combined = {}
        for term, coefficient in form.items():
            if term is not None:
                slot = get(term)
                combined[slot] = combined.get(slot, 0) + coefficient
        weighted = tuple(sorted((s, wrap(c)) for s, c in combined.items() if wrap(c)))
        bias = form.get(None, 0)
        if not weighted:
            return const(bias)
        if len(weighted) == 1 and weighted[0][1] == 1 and bias == 0:
            return weighted[0][0]
        key = weighted, bias
        if key not in linears:
            offset = len(terms)
            terms.extend(weighted)
            linears[key] = emit(LINEAR, offset, len(weighted), bias)
        return linears[key]

    def const(value):
        key = np.asarray(value, dtype=dtype).tobytes()
        if key not in constants:
            constants[key] = len(registers)
            registers.append(value)
        return constants[key]

    def emit(op, a=0, b=0, c=0):
        key = op, a, b, c
        if key in emitted:
            return emitted[key]
        dst = len(registers)
        registers.append(0)
        code.append((op, dst, a, b, c))
        emitted[key] = dst
        return dst

    def reglu(node):
        if floating or node.target is not torch.where or len(node.args) != 3:
            return None
        condition, product, otherwise = node.args
        if otherwise != 0 or not isinstance(condition, torch.fx.Node):
            return None
        if condition.target is not operator.gt or condition.args[1] != 0:
            return None
        shift = 0
        if isinstance(product, torch.fx.Node) and product.target is operator.rshift:
            product, shift = product.args
        if not isinstance(product, torch.fx.Node) or product.target is not operator.mul:
            return None
        gate = condition.args[0]
        a, b = product.args
        if gate is a:
            return gate, b, shift
        if gate is b:
            return gate, a, shift
        return None

    def get(value):
        if not isinstance(value, torch.fx.Node):
            return const(value)
        if value in values:
            return values[value]
        node, target, args = value, value.target, value.args
        if (not floating and getattr(module, "canonical_affine", True)
                and node.op == "call_function" and affine(node) != {node: 1}):
            result = linear(affine(node))
        elif node.op == "placeholder":
            result = str(target)
        elif node.op == "call_method" and target == "to":
            result = get(args[0])
        elif node.op == "call_function":
            fused = reglu(node)
            if fused is not None:
                gate, payload, shift = fused
                result = emit(REGLU, get(gate), get(payload), shift)
            elif target is torch.full:
                result = const(args[1])
            elif target is torch.empty:
                result = []
            elif target is operator.getitem:
                source, index = args
                if get(source) == "x":
                    if not isinstance(index, slice) or index.stop != index.start + 1:
                        raise ValueError("circuit input must be a scalar slice")
                    result = emit(INPUT, index.start)
                elif get(source) == "memory":
                    result = emit(LOAD, get(index))
                else:
                    raise ValueError("unsupported circuit indexing")
            elif target in BINARY:
                result = emit(BINARY[target], get(args[0]), get(args[1]))
            elif target in (torch.where, torch.clamp):
                result = emit(WHERE if target is torch.where else CLAMP, *(get(x) for x in args))
            elif target is torch.div:
                if node.kwargs != {"rounding_mode": "floor"}:
                    raise ValueError("unsupported circuit division")
                result = emit(DIV, get(args[0]), get(args[1]))
            elif target is torch.relu:
                result = emit(RELU, get(args[0]))
            elif target is torch.cat:
                result = [r for arg in args[0] for r in flatten(get(arg))]
            else:
                raise ValueError(f"unsupported circuit function: {target}")
        else:
            raise ValueError(f"unsupported circuit node: {node}")
        values[node] = result
        return result

    def flatten(value):
        return value if isinstance(value, list) else [value]

    output = next(n for n in module.graph.nodes if n.op == "output")
    result, reads = (flatten(get(v)) for v in output.args[0])
    return dict(
        code=np.asarray(code, dtype=np.int32).reshape(-1, 5),
        terms=np.asarray(terms, dtype=np.int32).reshape(-1, 2),
        registers=np.asarray(registers, dtype=dtype),
        outputs=np.asarray(result, dtype=np.int32), reads=np.asarray(reads, dtype=np.int32),
        inputs=module.input_width,
    )


def emit_cpp(program, index=0, *, scalar="int32_t", blocks=None):
    """Emit each SSA value once; V names the bank's exact scalar type."""
    floating = scalar == "float"
    defined = {int(i[1]) for i in program["code"]}
    names = {}

    def literal(value):
        if floating:
            if np.isnan(value):
                return "std::numeric_limits<float>::quiet_NaN()"
            if np.isinf(value):
                return ("-" if value < 0 else "") + "std::numeric_limits<float>::infinity()"
            return float(value).hex() + "f"
        return str(int(value)) if value != -(2**31) else "(-2147483647-1)"

    def v(slot):
        return names.get(slot, f"v{slot}") if slot in defined else literal(program["registers"][slot])

    body = [f"void circuit_{index}(const {scalar}* x, const uint8_t* ram, {scalar}* out, int32_t* reads) {{"]
    binary = {ADD: "+", MUL: "*", EQ: "==", GT: ">", GE: ">=", LT: "<"}
    rows = program["code"].tolist()
    position = 0
    while position < len(rows):
        if blocks and position in blocks:
            end, helper, bindings, outputs = blocks[position]
            run = rows[position:end]
            # Inputs are immutable during a forward. Snapshot any input
            # slices inside this region before binding the inline fragment.
            for op, dst, a, _b, _c in run:
                if op == INPUT:
                    body.append(f"const V v{dst} = x[{a}];")
            values = [v(value) if kind == "value" else literal(value) for kind, value in bindings]
            arg = f"a{position}" if values else "nullptr"
            if values:
                body.append(f"const V {arg}[] = {{{','.join(values)}}};")
            results = [row[1] for row in run if row[0] != INPUT]
            results = [results[i] for i in outputs]
            buffer = f"b{position}"
            body.append(f"V {buffer}[{len(results)}];")
            body.append(f"{helper}({arg}, ram, {buffer}, nullptr);")
            names.update((dst, f"{buffer}[{i}]") for i, dst in enumerate(results))
            position = end
            continue
        op, dst, a, b, c = rows[position]
        position += 1
        if op == INPUT:
            expr = f"x[{a}]"
        elif op == LINEAR:
            pieces = [] if c == 0 else [literal(c)]
            for slot, coefficient in program["terms"][a:a + b].tolist():
                pieces.append(v(slot) if coefficient == 1 else f"{literal(coefficient)}*{v(slot)}")
            expr = "+".join(pieces) or "0"
        elif op in binary:
            expr = f"{v(a)} {binary[op]} {v(b)}"
        elif op == AND:
            expr = f"bool({v(a)}) && bool({v(b)})" if floating else f"{v(a)} & {v(b)}"
        elif op in (SHL, SHR):
            # Tensor shifts are defined for oversized/negative counts too.
            expr = f"ssa_{'shl' if op == SHL else 'shr'}({v(a)}, {v(b)})"
        elif op == WHERE:
            expr = f"{v(a)} ? {v(b)} : {v(c)}"
        elif op == CLAMP:
            expr = f"std::min(std::max({v(a)}, {v(b)}), {v(c)})"
        elif op == DIV:
            expr = f"std::floor({v(a)} / {v(b)})" if floating else f"ssa_div({v(a)}, {v(b)})"
        elif op == LOAD:
            expr = f"ram[{v(a)}]"
        elif op == RELU:
            expr = f"std::max({v(a)}, {'0.0f' if floating else '0'})"
        elif op == REGLU:
            product = f"({v(a)} * {v(b)})"
            expr = f"{v(a)} > 0 ? {product}" + (f" >> {c}" if c else "") + " : 0"
        else:
            raise ValueError(f"unsupported SSA operation: {op}")
        body.append(f"const V v{dst} = {expr};")
    for name in ("outputs", "reads"):
        pointer = "out" if name == "outputs" else "reads"
        body.extend(f"{pointer}[{i}] = {v(int(s))};" for i, s in enumerate(program[name]))
    body.append("}")
    return "\n".join(body), [], dict(
        width=len(program["outputs"]), reads=len(program["reads"]), arguments=1, checks_division=False,
    )


CPP_PREAMBLE = r"""
// Scalar SSA requires -fwrapv and -ffp-contract=off.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
static inline int32_t ssa_shl(int32_t a, int32_t b) {
    return uint32_t(b) < 32 ? int32_t(uint32_t(a) << b) : 0;
}
static inline int32_t ssa_shr(int32_t a, int32_t b) {
    return uint32_t(b) < 32 ? a >> b : -(a < 0);
}
static inline int32_t ssa_div(int32_t a, int32_t b) {
    if (!b) throw std::runtime_error("integer division by zero");
    int64_t x = a, y = b;
    return int32_t(x / y - ((x % y != 0) && ((x < 0) != (y < 0))));
}
"""


def extract_fragment(program, instructions, defined):
    code, terms, bindings, outputs = [], [], [], []
    local = {}

    def operand(slot):
        if slot in local:
            return local[slot]
        arg = len(code)
        code.append((INPUT, arg, len(bindings), 0, 0))
        # Give each use its own formal argument. Different alias patterns
        # and coefficient values can share a fragment; the call binds them
        # exactly and native constant propagation recovers their identities.
        bindings.append(("value", int(slot)) if slot in defined
                        else ("constant", program["registers"][slot].item()))
        return arg

    for op, dst, a, b, c in instructions:
        if op == LINEAR:
            row = [(operand(s), int(k)) for s, k in program["terms"][a:a + b]]
            a, b = len(terms), len(row)
            terms.extend(row)
        else:
            a = operand(a)
            if op not in (LOAD, RELU):
                b = operand(b)
            if op in (WHERE, CLAMP):
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


def share_fragments(programs, *, scalar="int32_t"):
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
            if op == INPUT:
                continue
            if op == LINEAR:
                operands = [s for s, _k in program["terms"][a:a + b]]
            else:
                operands = [a] if op in (LOAD, RELU) else [a, b]
                if op in (WHERE, CLAMP):
                    operands.append(c)
            for slot in operands:
                last_use.setdefault(slot, i)
        instructions = [(i, row) for i, row in enumerate(rows)
                        if row[0] != INPUT]
        lines = emit_cpp(program, scalar=scalar)[0].splitlines()[1:1 + len(program["code"])]
        for size, stride in ((8, 4), (16, 8), (32, 8), (64, 16)):
            for start in range(0, len(instructions) - size + 1, stride):
                run = instructions[start:start + size]
                key, fragment, bindings = extract_fragment(program, [row for _, row in run], defined)
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
            if op != INPUT:
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
            if op in (INPUT, LINEAR):
                continue
            row[2] = remap.get(row[2], row[2])
            if op not in (LOAD, RELU):
                row[3] = remap.get(row[3], row[3])
            if op in (WHERE, CLAMP):
                row[4] = remap.get(row[4], row[4])
        for row in terms:
            row[0] = remap.get(row[0], row[0])
        fragment = original | dict(
            code=code, terms=terms, registers=registers,
            outputs=original["outputs"][positions], inputs=len(columns),
        )
        body = emit_cpp(fragment, scalar=scalar)[0]
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
