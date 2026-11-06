"""Canonical scalar SSA for weight-derived transformer arithmetic.

No machine or opcode semantics enter this representation. Every instruction
comes from the matrix expansion's arithmetic, memory access or ABI readout.
"""

from __future__ import annotations

import operator

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


def from_weights(expert, **kwargs):
    from transformer_nes.model.torch_codegen import weight_module

    return encode(weight_module(expert, **kwargs))


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

