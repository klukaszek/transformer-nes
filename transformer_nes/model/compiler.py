"""Compile analytic matrices into FX arithmetic and fixed native banks.

layer_arrays projects each layer into live integer or float32 matrices.
weight_module expands those matrices into FX, preserving every rounding point.
bank_source lowers and shares scalar SSA, then assembles the native adapter.
No opcode or rendering semantics are supplied to this compiler.
"""

from __future__ import annotations

import logging
import operator
import re
import time

import numpy as np
import torch
from torch.fx import Graph, GraphModule

from transformer_nes import build
from transformer_nes.model import ssa
from transformer_nes.model.weights import rescale_head_queries, split_clear_gates, to_fixed_point

LOOKUP, PASSTHROUGH = 1, 2
NO_MEMORY_GROUP = -1
READ_CODES = {"value": -1, "address": -2, **{f"bit{n}": n for n in range(8)}}


def share_queries(w_in, w_out, read_kind):
    """Share query projections and merge identical live value channels."""
    queries, rows, routes, columns, channels = {}, [], [], [], {}
    for h, kinds in enumerate(read_kind.tolist()):
        pair = w_in[2 * h : 2 * h + 2]
        key = pair.numpy().tobytes()
        if key not in queries:
            queries[key] = len(rows)
            rows.append(pair)
        for c, kind in enumerate(kinds):
            column = w_out[:, 2 * h + c]
            if kind == -3 or not column.any():
                continue
            value = (queries[key], kind)
            if value not in channels:
                channels[value] = len(columns)
                columns.append(column.to(torch.int64))
                routes.append(value)
            else:
                columns[channels[value]] += column
    projected = torch.cat([*rows, w_in[2 * len(read_kind) :]])
    values = torch.stack(columns, 1) if columns else w_out[:, :0]
    out = torch.cat([values, w_out[:, 2 * len(read_kind) :]], 1).to(torch.int64)
    if ((out < -(2**31)) | (out >= 2**31)).any():
        raise ValueError("shared value projection exceeds int32")
    return projected, out.to(w_out.dtype), torch.tensor(routes, dtype=torch.int32).reshape(-1, 2)


def layer_arrays(model, li: int, *, floating: bool = False) -> dict:
    """Layer `li` of a folded model as dense blocks over the residual slots
    it touches (construct.py stores them sparse)."""
    attn, heads = model.attn[li], model.attn[li].num_heads
    kinds, groups = model.head_kinds[li], model.head_memory_groups[li]
    if any(kinds[h] == LOOKUP and groups[h] == NO_MEMORY_GROUP for h in range(heads)):
        raise ValueError(f"layer {li}: a lookup head that reads no memory")
    # Head h's query is in-projection rows 2h, 2h + 1, its value rows
    # 4H + 2h, + 1; its output, out-projection columns 2h, 2h + 1.  No
    # head needs its keys.  In units of a half, the engine's fixed point.
    memory = [h for h in range(heads) if kinds[h] == LOOKUP and groups[h] != NO_MEMORY_GROUP]
    if floating:
        if memory:
            raise ValueError("float32 circuits require memory-free attention")
        passes = [2 * h + c for h in range(heads) if kinds[h] == PASSTHROUGH for c in range(2)]
        w_in = attn.in_proj_weight[[4 * heads + c for c in passes]]
        w_out = attn.out_proj.weight[:, passes]
        w_ff, w_down = model.ff_in[li].weight, model.ff_out[li].weight
        ins, outs = touched(w_in, 0), touched(w_out, 1)
        ff_ins, ff_outs = touched(w_ff, 0), touched(w_down, 1)
        return {
            "in_slots": ins, "out_slots": outs,
            "ff_in_slots": ff_ins, "ff_out_slots": ff_outs,
            "read_kind": np.empty((0, 2), dtype=np.int32),
            "w_in": w_in[:, ins].astype(np.float32),
            "w_out": w_out[outs].astype(np.float32),
            "w_ff": w_ff[:, ff_ins].astype(np.float32),
            "w_down": w_down[ff_outs].astype(np.float32),
        }
    if any(groups[h] >= 0 for h in memory):
        raise ValueError("Torch models require structural memory reads before scheduling")
    reads = [2 * h + c for h in memory for c in range(2)]
    passes = [2 * h + c for h in range(heads) if kinds[h] == PASSTHROUGH for c in range(2)]
    w_in = split_clear_gates(rescale_head_queries(attn.in_proj_weight, heads))[0]
    w_in = to_fixed_point(w_in[reads + [4 * heads + c for c in passes]], "w_in")
    w_out = to_fixed_point(attn.out_proj.weight[:, reads + passes], "w_out")
    w_ff = to_fixed_point(model.ff_in[li].weight, "w_ff")
    w_down = to_fixed_point(model.ff_out[li].weight, "w_down")
    if (w_in % 2).any():
        raise ValueError(f"layer {li}: half coefficients in the in-projection")
    outputs = model.head_memory_outputs[li]
    w_in, w_out, read_kind = share_queries(
        torch.from_numpy(w_in),
        torch.from_numpy(w_out),
        torch.tensor([read_codes(outputs[h]) for h in memory], dtype=torch.int32).reshape(-1, 2),
    )
    w_in, w_out, read_kind = w_in.numpy(), w_out.numpy(), read_kind.numpy()
    in_slots, out_slots = touched(w_in, 0), touched(w_out, 1)
    ff_in_slots, ff_out_slots = touched(w_ff, 0), touched(w_down, 1)
    doubled = {
        "w_in": w_in[:, in_slots],
        "w_out": w_out[out_slots],
        "w_ff": w_ff[:, ff_in_slots],
        "w_down": w_down[ff_out_slots],
    }
    halves = {f"h_{name[2:]}": W - 2 * (W >> 1) for name, W in doubled.items() if name != "w_in"}
    arrays = {
        "in_slots": in_slots,
        "out_slots": out_slots,
        "ff_in_slots": ff_in_slots,
        "ff_out_slots": ff_out_slots,
        "read_kind": read_kind,
        **{name: W >> 1 for name, W in doubled.items()},
    }
    if any(h.any() for h in halves.values()):
        arrays |= halves
    arrays = {name: np.ascontiguousarray(a, dtype=np.int32) for name, a in arrays.items()}
    return arrays


def touched(W: np.ndarray, axis: int) -> np.ndarray:
    """The columns (axis 0) or rows (axis 1) of W holding a nonzero."""
    return np.flatnonzero((W != 0).any(axis=axis)).astype(np.int32)


def read_codes(outputs) -> list[int]:
    kinds = list(outputs or ("value", "address"))
    return [READ_CODES[k] for k in kinds] + [-3] * (2 - len(kinds))


INT = torch.int32
logger = logging.getLogger(__name__)


def circuit_layers(expert) -> tuple[list[dict], int, int]:
    """(per-layer arrays, d_model, one_slot) of a circuit: its own
    ``layers`` (a model.transformer.Transformer's buffers), else cut from
    its ``model`` (a weights.WeightModel)."""
    if "layers" in expert:
        return expert["layers"], expert["d_model"], expert["one_slot"]
    model = expert["model"]
    floating = expert.get("dtype") == "float32"
    layers = [layer_arrays(model, li, floating=floating) for li in range(len(model.attn))]
    return layers, model.tok.weight.shape[1], model.one_slot


def weight_module(expert, *, memory_size=65536, trace=False, ablation=False, report_reads=False):
    """Expand actual matrices at their exact residual scale.

    Byte/flag masks state the native embedding's exact input domain. They
    let the compiler simplify the transformer's arithmetic on valid inputs.
    Only declared ABI readouts escape; other residual slots are dead.
    `report_reads` also returns every memory read's address (plus its layer
    times the memory size), for the X-ray's read counts.
    """
    dtype = torch.float32 if expert.get("dtype") == "float32" else INT
    floating = dtype == torch.float32
    f_bits = expert.get("fraction_bits", 1)
    scale = 1 << f_bits
    layers, D, one_slot = circuit_layers(expert)
    g = Graph()
    x = g.placeholder("x")
    ram = g.placeholder("memory")
    constants = {}
    constant_values = {}
    expressions = {}

    def wrap32(value):
        return (value + 2**31) % 2**32 - 2**31

    def freeze(value):
        if isinstance(value, (tuple, list)):
            return tuple(freeze(v) for v in value)
        if isinstance(value, slice):
            return (slice, value.start, value.stop, value.step)
        return value

    def const(v, numeric_type=dtype):
        key = (v, numeric_type)
        if key not in constants:
            node = g.call_function(torch.full, ((1,), v), {"dtype": numeric_type})
            constants[key] = node
            constant_values[node] = v
        return constants[key]

    def op(f, *args, **kwargs):
        # Preserve the int32 rounding/overflow boundary at every operation.
        # Simplifying while expanding the sparse matrices also preserves the
        # identity of the constant query channel, avoiding needless divides.
        # Float circuits retain their original IEEE arithmetic (including NaNs
        # and signed zero), without algebraic identity rewrites.
        if not floating:
            values = tuple(constant_values.get(v, v) if isinstance(v, torch.fx.Node) else v for v in args)
            if len(args) == 2:
                left, right = values
                if f in (
                    operator.add, operator.mul, operator.and_, operator.lshift, operator.rshift
                ) and all(isinstance(v, int) for v in values):
                    value = f(left, right)
                    return const(value, torch.bool) if isinstance(value, bool) else const(wrap32(value))
                if f is operator.add:
                    if left == 0:
                        return args[1]
                    if right == 0:
                        return args[0]
                if f is operator.mul:
                    if left == 0 or right == 0:
                        return const(0)
                    if left == 1:
                        return args[1]
                    if right == 1:
                        return args[0]
                if f in (operator.lshift, operator.rshift) and right == 0:
                    return args[0]
                if f in (operator.eq, operator.gt, operator.ge, operator.lt) and all(
                    isinstance(v, int) for v in values
                ):
                    return const(f(left, right), torch.bool)
            if f is torch.where and isinstance(values[0], (bool, int)):
                selected = args[1] if values[0] else args[2]
                return const(selected) if isinstance(selected, int) else selected
            if f is torch.clamp and len(values) == 3 and all(
                type(v) is int and -(2**31) <= v < 2**31 for v in values
            ):
                return const(min(max(values[0], values[1]), values[2]))
        key = (f, freeze(args), tuple(kwargs.items()))
        if key not in expressions:
            expressions[key] = g.call_function(f, args, kwargs)
        return expressions[key]

    zero = const(0)
    one = const(scale)
    state = [zero] * D
    for field, slot in enumerate(expert["input_slots"]):
        if slot >= 0:
            value = op(operator.getitem, x, slice(field, field + 1))
            state[slot] = op(operator.mul, value, scale) if f_bits else value
    state[one_slot] = one

    def dot(weights, values):
        result = None
        for c, v in zip(weights, values, strict=True):
            if not c:
                continue
            term = v if c == 1 else op(operator.mul, v, float(c) if floating else int(c))
            result = term if result is None else op(operator.add, result, term)
        return zero if result is None else result

    def linear(weights, values, halves=None):
        result = []
        for i, row in enumerate(weights):
            value = dot(row, values)
            if halves is not None and halves[i].any():
                value = op(operator.add, value, op(operator.rshift, dot(halves[i], values), 1))
            result.append(value)
        return result

    affine_forms = {}

    def affine(value):
        """Collect integer linear terms modulo 2**32, stopping at every
        nonlinear operation, shift and memory read. In particular, never
        move coefficients across the half-weight rounding points."""
        if not isinstance(value, torch.fx.Node):
            return {None: value}
        if value not in affine_forms:
            if value in constant_values:
                form = {None: constant_values[value]}
            elif value.op == "call_function" and value.target is operator.add:
                form = dict(affine(value.args[0]))
                for term, coefficient in affine(value.args[1]).items():
                    form[term] = form.get(term, 0) + coefficient
            elif value.op == "call_function" and value.target is operator.mul:
                left, right = value.args
                lc = constant_values.get(left, left) if isinstance(left, torch.fx.Node) else left
                rc = constant_values.get(right, right) if isinstance(right, torch.fx.Node) else right
                if isinstance(lc, int):
                    form = {term: lc * c for term, c in affine(right).items()}
                elif isinstance(rc, int):
                    form = {term: rc * c for term, c in affine(left).items()}
                else:
                    form = {value: 1}
            else:
                form = {value: 1}
            affine_forms[value] = {
                term: wrap32(c) for term, c in form.items() if wrap32(c)
            }
        return affine_forms[value]

    def accumulate(*values):
        # Slot reuse emits old + (-old + new). Cancel the cleared stream
        # before lowering chooses which intermediates to materialize.
        if floating or not expert.get("reassociate_residuals", True):
            old = values[0]
            delta = op(operator.add, values[1], values[2]) if len(values) == 3 else values[1]
            if not floating:
                memo = {}

                def strip(value):
                    # Treat the old value as an opaque atom. Only remove
                    # its explicit linear clear; preserve the grouping of
                    # every other projection, shift and nonlinear value.
                    if value is old:
                        return zero, 1
                    if not isinstance(value, torch.fx.Node):
                        return value, 0
                    if value in memo:
                        return memo[value]
                    result = value, 0
                    if value.op == "call_function" and value.target is operator.add:
                        a, na = strip(value.args[0])
                        b, nb = strip(value.args[1])
                        result = op(operator.add, a, b), na + nb
                    elif value.op == "call_function" and value.target is operator.mul:
                        a, b = value.args
                        ac, bc = constant_values.get(a, a), constant_values.get(b, b)
                        if type(ac) is int and ac in (-1, 1):
                            rest, n = strip(b)
                            result = op(operator.mul, rest, ac), n * ac
                        elif type(bc) is int and bc in (-1, 1):
                            rest, n = strip(a)
                            result = op(operator.mul, rest, bc), n * bc
                    memo[value] = result
                    return result

                rest, coefficient = strip(delta)
                if coefficient == -1:
                    return rest
            return op(operator.add, old, delta)
        form = {}
        for value in values:
            for term, c in affine(value).items():
                form[term] = form.get(term, 0) + c
        result = zero
        for term, c in form.items():
            c = wrap32(c)
            if c:
                value = const(c) if term is None else op(operator.mul, term, c)
                result = op(operator.add, result, value)
        return result

    reads, snapshots = [], []
    if trace:
        snapshots.append(op(torch.cat, state))
    for li, a in enumerate(layers):
        nvalues = len(a["read_kind"])
        qrows = len(a["w_in"]) - a["w_out"].shape[1] + nvalues
        projected = linear(a["w_in"], [state[s] for s in a["in_slots"]])
        queries = []
        for i in range(qrows // 2):
            qx, qy = projected[2 * i : 2 * i + 2]
            if qy is one:
                # The actual projection is the unchanged constant channel.
                # Avoid introducing a divide and its runtime error tracking.
                address = op(operator.rshift, qx, f_bits) if f_bits else qx
            else:
                address = op(
                    torch.div, qx, op(torch.where, op(operator.eq, qy, 0), 1, qy), rounding_mode="floor"
                )
            valid = op(
                operator.and_, op(operator.ge, address, 0), op(operator.lt, address, memory_size)
            )
            byte = g.call_method(
                "to", (op(operator.getitem, ram, op(torch.clamp, address, 0, memory_size - 1)), INT)
            )
            byte = op(operator.mul, byte, valid)
            masked = op(operator.mul, address, valid)
            queries.append((byte, masked))
            # Deduplicate within a layer while preserving reads across layers.
            if report_reads:
                reads.append(
                    op(torch.where, valid, op(operator.add, address, li * memory_size), -1)
                )
        served = []
        for query, kind in a["read_kind"]:
            if kind not in (-2, -1, *range(8)):
                raise ValueError(f"unsupported memory output kind: {kind}")
            byte, address = queries[query]
            value = (
                byte
                if kind == -1
                else address
                if kind == -2
                else op(operator.and_, op(operator.rshift, byte, int(kind)), 1)
            )
            served.append(op(operator.lshift, value, f_bits) if f_bits else value)
        halves = a.get("h_out")
        hread, hvalue = (
            (None, None) if halves is None else (halves[:, :nvalues], halves[:, nvalues:])
        )
        read = linear(a["w_out"][:, :nvalues], served, hread)
        value = linear(a["w_out"][:, nvalues:], projected[qrows:], hvalue)
        for slot, r, v in zip(a["out_slots"], read, value, strict=True):
            state[slot] = accumulate(state[slot], r, v)
        projected = linear(a["w_ff"], [state[s] for s in a["ff_in_slots"]], a.get("h_ff"))
        nff = a["w_down"].shape[1]
        acts = []
        for gate, value in zip(projected[:nff], projected[nff:], strict=True):
            if floating:
                act = op(operator.mul, op(torch.relu, gate), value)
            else:
                product = op(operator.mul, gate, value)
                act = op(
                    torch.where,
                    op(operator.gt, gate, 0),
                    op(operator.rshift, product, f_bits) if f_bits else product,
                    0,
                )
            acts.append(act)
        down = linear(a["w_down"], acts, a.get("h_down"))
        for slot, v in zip(a["ff_out_slots"], down, strict=True):
            state[slot] = accumulate(state[slot], v)
        if ablation:
            control = len(expert["input_slots"])
            selected = op(operator.getitem, x, slice(control, control + 1))
            row = op(operator.getitem, x, slice(control + 1, control + 2))
            forced = op(operator.getitem, x, slice(control + 2, control + 3))
            forced = op(operator.mul, forced, scale) if f_bits else forced
            enabled = op(operator.eq, row, li + 1)
            state = [
                op(torch.where, op(operator.and_, enabled, op(operator.eq, selected, i)), forced, v)
                for i, v in enumerate(state)
            ]
        if trace:
            snapshots.append(op(torch.cat, state))
    readout_slots = tuple(expert["output_slots"]) if not trace else None
    values = [state[s] for s in readout_slots or ()]
    out = op(torch.cat, snapshots if trace else values)
    addresses = op(torch.cat, reads) if reads else op(torch.empty, (0,), dtype=INT)
    g.output((out, addresses))
    g.eliminate_dead_code()
    module = GraphModule({}, g)
    module.input_width = len(expert["input_slots"]) + (3 if ablation else 0)
    module.readout_slots = readout_slots
    module.f_bits = f_bits
    module.dtype = dtype
    module.canonical_affine = expert.get("reassociate_residuals", True)
    # Fixed-address reads (e.g. all 64 OAM Y bytes) have the same raw-buffer
    # circuit across RAM layouts. Prove the required prefix after DCE; any
    # dynamic index conservatively retains the complete memory extent.
    module.memory_extent = 0
    for user in ram.users:
        index = (
            constant_values.get(user.args[1])
            if user.op == "call_function" and user.target is operator.getitem
            and isinstance(user.args[1], torch.fx.Node) else None
        )
        if type(index) is not int or not 0 <= index < memory_size:
            module.memory_extent = memory_size
            break
        module.memory_extent = max(module.memory_extent, index + 1)
    return module


def bank_kernels(experts, *, memory_size=65536, trace=False, ablation=False, reads=False) -> tuple[str, list[dict], str]:
    """Lower weight-derived arithmetic once per value to compact native SSA.

    Returns (kernels, metadata per circuit, scalar type). The kernels need
    ssa.CPP_PREAMBLE before them; ``circuit_<m["circuit"]>`` runs circuit i,
    whose metadata also gives its output width, reported reads and input width.

    The IR cancels integer residual clears, shares affine forms and treats
    half shifts, ReGLUs and memory loads as explicit arithmetic boundaries.
    Float32 circuits retain the original order of every arithmetic operation.
    """
    bodies, metadata, programs, indices = [], [], [], []
    dtypes = {expert.get("dtype", "int32") for expert in experts}
    if len(dtypes) != 1 or not dtypes <= {"int32", "float32"}:
        raise ValueError("a weight bank requires one numeric type")
    scalar = "float" if dtypes == {"float32"} else "int32_t"
    compiled = {}
    cache_hits = 0
    start = time.perf_counter()
    for index, expert in enumerate(experts):
        module = weight_module(
            expert, memory_size=memory_size, trace=trace, ablation=ablation, report_reads=reads
        )
        program = ssa.encode(module)
        key = (program["inputs"], tuple(
            (name, program[name].dtype.str, program[name].shape, program[name].tobytes())
            for name in ("code", "terms", "registers", "outputs", "reads")
        ))
        digest = build.kernel_fingerprint(
            key, {}, memory_size=module.memory_extent,
            scalar=scalar, extractor=ssa.emit_cpp,
        )
        if digest in compiled:
            metadata.append(compiled[digest])
            continue

        body, _headers, meta, hit = build.cached_kernel(
            digest, lambda program=program: ssa.emit_cpp(program, scalar=scalar),
        )
        cache_hits += hit
        bodies.append(re.sub(r"\bcircuit_0\b", f"circuit_{index}", body))
        programs.append(program)
        indices.append(index)
        meta["circuit"] = index
        meta["input_width"] = module.input_width
        compiled[digest] = meta
        metadata.append(meta)
    logger.info(
        "weight circuits: %d unique, %d reused, %d lowered in %.1fs",
        len(compiled), cache_hits, len(compiled) - cache_hits, time.perf_counter() - start,
    )
    helpers, plans = ssa.share_fragments(programs, scalar=scalar)
    shared_bodies = list(bodies)
    for i, plan in enumerate(plans):
        if plan:
            shared_bodies[i] = ssa.emit_cpp(programs[i], indices[i], scalar=scalar, blocks=plan)[0]
    # The planner estimates call costs. Verify the complete bank too: longer
    # references to fragment results can outweigh a small local text saving.
    if len(helpers) + 1 + sum(map(len, shared_bodies)) < sum(map(len, bodies)):
        bodies = shared_bodies
    else:
        helpers = ""
    logger.info("shared %d compiled fragments", helpers.count("always_inline"))
    shared = helpers + "\n" if helpers else ""
    return f"using V = {scalar};\n" + shared + "\n".join(bodies), metadata, scalar
