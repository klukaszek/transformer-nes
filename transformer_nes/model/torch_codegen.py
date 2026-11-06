"""Compile fixed transformer weights through a compact scalar SSA.

The FX graph is a scalar expansion of the checkpoint's attention and ReGLU
matrices, including integer rounding points or float32 residual arithmetic. The scalar IR folds immutable
coefficients, removes dead residual slots and shares each surviving value. No opcode
semantics are supplied to this compiler.
"""

import logging
import operator
import re
import time

import torch
from torch.fx import Graph, GraphModule

from transformer_nes.model import kernel_cache, outline, scalar_ir
from transformer_nes.model.torch_layout import layer_arrays

INT = torch.int32
logger = logging.getLogger(__name__)


def weight_module(expert, *, memory_size=65536, trace=False, ablation=False):
    """Expand actual matrices at their exact residual scale.

    Byte/flag masks state the native embedding's exact input domain. They
    let the compiler simplify the transformer's arithmetic on valid inputs.
    Only declared ABI readouts escape; other residual slots are dead.
    """
    model, spec = expert["model"], expert.get("step_spec")
    dtype = torch.float32 if expert.get("dtype") == "float32" else INT
    floating = dtype == torch.float32
    if floating and spec is not None:
        raise ValueError("float32 circuits require a graph token input contract")
    f_bits = expert.get("fraction_bits", 1)
    scale = 1 << f_bits
    D = model.tok.weight.shape[1]
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
    for name, slot in spec["inputs"].items() if spec else ():
        if name == "is_state":
            state[slot] = one
        else:
            bits = name.startswith("a_bit") or name in ("nmi", "irq", "halted")
            state[slot] = op(
                operator.and_, op(operator.getitem, x, slice(slot, slot + 1)), 2 if bits else 510
            )
    for slot, _mask in spec["p_flags"] if spec else ():
        state[slot] = op(operator.and_, op(operator.getitem, x, slice(slot, slot + 1)), 2)
    if spec is None:
        for field, slot in enumerate(expert["input_slots"]):
            if slot >= 0:
                value = op(operator.getitem, x, slice(field, field + 1))
                state[slot] = op(operator.mul, value, scale) if f_bits else value
    state[model.one_slot] = one

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
    for li in range(len(model.attn)):
        a = layer_arrays(model, li, floating=floating)
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
            if spec is not None:
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
            control = D if spec else len(expert["input_slots"])
            selected = op(operator.getitem, x, slice(control, control + 1))
            row = op(operator.getitem, x, slice(control + 1, control + 2))
            forced = op(operator.getitem, x, slice(control + 2, control + 3))
            if spec is None:
                forced = op(operator.mul, forced, scale) if f_bits else forced
            control_scale = 2 if spec else 1
            enabled = op(operator.eq, row, control_scale * (li + 1))
            state = [
                op(
                    torch.where,
                    op(operator.and_, enabled, op(operator.eq, selected, control_scale * i)),
                    forced,
                    v,
                )
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
    module.input_width = (D if spec else len(expert["input_slots"])) + (3 if ablation else 0)
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


def bank_source(experts, adapter, *, memory_size=65536, trace=False, ablation=False):
    """Lower weight-derived arithmetic once per value to compact native SSA.

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
        module = weight_module(expert, memory_size=memory_size, trace=trace, ablation=ablation)
        program = scalar_ir.encode(module)
        key = (program["inputs"], tuple(
            (name, program[name].dtype.str, program[name].shape, program[name].tobytes())
            for name in ("code", "terms", "registers", "outputs", "reads")
        ))
        digest = kernel_cache.fingerprint(
            key, {}, memory_size=module.memory_extent,
            scalar=scalar, extractor=scalar_ir.emit_cpp,
        )
        if digest in compiled:
            metadata.append(compiled[digest])
            continue

        body, _headers, meta, hit = kernel_cache.get_kernel(
            digest, lambda program=program: scalar_ir.emit_cpp(program, scalar=scalar),
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
    helpers, plans = outline.share(programs, scalar=scalar)
    shared_bodies = list(bodies)
    for i, plan in enumerate(plans):
        if plan:
            shared_bodies[i] = scalar_ir.emit_cpp(programs[i], indices[i], scalar=scalar, blocks=plan)[0]
    # The planner estimates call costs. Verify the complete bank too: longer
    # references to fragment results can outweigh a small local text saving.
    if len(helpers) + 1 + sum(map(len, shared_bodies)) < sum(map(len, bodies)):
        bodies = shared_bodies
    else:
        helpers = ""
    logger.info("shared %d compiled fragments", helpers.count("always_inline"))
    entries = ",\n".join(
        f"{{circuit_{m['circuit']}, {m['width']}, {m['reads']}, {str(m['checks_division']).lower()}}}"
        for m in metadata
    )
    from pathlib import Path

    runtime = Path(__file__).with_name("native").joinpath("torch_bank.h").read_text()
    hook = runtime + "\n" + adapter
    for name, value in {
        "@ENTRIES@": entries,
        "@SCALAR@": scalar,
        "@MAX_WIDTH@": str(max(meta["width"] for meta in metadata)),
        "@MAX_AUX@": str(max(1, max(meta["reads"] for meta in metadata))),
        "@INPUT_WIDTHS@": ", ".join(str(meta["input_width"]) for meta in metadata),
    }.items():
        hook = hook.replace(name, value)
    shared = helpers + "\n" if helpers else ""
    return scalar_ir.CPP_PREAMBLE + f"using V = {scalar};\n" + shared + "\n".join(bodies) + "\n" + hook
