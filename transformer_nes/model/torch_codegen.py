"""Compile a fixed bank of transformer weights with PyTorch Inductor.

The FX graph is a scalar expansion of the checkpoint's attention and ReGLU
matrices, including integer rounding points or float32 residual arithmetic. Inductor folds immutable
coefficients, removes dead residual slots and fuses each expert. No opcode
semantics are supplied to this compiler.
"""

import ast
import math
import operator
import re

import torch
from torch.fx import Graph, GraphModule

from transformer_nes.model.torch_layout import layer_arrays

INT = torch.int32


def weight_module(expert, *, memory_size=65536, trace=False, ablation=False):
    """Expand actual matrices at their exact residual scale.

    Byte/flag masks state the native embedding's exact input domain. They
    let Inductor simplify the transformer's arithmetic on valid inputs.
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

    def op(f, *args, **kwargs):
        return g.call_function(f, args, kwargs)

    def const(v):
        if v not in constants:
            constants[v] = op(torch.full, (1,), v, dtype=dtype)
        return constants[v]

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
            state[slot] = op(operator.add, state[slot], op(operator.add, r, v))
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
            state[slot] = op(operator.add, state[slot], v)
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
    return module


def extract_kernel(source, index, *, scalar="int32_t"):
    """Parse a fused kernel's typed buffers and fixed slice aliases.

    Input pointers bind to residual/RAM; returned buffers bind to out/reads;
    intermediate buffers get local stack storage. Reject unfamiliar layouts
    rather than executing an incompletely bound generated kernel.
    """
    tree = ast.parse(source)
    kernels = {}
    for n in ast.walk(tree):
        if (
            isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "cpp_pybinding"
        ):
            assert len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
            kernels[n.targets[0].id] = (
                ast.literal_eval(n.value.args[0]),
                ast.literal_eval(n.value.args[1]),
            )
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call"]
    calls = [
        n
        for fn in functions
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in kernels
    ]
    assert calls, ("kernel calls", index)
    fn = next(fn for fn in functions if calls[0] in ast.walk(fn))
    assert all(call in ast.walk(fn) for call in calls)
    assert not any(
        isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "extern_kernels"
        for n in ast.walk(fn)
    )
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    returned = [n.id for n in ret.value.elts]
    assert len(returned) == 2
    buffers = {}
    sizes, types = {}, {}
    arguments = {}
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], (ast.List, ast.Tuple))
            and isinstance(n.value, ast.Name)
            and n.value.id == "args"
        ):
            arguments = {
                v.id: name for v, name in zip(n.targets[0].elts, ("x", "ram"), strict=True)
            }
        if (
            not isinstance(n, ast.Assign)
            or len(n.targets) != 1
            or not isinstance(n.targets[0], ast.Name)
        ):
            continue
        name = n.targets[0].id
        v = n.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name):
            if v.func.id == "empty_strided_cpu":
                size = math.prod(ast.literal_eval(v.args[0]))
                sizes[name] = size
                assert isinstance(v.args[2], ast.Attribute)
                types[name] = {"int32": "int32_t", "int64": "int64_t", "bool": "bool", "float32": "float"}[
                    v.args[2].attr
                ]
                buffers[name] = (name, 0)
            elif v.func.id == "reinterpret_tensor":
                base = v.args[0].id
                offset = ast.literal_eval(v.args[3]) if len(v.args) > 3 else 0
                assert all(s == 1 for s in ast.literal_eval(v.args[2]))
                buffers[name] = (base, offset)
        elif isinstance(v, ast.Name):
            buffers[name] = (v.id, 0)

    def resolve(name):
        delta = 0
        seen = set()
        while name in buffers:
            base, offset = buffers[name]
            delta += offset
            if base == name:
                break
            assert name not in seen
            seen.add(name)
            name = base
        return name, delta

    out, read = map(lambda n: resolve(n)[0], returned)
    assert out in sizes and read in sizes
    declarations = [
        f"    {types[name]} scratch_{name}[{max(1, size)}];"
        for name, size in sizes.items()
        if name not in (out, read)
    ]
    includes, helpers, dispatch = [], [], []
    for phase, call in enumerate(calls):
        kinds, body = kernels[call.func.id]
        match = re.search(r'extern\s+"C"\s+void\s+kernel\s*\((.*?)\)\s*\{', body, re.S)
        assert match
        params = [p.strip() for p in match.group(1).split(",")]
        assert len(params) == len(kinds) == len(call.args)
        values = []
        for _param, kind, arg in zip(params, kinds, call.args, strict=True):
            assert isinstance(arg, ast.Name), (index, ast.dump(arg))
            base, offset = resolve(arg.id)
            if base in sizes:
                value = "out" if base == out else "reads" if base == read else f"scratch_{base}"
            elif base in arguments:
                value = arguments[base]
            else:
                raise ValueError((index, kind, arg.id, base))
            if offset:
                value += f" + {offset}"
            values.append(value)
        prefix = body[: match.start()]
        includes.extend(re.findall(r"^\s*#include[^\n]+", prefix, re.M))
        assert not re.sub(r"^\s*#include[^\n]+", "", prefix, flags=re.M).strip()
        if len(calls) == 1:
            for param, value in zip(params, values, strict=True):
                declarations.append(f"    {param} = {value};")
            dispatch.append(body[match.end() :])
        else:
            name = f"phase_{index}_{phase}"
            helpers.append(
                f"static __attribute__((always_inline)) void {name}("
                + ", ".join(params)
                + ") {"
                + body[match.end() :]
            )
            dispatch.append(f"    {name}({', '.join(values)});")
    generated = "\n".join(helpers) + (
        f"\nvoid circuit_{index}(const {scalar}* x, const uint8_t* ram, {scalar}* out, int32_t* reads) {{\n"
        + "\n".join(declarations)
        + "\n"
        + "\n".join(dispatch)
        + ("\n}" if len(calls) > 1 else "")
    )
    return generated, includes, dict(
        width=sizes[out], reads=sizes[read], arguments=len(calls),
        checks_division="inductor_cpu_integer_div_error_flag" in generated,
    )


INDUCTOR_CONFIG = {
    "cpp.threads": 1,
    "cpp.vec_isa_ok": False,
    "memory_planning": False,
    "fx_graph_cache": False,
    # Materialize shared scalar expressions before fusion duplicates their
    # text across consumers. Clang can still inline the resulting phases.
    "realize_cpu_opcount_threshold": 5,
    "realize_cpu_acc_reads_threshold": 2,
    "realize_opusers_threshold": 2,
    "max_fusion_size": 16,
}


def bank_source(experts, adapter, *, memory_size=65536, trace=False, ablation=False):
    """Fuse the actual layer weights, retaining Inductor's arithmetic bodies.

    Inductor normally wraps each output slice in a Tensor. The native ABI
    already provides contiguous residual/read buffers, so extract_kernel
    binds those same slice pointers directly and allocates scratch on stack.
    Unsupported compiler layouts fail here, before a hook can be installed.
    """
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.utils import get_code

    bodies, metadata, includes = [], [], set()
    dtypes = {expert.get("dtype", "int32") for expert in experts}
    if len(dtypes) != 1 or not dtypes <= {"int32", "float32"}:
        raise ValueError("a weight bank requires one numeric type")
    scalar = "float" if dtypes == {"float32"} else "int32_t"
    compiled = {}
    for index, expert in enumerate(experts):
        module = weight_module(expert, memory_size=memory_size, trace=trace, ablation=ablation)
        nodes = {node: i for i, node in enumerate(module.graph.nodes)}

        def freeze(value, nodes=nodes):
            if isinstance(value, torch.fx.Node):
                return ("node", nodes[value])
            if isinstance(value, (tuple, list)):
                return tuple(freeze(v) for v in value)
            if isinstance(value, dict):
                return tuple((k, freeze(v)) for k, v in value.items())
            if isinstance(value, slice):
                return ("slice", value.start, value.stop, value.step)
            return value

        key = (
            module.input_width,
            tuple((n.op, n.target, freeze(n.args), freeze(n.kwargs)) for n in module.graph.nodes),
        )
        if key in compiled:
            metadata.append(compiled[key])
            continue
        example = (
            torch.zeros(module.input_width, dtype=module.dtype),
            torch.zeros(memory_size, dtype=torch.uint8),
        )
        with torch._inductor.config.patch(INDUCTOR_CONFIG):
            wrappers = [
                code
                for code in get_code(
                    lambda module=module, example=example: compile_fx(module.eval(), list(example))
                )
                if code.strip()
            ]
        if len(wrappers) != 1:
            raise RuntimeError(f"expert {index}: expected one fused Inductor wrapper")
        try:
            body, headers, meta = extract_kernel(wrappers[0], index, scalar=scalar)
        except (AssertionError, ValueError, AttributeError, StopIteration) as error:
            raise RuntimeError(f"expert {index}: unsupported Inductor buffer layout") from error
        bodies.append(body)
        meta["circuit"] = index
        meta["input_width"] = module.input_width
        compiled[key] = meta
        metadata.append(meta)
        includes.update(header.strip() for header in headers)
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
    return "\n".join(sorted(includes)) + "\n" + "\n".join(bodies) + "\n" + hook
