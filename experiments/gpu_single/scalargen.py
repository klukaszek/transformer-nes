"""Structural lowering: whole routed forwards as scalar Metal.

A v14+ routed model (no register rows, fused/constant reads only) is an
exact closed-form program over the machine state and RAM: every residual
slot is a dyadic-linear expression, every FFN unit a ReGLU of two such,
every read a guarded ram[] load.  This module emits that program as
straight-line scalar expressions -- literal weight evaluation, identical
numbers -- executed by lane 0 for the WHOLE forward: leaves read stg[]
through the kernel's embed map (non-embed leaves are exact zeros and
their terms drop), the readout lands in outk directly, and the state/
halt head sums return by reference.  No residual buffer, no stream
walks, no embed or readout waves, no per-layer barriers.  Liveness runs
backward from the readout and head slots, so dead width never emits.
Dispatch stays a threadgroup-uniform switch; everything not lowered
(and the union) runs the interpreter.  Replay passes rerun the body
verbatim -- inputs are unchanged until commit, so the stores rewrite
identical values.
"""

from __future__ import annotations

# The kernel's branch-free STATE embed map (ESRC/ESH/EMSK): value i of
# the 31-slot state_slots table is (stg[src] >> sh) & msk, src -1 = 1.0.
ESRC = (0, 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 7, 8, 9, 9, 10, 11, 12, -1, 1, 1, 1, 1, 1, 1, 1, 1)
ESH = (0, 8, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
       0, 0, 0, 0, 8, 0, 0, 0, 0, 7, 6, 5, 4, 3, 2, 1, 0)
EMSK = (0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 1, 1, 1, 1, 1, 1, 1, 1,
        0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0,
        1, 1, 1, 1, 1, 1, 1, 1)


def f32(v: float) -> str:
    s = f"{v:.9g}"
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


class Fn:
    """Emission state for one lowered model."""

    def __init__(self, dm):
        self.dm = dm
        self.lines: list[str] = []
        self.n = 0
        self.ops = 0

    def emit(self, s):
        self.lines.append("    " + s)

    def temp(self, expr):
        name = f"t{self.n}"
        self.n += 1
        self.emit(f"const float {name} = {expr};")
        return name

    def sum_expr(self, base, terms):
        """base ('0.0f' or an expr) plus coefficient-weighted sources;
        exact-zero sources contribute nothing and drop."""
        parts = [] if base == "0.0f" else [base]
        for coeff, src in terms:
            if src == "0.0f":
                continue
            self.ops += 1
            if coeff == 1.0:
                parts.append(f"+ {src}")
            elif coeff == -1.0:
                parts.append(f"- {src}")
            else:
                parts.append(f"+ {f32(coeff)} * {src}")
        if not parts:
            return "0.0f"
        expr = " ".join(parts)
        if expr.startswith("+ "):
            expr = expr[2:]
        return expr


def layer_maps(lay, dm):
    a_res, a_acc, g_acc, f_rows = {}, {}, {}, {}
    for dst, cols, vals in lay["a"]:
        terms = list(zip(cols, vals, strict=True))
        (a_res if dst < dm else a_acc).setdefault(
            dst if dst < dm else dst - dm, []).extend(terms)
    for dst, cols, vals in lay["g"]:
        assert dst >= dm
        g_acc.setdefault(dst - dm, []).extend(zip(cols, vals, strict=True))
    for dst, cols, vals in lay["f"]:
        assert dst < dm
        f_rows.setdefault(dst, []).extend(zip(cols, vals, strict=True))
    # ho position -> (mem entry index, offset within its served values)
    ho_of = {}
    for i, ((_base, pos, nv, _qconst), _q) in enumerate(lay["mem"]):
        for off in range(max(nv, 2)):
            ho_of[pos + off] = (i, off)
    return a_res, a_acc, g_acc, f_rows, ho_of


def liveness(cap):
    """Backward closure from readout + head slots; per layer returns the
    live f dsts, attn slots, acc ids, and (mem entry, offsets) uses."""
    dm = cap["dm"]
    live = (set(cap["readout"])
            | set(cap["shead"][0]) | set(cap["hhead"][0]))
    plans = []
    for lay in reversed(cap["layers"]):
        a_res, a_acc, g_acc, f_rows, ho_of = layer_maps(lay, dm)
        f_live = {d for d in live if d in f_rows}
        xa_need = set(live)
        acc_need = set()
        for d in f_live:
            for c, _v in f_rows[d]:
                if c < dm:
                    xa_need.add(c)
                else:
                    acc_need.update((2 * (c - dm), 2 * (c - dm) + 1))
        for i in acc_need:
            for c, _v in g_acc.get(i, ()):
                xa_need.add(c)
        pre, ho_need = set(), set()
        for i in acc_need:
            for c, _v in a_acc.get(i, ()):
                (pre if c < dm else ho_need).add(c if c < dm else c - dm)
        for s in xa_need:
            if s in a_res:
                pre.add(s)
                for c, _v in a_res[s]:
                    (pre if c < dm else ho_need).add(
                        c if c < dm else c - dm)
            else:
                pre.add(s)
        mem_use = {}
        for pos in ho_need:
            if pos in ho_of:
                i, off = ho_of[pos]
                mem_use.setdefault(i, set()).add(off)
        for i in mem_use:
            qcols, _qvals = lay["mem"][i][1]
            pre.update(int(c) for c in qcols)
        plans.append((f_live, xa_need, acc_need, mem_use))
        live = pre
    plans.reverse()
    return plans


def emit_model(cap):
    dm = cap["dm"]
    fn = Fn(dm)
    plans = liveness(cap)
    cur: dict[int, str] = {}
    embed_idx = {s: i for i, s in enumerate(cap["state_slots"])}
    assert len(cap["state_slots"]) == 31

    def x_of(s):
        e = cur.get(s)
        if e is not None:
            return e
        if s == cap["one_slot"]:
            return "1.0f"
        i = embed_idx.get(s)
        if i is None:
            return "0.0f"
        src, sh, msk = ESRC[i], ESH[i], EMSK[i]
        if src < 0:
            return "1.0f"
        if sh:
            return f"(float)((stg[{src}] >> {sh}) & {msk})"
        return f"(float)(stg[{src}] & {msk})"

    for lay, (f_live, xa_need, acc_need, mem_use) in zip(
            cap["layers"], plans, strict=True):
        a_res, a_acc, g_acc, f_rows, _ho_of = layer_maps(lay, dm)
        fn.emit(f"// L{lay['li']}")

        # Lookups: guarded RAM reads; served bits only where consumed.
        ho: dict[int, str] = {}
        for i in sorted(mem_use):
            (base, pos, nv, qconst), (qcols, qvals) = lay["mem"][i]
            n = fn.n
            fn.n += 1
            if qconst >= 0:
                fn.emit(f"const int q{n} = {qconst};")
            else:
                q = fn.sum_expr("0.0f", [
                    (float(v), x_of(int(c)))
                    for c, v in zip(qcols, qvals, strict=True)])
                fn.emit(f"const int q{n} = (int)rint({q});")
            bound = 65536 if base < 0 else 4096
            addr = f"q{n}" if base < 0 else f"{base} + q{n}"
            fn.emit(f"float h{n} = 0.0f, ha{n} = 0.0f;")
            fn.emit(f"if (q{n} >= 0 && q{n} < {bound}) {{")
            fn.emit(f"    const int a{n} = {addr};")
            fn.emit(f"    if (wr[a{n}] != 0.0f) "
                    f"{{ h{n} = ram[a{n}]; ha{n} = (float)a{n}; }}")
            fn.emit("}")
            ho[pos] = f"h{n}"
            ho[pos + 1] = f"ha{n}"
            offs = mem_use[i]
            if any(off >= 2 for off in offs):
                fn.emit(f"const int b{n} = (int)h{n};")
                for off in sorted(offs):
                    if off >= 2 and off < nv:
                        ho[pos + off] = (
                            f"(float)((b{n} >> {7 - (off - 2)}) & 1)")

        def a_src(c, ho=ho):
            return x_of(c) if c < dm else ho[c - dm]

        def val(base, terms):
            e = fn.sum_expr(base, terms)
            return e if e == base or e == "0.0f" else fn.temp(e)

        # Attention adds -> post-attn residual view (xa).
        xa: dict[int, str] = {}
        for s in sorted(xa_need):
            if s in a_res:
                xa[s] = val(x_of(s),
                            [(float(v), a_src(c)) for c, v in a_res[s]])

        def xa_of(s, xa=xa):
            return xa.get(s, x_of(s))

        # Acc slots (merged rows read pre-attn, gate rows read post-attn).
        acc: dict[int, str] = {}
        for i in sorted(acc_need):
            if i in a_acc:
                acc[i] = val("0.0f",
                             [(float(v), a_src(c)) for c, v in a_acc[i]])
            elif i in g_acc:
                acc[i] = val("0.0f",
                             [(float(v), xa_of(c)) for c, v in g_acc[i]])

        reglu_memo: dict[int, str] = {}

        def reglu(p, acc=acc, memo=reglu_memo):
            if p not in memo:
                g = acc.get(2 * p, "0.0f")
                v = acc.get(2 * p + 1, "0.0f")
                if g == "0.0f" or v == "0.0f":
                    memo[p] = "0.0f"
                else:
                    fn.ops += 1
                    memo[p] = fn.temp(f"({g} > 0.0f ? {g} * {v} : 0.0f)")
            return memo[p]

        # FFN adds on top of the copied residual.
        newx = {}
        for d in sorted(f_live):
            terms = [(float(v), xa_of(c) if c < dm else reglu(c - dm))
                     for c, v in f_rows[d]]
            newx[d] = val(xa_of(d), terms)
        cur.update(xa)
        cur.update(newx)

    stores = 0
    for i, slot in enumerate(cap["readout"]):
        fn.emit(f"outk[{i}] = rint({x_of(slot)});")
        stores += 1
    for name, (cols, vals) in (("ss", cap["shead"]), ("hs", cap["hhead"])):
        fn.emit(f"{name} = " + fn.sum_expr("0.0f", [
            (float(v), x_of(int(c)))
            for c, v in zip(cols, vals, strict=True)]) + ";")
    return fn, stores


def emit_lowered(captures, threads, params):
    """captures: [(desc_index, cap)] hottest-first.  Returns
    (metal_source, stats); models with register rows fall back."""
    del threads, params  # scalar bodies are lane-0 code; no lane layout
    models = []
    skipped = 0
    for mdesc, cap in captures:
        if any(lay["nreg"] for lay in cap["layers"]):
            skipped += 1
            continue
        models.append((mdesc, *emit_model(cap)))

    if not models:
        return "", {"lowered": 0, "low_skipped": skipped}

    src = [
        "// Generated by scalargen.py: lowered window stacks (module doc).\n",
        "#include <metal_stdlib>\n",
        "using namespace metal;\n",
        f"#define LOWERED_MODELS {len(models)}\n",
    ]
    args = ("threadgroup float* outk, threadgroup const int* stg,\n"
            "        device float* ram, device float* wr,\n"
            "        thread float& ss, thread float& hs")
    for k, (_mdesc, fn, _stores) in enumerate(models):
        src.append(f"inline void lowered_m{k}({args}) {{\n")
        src.append("\n".join(fn.lines) + "\n}\n")
    src.append("inline int lowered_model_of(int e) {\n    switch (e) {\n")
    for k, (mdesc, _fn, _stores) in enumerate(models):
        src.append(f"    case {mdesc}: return {k};\n")
    src.append("    }\n    return -1;\n}\n")
    src.append(f"inline void lowered_stack(int m, {args}) {{\n"
               "    switch (m) {\n")
    for k in range(len(models)):
        src.append(
            f"    case {k}: lowered_m{k}(outk, stg, ram, wr, ss, hs);"
            " break;\n")
    src.append("    }\n}\n")

    stats = {
        "lowered": len(models),
        "low_skipped": skipped,
        "low_ops": sum(fn.ops for _d, fn, _s in models),
        "low_temps": sum(fn.n for _d, fn, _s in models),
        "low_stores": sum(s for _d, _fn, s in models),
    }
    return "".join(src), stats
