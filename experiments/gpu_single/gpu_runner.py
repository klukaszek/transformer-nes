"""Single-machine GPU runner for the 6502 transformer.

One 6502, one step per forward, on torch/MPS (or CPU): the experiment is
LATENCY -- can a GPU drive the emulator at a respectable steps/s -- not
batch throughput.  The architecture is the generated engine's O(1)
attention forms lifted to tensors, licensed by model/classify.py exactly
as engine_gen is:

- passthrough heads compose into a per-layer (I + attn_direct) matrix
  (self-attention == identity, so their attention IS a linear map);
- kind-2 FFN units compose into (I + ffn_direct);
- memory heads are a RAM gather: (ram[addr], addr) on an exact written
  hit, else (0, 0) == the empty-group result (the exact-address mask
  downstream zeroes everything else);
- constant-query heads are running-argmax registers (replace-on->= in
  insertion order IS the LATEST tie-break).

WRITE tokens never run forwards (scatter into the RAM tensors, exactly
like the engine's insert_write); STATE tokens are the per-step forwards.

Usage:
  uv run python experiments/gpu_single/gpu_runner.py --verify [--device mps]
  uv run python experiments/gpu_single/gpu_runner.py --bench 2000 --device mps [--check]
  uv run python experiments/gpu_single/gpu_runner.py --baseline 2000
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from transformer_nes.cpu6502.backends.graph import RUNTIME_READ_DIMS, Cpu6502Runtime
from transformer_nes.graph.core import InputDimension, ProgramGraph
from transformer_nes.model.classify import (
    integer_score_terms,
    memory_query,
    register_query,
)
from transformer_nes.model.weights import build_model
from transformer_nes.scheduler.milp import asap_schedule

STATE_FIELDS = (
    ("in_pc_lo", lambda s: s["pc"] & 0xFF),
    ("in_pc_hi", lambda s: (s["pc"] >> 8) & 0xFF),
    ("in_a", lambda s: s["a"]), ("in_x", lambda s: s["x"]),
    ("in_y", lambda s: s["y"]), ("in_s", lambda s: s["s"]),
    ("in_p_c", lambda s: (s["p"] >> 0) & 1), ("in_p_z", lambda s: (s["p"] >> 1) & 1),
    ("in_p_i", lambda s: (s["p"] >> 2) & 1), ("in_p_d", lambda s: (s["p"] >> 3) & 1),
    ("in_p_b", lambda s: (s["p"] >> 4) & 1), ("in_p_u", lambda s: (s["p"] >> 5) & 1),
    ("in_p_v", lambda s: (s["p"] >> 6) & 1), ("in_p_n", lambda s: (s["p"] >> 7) & 1),
    ("in_ir", lambda s: s["ir"]), ("in_oplo", lambda s: s["oplo"]),
    ("in_ophi", lambda s: s["ophi"]),
    ("in_ea_lo", lambda s: s["ea"] & 0xFF),
    ("in_ea_hi", lambda s: (s["ea"] >> 8) & 0xFF),
    ("in_data", lambda s: s["data"]), ("in_halted", lambda s: s["halted"]),
    ("in_nmi", lambda s: s["nmi"]),
)


def build_union(bcd: bool = False):
    # wide_bits: the depth-lean bit-extraction ladder -- GPU forwards pay
    # for depth, not unit count (the C++ engine keeps the serial form).
    rt = Cpu6502Runtime(bcd=bcd, wide_bits=True)
    dim_by_id = {d.id: d for d in rt.all_dims}
    output_tokens = dict(rt.output_tokens)
    readout = 0
    for name in RUNTIME_READ_DIMS:
        readout = readout + 1e-6 * dim_by_id[rt.dim_id_by_name[name]]
    output_tokens["READOUT"] = readout
    pg = ProgramGraph(rt.input_tokens, output_tokens)
    sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
    model, _tokens, tok_to_idx, info = build_model(program_graph=pg, schedule=sched)
    input_slots = {
        d.name: int(info.slot_of[d])
        for d in pg.all_dims
        if isinstance(d, InputDimension) and d in info.slot_of
    }
    readout_slots = {
        name: int(info.slot_of[dim_by_id[rt.dim_id_by_name[name]]])
        for name in RUNTIME_READ_DIMS
    }
    return model, info, tok_to_idx, input_slots, readout_slots, rt, pg


class GpuMachine:
    """The union model as a stepped 6502 on a torch device."""

    def __init__(self, device: str = "cpu", bcd: bool = False):
        model, info, tok_to_idx, input_slots, readout_slots, rt, _pg = build_union(bcd)
        self.device = torch.device(device)
        self.d = info.d_model
        self.n_layers = info.n_layers
        self.slots = input_slots
        self.readout = readout_slots
        self.one_slot = model.one_slot
        dm, dev = self.d, self.device

        def t(a):
            return torch.tensor(np.ascontiguousarray(a, dtype=np.float32),
                                device=dev)

        reg_of: dict[tuple, int] = {}
        self.layers = []
        eye = np.eye(dm)
        for li in range(self.n_layers):
            ip = model.attn[li].in_proj_weight
            op = model.attn[li].out_proj.weight
            fi = model.ff_in[li].weight
            fo = model.ff_out[li].weight
            ffn = fo.shape[1]
            kinds = model.head_kinds[li]
            nh = model.attn[li].num_heads
            lookup = [h for h in range(nh) if kinds[h] == 1]
            pt = [h for h in range(nh) if kinds[h] == 2]

            pt_cols = [2 * h + c for h in pt for c in range(2)]
            ad = op[:, pt_cols] @ ip[[2 * dm + c for c in pt_cols]]
            lu_cols = [2 * h + c for h in lookup for c in range(2)]
            op_rest = op.copy()
            op_rest[:, pt_cols] = 0.0
            op_rest[:, lu_cols] = 0.0
            if op_rest.any():
                raise AssertionError(f"L{li}: out_proj column at unused head")
            opk = op[:, lu_cols]

            fk = model.ffn_kinds[li]
            pt_j = [j for j in range(ffn) if fk[j] == 2]
            fd = fo[:, pt_j] @ fi[[ffn + j for j in pt_j]]
            live = [j for j in range(ffn) if fk[j] == 1 and fo[:, j].any()]

            # Classify lookup heads with the engine's licenses.
            mem_rows, mem_base, mem_pos = [], [], []
            reg_rows, reg_gate, reg_v, reg_idx, reg_pos = [], [], [], [], []
            for pos, h in enumerate(lookup):
                qx = [(c, i) for i, c in enumerate(ip[2 * h]) if c]
                qy = [(c, i) for i, c in enumerate(ip[2 * h + 1]) if c]
                if model.head_memory_groups[li][h] >= 0:
                    terms = memory_query(qx, qy, one_id=self.one_slot)
                    if terms is None:
                        raise NotImplementedError(f"L{li} h{h}: hull memory head")
                    row = np.zeros(dm)
                    for c, i in terms:
                        row[i] += c
                    mem_rows.append(row)
                    mem_base.append(model.head_memory_groups[li][h] * 4096)
                    mem_pos.append(pos)
                    continue
                rq = register_query(qx, qy, one_id=self.one_slot,
                                    latest=bool(model.head_tiebreak[li][h]))
                if rq is None:
                    raise NotImplementedError(f"L{li} h{h}: hull head")
                kx = [(c, i) for i, c in enumerate(ip[dm + 2 * h]) if c]
                ky = [(c, i) for i, c in enumerate(ip[dm + 2 * h + 1]) if c]
                sc = integer_score_terms(rq[0], rq[1], kx, ky)
                if sc is None:
                    raise NotImplementedError(f"L{li} h{h}: unlicensed score")
                score_terms, gate_ids = sc
                srow = np.zeros(dm)
                for c, i in score_terms:
                    srow[i] += c
                grow = np.zeros(dm)
                grow[gate_ids] = 1.0
                reg_rows.append(srow)
                reg_gate.append(grow)
                reg_v.append(ip[2 * dm + 2 * h:2 * dm + 2 * h + 2])
                key = (model.head_lookup_ids[li][h], model.head_pairs[li][h])
                if key in reg_of:
                    raise AssertionError(f"L{li} h{h}: register key repeated")
                reg_of[key] = len(reg_of)
                reg_idx.append(reg_of[key])
                reg_pos.append(pos)

            lay = {
                "adp": t(eye + ad),
                "fdp": t(eye + fd),
                "n_lookup": len(lookup),
            }
            if lookup:
                lay["opk"] = t(opk)
            if live:
                lay["fi"] = t(np.vstack([fi[live], fi[[ffn + j for j in live]]]))
                lay["fo"] = t(fo[:, live])
                lay["n_live"] = len(live)
            if mem_rows:
                lay["mem_q"] = t(np.vstack(mem_rows))
                lay["mem_base"] = torch.tensor(mem_base, device=dev,
                                               dtype=torch.long)
                lay["mem_pos"] = torch.tensor(
                    [p for m in mem_pos for p in (2 * m, 2 * m + 1)],
                    device=dev, dtype=torch.long)
            if reg_rows:
                lay["reg_s"] = t(np.vstack(reg_rows))
                lay["reg_g"] = t(np.vstack(reg_gate))
                lay["reg_v"] = t(np.vstack(reg_v))  # (2r, D): v0/v1 interleaved
                lay["reg_idx"] = torch.tensor(reg_idx, device=dev,
                                              dtype=torch.long)
                lay["reg_pos"] = torch.tensor(
                    [p for m in reg_pos for p in (2 * m, 2 * m + 1)],
                    device=dev, dtype=torch.long)
            self.layers.append(lay)

        self.n_regs = len(reg_of)
        head = model.head.weight
        self.state_head = t(head[tok_to_idx["STATE"]])
        self.halt_head = t(head[tok_to_idx["HALT"]])
        self.readout_idx = torch.tensor(
            [self.readout[n] for n in RUNTIME_READ_DIMS],
            device=dev, dtype=torch.long)
        self.readout_names = list(RUNTIME_READ_DIMS)
        self.state_slots = torch.tensor(
            [self.slots[n] for n, _ in STATE_FIELDS] + [self.slots["is_state"]],
            device=dev, dtype=torch.long)
        self.reset()

    def reset(self):
        dev = self.device
        self.ram_val = torch.zeros(65536, device=dev)
        self.ram_wr = torch.zeros(65536, device=dev)
        self.regs_any = torch.zeros(self.n_regs, device=dev, dtype=torch.bool)
        self.regs_s = torch.full((self.n_regs,), -3.0e38, device=dev)
        self.regs_v = torch.zeros(2 * self.n_regs, device=dev)

    # ── One genuine step forward over the layer stack ────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lay in self.layers:
            if lay["n_lookup"]:
                ho = torch.zeros(2 * lay["n_lookup"], device=self.device)
                if "mem_q" in lay:
                    qi = lay["mem_q"] @ x
                    addr = (qi.long() + lay["mem_base"]).clamp(0, 65535)
                    hit = ((qi >= 0) & (qi < 4096)).float() * self.ram_wr[addr]
                    pair = torch.stack(
                        [hit * self.ram_val[addr], hit * addr.float()], 1)
                    ho[lay["mem_pos"]] = pair.reshape(-1)
                if "reg_s" in lay:
                    s = lay["reg_s"] @ x
                    live = (lay["reg_g"] @ x.abs()) == 0
                    v = (lay["reg_v"] @ x).reshape(-1, 2)
                    idx = lay["reg_idx"]
                    upd = live & (~self.regs_any[idx] | (s >= self.regs_s[idx]))
                    self.regs_any[idx] |= upd
                    self.regs_s[idx] = torch.where(upd, s, self.regs_s[idx])
                    vi = torch.stack([2 * idx, 2 * idx + 1], 1).reshape(-1)
                    self.regs_v[vi] = torch.where(
                        upd.repeat_interleave(2), v.reshape(-1), self.regs_v[vi])
                    out = self.regs_any[idx].float().repeat_interleave(2) \
                        * self.regs_v[vi]
                    ho[lay["reg_pos"]] = out
                x = lay["adp"] @ x + lay["opk"] @ ho
            else:
                x = lay["adp"] @ x
            if "fi" in lay:
                z = lay["fi"] @ x
                n = lay["n_live"]
                g, v = z[:n], z[n:]
                act = torch.where(g > 0, g * v, torch.zeros((), device=x.device))
                x = lay["fdp"] @ x + lay["fo"] @ act
            else:
                x = lay["fdp"] @ x
        return x

    def forward_values(self, values: dict[str, float]) -> torch.Tensor:
        x = torch.zeros(self.d, device=self.device)
        x[self.one_slot] = 1.0
        for name, v in values.items():
            x[self.slots[name]] = float(v)
        return self.forward(x)

    # ── 6502 machine protocol (mirrors torch_runner) ─────────────────
    def boot_image(self, image: bytes, *, start_pc: int):
        self.reset()
        img = np.frombuffer(bytes(image), dtype=np.uint8)
        nz = np.nonzero(img)[0]
        idx = torch.tensor(nz, device=self.device, dtype=torch.long)
        self.ram_val[idx] = torch.tensor(img[nz], device=self.device,
                                         dtype=torch.float32)
        self.ram_wr[idx] = 1.0
        self.forward_values({
            "start_pc_lo": start_pc & 0xFF,
            "start_pc_hi": (start_pc >> 8) & 0xFF,
            "is_start": 1,
        })
        self.state = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": 0x24, "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0,
            "halted": 0, "nmi": 0,
        }
        self.halted = False
        self._vals_buf = torch.zeros(len(STATE_FIELDS) + 1, device=self.device)

    def insert_write(self, addr: int, value: int):
        # The engine's insert_write: a WRITE token is a memory-structure
        # scatter, never a forward (all other heads clear-gate WRITEs).
        self.ram_val[addr & 0xFFFF] = float(value & 0xFF)
        self.ram_wr[addr & 0xFFFF] = 1.0

    def step(self) -> dict:
        s = self.state
        vals = [float(get(s)) for _, get in STATE_FIELDS] + [1.0]
        x = torch.zeros(self.d, device=self.device)
        x[self.one_slot] = 1.0
        self._vals_buf.copy_(torch.tensor(vals))
        x[self.state_slots] = self._vals_buf
        h = self.forward(x)

        out = np.rint(h[self.readout_idx].cpu().numpy()).astype(np.int64)
        halt = float(self.halt_head @ h)
        state_sc = float(self.state_head @ h)
        rd = dict(zip(self.readout_names, out.tolist(), strict=True))

        writes = []
        for store, addr, value in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            if rd[store] & 0xFF:
                writes.append((rd[addr] & 0xFFFF, rd[value] & 0xFF))

        row = {
            "pc": rd["current_pc"] & 0xFFFF, "a": rd["new_a"] & 0xFF,
            "x": rd["new_x"] & 0xFF, "y": rd["new_y"] & 0xFF,
            "s": rd["new_s"] & 0xFF, "p": rd["new_p"] & 0xFF,
            "ir": rd["fetched_opcode"] & 0xFF,
            "oplo": rd["fetched_operand_lo"] & 0xFF,
            "ophi": rd["fetched_operand_hi"] & 0xFF,
            "ea": rd["ea"] & 0xFFFF, "data": rd["data_out"] & 0xFF,
            "writes": tuple(writes),
        }
        for addr, value in writes:
            self.insert_write(addr, value)
        self.state.update(
            pc=rd["next_pc"] & 0xFFFF, a=row["a"], x=row["x"], y=row["y"],
            s=row["s"], p=row["p"], ir=row["ir"], oplo=row["oplo"],
            ophi=row["ophi"], ea=row["ea"], data=row["data"],
            halted=rd["new_halted"] & 0xFF,
        )
        if self.state["halted"] or halt > state_sc:
            self.halted = True
        return row


# ── harness ──────────────────────────────────────────────────────────

# LDX #$FF; DEX; BNE -3; JMP $0600 -- an endless mixed-op loop.
BENCH_ORG = 0x0600
BENCH_CODE = bytes([0xA2, 0xFF, 0xCA, 0xD0, 0xFD, 0x4C, 0x00, 0x06])


def bench_image() -> bytes:
    image = bytearray(65536)
    image[BENCH_ORG:BENCH_ORG + len(BENCH_CODE)] = BENCH_CODE
    return bytes(image)


def ref_machine(image: bytes, start_pc: int, *, bcd: bool = False):
    from transformer_nes.cpu6502.backends import reference
    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=bcd)
    ref.pc = start_pc
    return ref


def check_row(row, r, i):
    s = r.state
    got = (row["pc"], row["a"], row["x"], row["y"], row["s"], row["p"],
           row["ir"], row["oplo"], row["ophi"], row["ea"], row["data"],
           set(row["writes"]))
    want = (s.pc, s.a, s.x, s.y, s.s, s.p, s.ir, s.operand_lo, s.operand_hi,
            s.ea, s.data, {(w.addr, w.value) for w in r.writes})
    if got != want:
        raise AssertionError(f"step {i}: got {got} want {want}")


def verify(device: str):
    from transformer_nes.cpu6502.backends.weights.runner import (
        VERIFY_START,
        verify_program,
    )
    m = GpuMachine(device)
    image = verify_program()
    m.boot_image(image, start_pc=VERIFY_START)
    ref = ref_machine(image, VERIFY_START)
    steps = 0
    for i in range(40):
        check_row(m.step(), ref.step(), i)
        steps += 1
        if m.halted:
            break
    print(f"[{device}] verify_program: {steps} steps lockstep-exact")


def bench(device: str, n: int, check: bool, compile_: bool = False):
    m = GpuMachine(device)
    if compile_:
        m.forward = torch.compile(m.forward)
    m.boot_image(bench_image(), start_pc=BENCH_ORG)
    ref = ref_machine(bench_image(), BENCH_ORG) if check else None
    for i in range(50):
        row = m.step()
        if ref is not None:
            check_row(row, ref.step(), i)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        row = m.step()
        if ref is not None:
            check_row(row, ref.step(), 50 + i)
    dt = time.perf_counter() - t0
    tag = " (lockstep-checked)" if check else ""
    print(f"[{device}] {n} steps in {dt:.2f}s = {n / dt:,.0f} steps/s "
          f"({dt / n * 1e3:.2f} ms/step){tag}")


def baseline(n: int):
    from transformer_nes.cpu6502.backends.weights.runner import (
        Cpu6502TransformerMachine,
    )
    m = Cpu6502TransformerMachine(bcd=False)
    m.load_image(bench_image(), start_pc=BENCH_ORG)
    m.run(50)
    t0 = time.perf_counter()
    m.run(n)
    dt = time.perf_counter() - t0
    print(f"[cpp engine] {n} steps in {dt:.3f}s = {n / dt:,.0f} steps/s")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--bench", type=int, default=0)
    p.add_argument("--check", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--baseline", type=int, default=0)
    args = p.parse_args()
    if args.verify:
        verify(args.device)
    if args.bench:
        bench(args.device, args.bench, args.check, args.compile)
    if args.baseline:
        baseline(args.baseline)


if __name__ == "__main__":
    main()
