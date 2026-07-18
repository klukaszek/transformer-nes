"""Standalone PyTorch runner for the exported 6502 transformer.

Deliberately imports NOTHING from transformer_nes: the artifact
(weights.safetensors + machine.json) plus the semantics below are the
whole system.  Execution is genuine transformer inference in float64 --
sparse matmuls of the analytic weights, hardmax attention with the
LATEST tie-break over an append-only token log, ReGLU FFN, residual
stream, no normalization -- driven step-by-step as a 6502 machine.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

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


class TorchMachine:
    """The exported weights as a stepped 6502 machine."""

    def __init__(self, artifact_dir: str):
        from safetensors.numpy import load_file

        with open(os.path.join(artifact_dir, "machine.json")) as f:
            self.meta = json.load(f)
        t = load_file(os.path.join(artifact_dir, "weights.safetensors"))
        m = self.meta
        self.d = m["d_model"]
        self.n_layers = m["n_layers"]
        self.layer_heads = m["layer_heads"]
        self.layer_ffn = m["layer_ffn"]
        self.slots = m["input_slots"]
        self.readout = m["readout_slots"]

        def csr(name, shape):
            # Storage is f32 (exact by the artifact's precision contract,
            # meta["format"]); this reference runner executes in f64.
            return torch.sparse_csr_tensor(
                torch.from_numpy(t[f"{name}.indptr"].astype(np.int64)),
                torch.from_numpy(t[f"{name}.indices"].astype(np.int64)),
                torch.from_numpy(t[f"{name}.data"].astype(np.float64)),
                size=shape,
            )

        d = self.d
        # Layers are ragged: layer li carries only its live heads/units.
        self.layers = [
            {
                "in_proj": csr(f"L{li}.in_proj", (6 * h, d)),
                "out_proj": csr(f"L{li}.out_proj", (d, 2 * h)),
                "ff_in": csr(f"L{li}.ff_in", (2 * f, d)) if f else None,
                "ff_out": csr(f"L{li}.ff_out", (d, f)) if f else None,
            }
            for li, (h, f) in enumerate(
                zip(self.layer_heads, self.layer_ffn, strict=True))
        ]
        head = torch.from_numpy(t["head"].astype(np.float64))
        self.state_head = head[m["state_row"]]
        self.halt_head = head[m["halt_row"]]
        self.reset()

    def reset(self):
        self.pos = 0
        # Per-layer append-only K/V logs (the attention state).
        self.k_log = [[] for _ in range(self.n_layers)]
        self.v_log = [[] for _ in range(self.n_layers)]

    # ── One genuine transformer position ─────────────────────────────
    def forward(self, values: dict[str, float]) -> torch.Tensor:
        x = torch.zeros(self.d, dtype=torch.float64)
        x[self.meta["one_slot"]] = 1.0
        for name, v in values.items():
            x[self.slots[name]] = float(v)
        self.pos += 1

        for li, lay in enumerate(self.layers):
            h2 = 2 * self.layer_heads[li]
            qkv = lay["in_proj"] @ x
            q, k, v = qkv[:h2], qkv[h2:2 * h2], qkv[2 * h2:]
            self.k_log[li].append(k)
            self.v_log[li].append(v)
            keys = torch.stack(self.k_log[li])    # (T, 2H): kx/ky interleaved
            vals = torch.stack(self.v_log[li])
            # Hardmax attention, every head at once: score each token by
            # qx*kx + qy*ky; exact ties go to the LATEST token (flip +
            # argmax picks the first of the reversed log).
            scores = keys[:, 0::2] * q[0::2] + keys[:, 1::2] * q[1::2]  # (T, H)
            latest = scores.shape[0] - 1 - torch.argmax(scores.flip(0), dim=0)
            o = torch.empty(h2, dtype=torch.float64)
            o[0::2] = vals[latest, torch.arange(h2 // 2) * 2]
            o[1::2] = vals[latest, torch.arange(h2 // 2) * 2 + 1]
            x = x + lay["out_proj"] @ o

            if lay["ff_in"] is not None:
                z = lay["ff_in"] @ x
                f = self.layer_ffn[li]
                gate, val = z[:f], z[f:]
                act = torch.where(gate > 0, gate * val, torch.zeros(()))
                x = x + lay["ff_out"] @ act
        return x

    # ── 6502 machine protocol ────────────────────────────────────────
    def boot_image(self, image: bytes, *, start_pc: int):
        self.reset()
        self.forward({
            "start_pc_lo": start_pc & 0xFF,
            "start_pc_hi": (start_pc >> 8) & 0xFF,
            "is_start": 1,
        })
        for addr, byte in enumerate(image):
            if byte:
                self.insert_write(addr, byte)
        self.state = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": 0x24, "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0,
            "halted": 0, "nmi": 0,
        }
        self.halted = False

    def insert_write(self, addr: int, value: int):
        self.forward({
            "byte_value": value & 0xFF, "is_memory_write": 1,
            "addr_lo": addr & 0xFF, "addr_hi": (addr >> 8) & 0xFF,
        })

    def step(self) -> dict:
        """One 6502 instruction: STATE-token forward, readout, writes."""
        values = {"is_state": 1}
        for name, get in STATE_FIELDS:
            values[name] = get(self.state)
        h = self.forward(values)

        def rd(name, mask):
            # Python round() is half-to-even; np.rint matches it.
            return int(np.rint(float(h[self.readout[name]]))) & mask

        writes = []
        for store, addr, value in (
            ("is_store", "write_addr_out", "write_value_out"),
            ("is_store2", "write_addr2_out", "write_value2_out"),
            ("is_store3", "write_addr3_out", "write_value3_out"),
        ):
            if rd(store, 0xFF):
                writes.append((rd(addr, 0xFFFF), rd(value, 0xFF)))

        row = {
            "pc": rd("current_pc", 0xFFFF), "a": rd("new_a", 0xFF),
            "x": rd("new_x", 0xFF), "y": rd("new_y", 0xFF),
            "s": rd("new_s", 0xFF), "p": rd("new_p", 0xFF),
            "ir": rd("fetched_opcode", 0xFF),
            "oplo": rd("fetched_operand_lo", 0xFF),
            "ophi": rd("fetched_operand_hi", 0xFF),
            "ea": rd("ea", 0xFFFF), "data": rd("data_out", 0xFF),
            "writes": tuple(writes),
        }
        for addr, value in writes:
            self.insert_write(addr, value)
        self.state.update(
            pc=rd("next_pc", 0xFFFF), a=row["a"], x=row["x"], y=row["y"],
            s=row["s"], p=row["p"], ir=row["ir"], oplo=row["oplo"],
            ophi=row["ophi"], ea=row["ea"], data=row["data"],
            halted=rd("new_halted", 0xFF),
        )
        halt = float(self.halt_head @ h)
        state = float(self.state_head @ h)
        if self.state["halted"] or halt > state:
            self.halted = True
        return row
