"""Weight the MoE probe results by SMB's real opcode mix.

Boots SMB on the reference backend, presses Start, then holds right with
periodic jumps so Mario actually plays World 1-1; histograms the executed
opcodes over the gameplay window and weights moe_probe_results.json by it.

Run after moe_probe.py:  uv run python experiments/moe_probe/smb_weighting.py
Needs the user-supplied ROM at transformer_vm/nes/examples/smb.nes.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from transformer_vm.nes.console import Console
from transformer_vm.cpu6502.spec import OPCODES

HERE = Path(__file__).resolve().parent
SMB = HERE.parents[1] / "transformer_vm" / "nes" / "examples" / "smb.nes"

console = Console(str(SMB), backend="reference")
cpu = console.bus.core.cpu
bus = console.bus

hist: Counter[int] = Counter()
record = [False]
orig_step = bus.step


def counting_step():
    if record[0] and not getattr(cpu, "nmi_pending", False):
        hist[bus.ram[cpu.pc & 0xFFFF]] += 1
    return orig_step()


bus.step = counting_step

t0 = time.time()
for f in range(1000):
    if f == 60:
        console.controller.set_buttons(start=True)
    if f == 70:
        console.controller.set_buttons(start=False)
    if f == 250:
        console.controller.set_buttons(right=True)
        record[0] = True
    if f >= 250:  # hop periodically so Mario clears pipes/goombas
        console.controller.set_buttons(a=((f - 250) % 90 < 25))
    console.run_frame()

total = sum(hist.values())
page = bus.ram[0x006D]  # player page: >0 means Mario crossed screen pages
print(
    f"gameplay window: {total} instructions over 750 frames "
    f"({total / 750:.0f}/frame); player page ${page:02X} (moved: {page > 0})"
)
print("\ntop 15 opcodes:")
for b, n in hist.most_common(15):
    op = OPCODES.get(b)
    name = f"{op.mnemonic} {op.mode}" if op else f"??{b:02X}"
    print(f"  {b:02X} {name:16} {n:>8} ({100 * n / total:.1f}%)")

res = json.load(open(HERE / "moe_probe_results.json"))
base, per_op, per_fam = res["union"], res["per_opcode"], res["per_family"]
fam_of = {op.byte: op.operation for op in OPCODES.values()}
wt_o = wd_o = wt_f = wd_f = wl_o = 0.0
for b, n in hist.items():
    if b not in OPCODES:
        continue
    w = n / total
    ro, rf = per_op[f"{b:02X}"], per_fam[fam_of[b]]
    wt_o += w * ro["terms"]
    wd_o += w * ro["depth"]
    wl_o += w * ro["lookups"]
    wt_f += w * rf["terms"]
    wd_f += w * rf["depth"]

print(f"\nunion baseline:  depth {base['depth']}, terms {base['terms']}, lookups {base['lookups']}")
print(
    f"gameplay per-op: depth {wd_o:.1f} ({wd_o / base['depth']:.3f}x), "
    f"terms {wt_o:.0f} ({wt_o / base['terms']:.3f}x), lookups {wl_o:.1f}"
)
print(
    f"gameplay family: depth {wd_f:.1f} ({wd_f / base['depth']:.3f}x), "
    f"terms {wt_f:.0f} ({wt_f / base['terms']:.3f}x)"
)

json.dump(
    {f"{b:02X}": n for b, n in hist.most_common()},
    open(HERE / "smb_opcode_hist_gameplay.json", "w"),
    indent=1,
)
print(f"[{time.time() - t0:.0f}s]")
