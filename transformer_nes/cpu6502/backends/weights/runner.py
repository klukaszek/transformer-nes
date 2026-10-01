"""Construct and run the 6502 analytic transformer weights backend.

The graph is scheduled, converted to weights, and executed by generated C++.
Python owns construction, cache orchestration, and the public machine API;
the native engine owns instruction execution and attention state.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import logging
import os
import pickle
import shutil
import time

import numpy as np

from transformer_nes.cpu6502.backends import reference
from transformer_nes.cpu6502.backends.weights.experts import (
    WindowLicenseError,
    build_core_experts,
    build_sequence_expert,
    decode_ops,
    head_row,
    step_spec_base,
    window_layout,
)
from transformer_nes.cpu6502.backends.weights.windows import (
    block_code,
    match_shape_windows,
    static_window_limit,
)
from transformer_nes.cpu6502.construction import RUNTIME_READ_DIMS, Cpu6502Graph
from transformer_nes.cpu6502.spec import OPCODES, written_addresses
from transformer_nes.cpu6502.trace import Cpu6502State, MemWrite, Trace, TraceRow
from transformer_nes.files import write_atomic
from transformer_nes.model.engine_gen import (
    CACHE_DIR,
    GeneratedWeightsEngine,
    SequenceModule,
    engine_config_key,
    seqmod_config_key,
)
from transformer_nes.model.schedule import asap_schedule
from transformer_nes.model.sources import source_digest
from transformer_nes.model.weights import build_model

logger = logging.getLogger(__name__)


def window_key(window) -> tuple:
    """A window as (head, code, repeat, forward, taken): store forwarding
    and branch predictions default off."""
    head, code, repeat, forward, taken = (*window, *(False, ())[len(window) - 3:])
    return (int(head) & 0xFFFF, bytes(code), int(repeat), bool(forward),
            tuple(bool(t) for t in taken))


# Folds (minutes for a ROM's windows) are cached per expert, keyed by the
# code that makes them: the expert construction and the CPU graph, with
# their imports (model/sources.py).
FOLD_SOURCES = ("transformer_nes.cpu6502.backends.weights.experts", "transformer_nes.cpu6502.construction")


@functools.cache
def fold_dir() -> str:
    """This code's folds; any other code's are deleted on first use."""
    root = os.path.join(CACHE_DIR, "folds")
    digest = source_digest(*FOLD_SOURCES)[:16]
    if os.path.isdir(root):
        for stale in set(os.listdir(root)) - {digest}:
            shutil.rmtree(os.path.join(root, stale), ignore_errors=True)
    os.makedirs(os.path.join(root, digest), exist_ok=True)
    return os.path.join(root, digest)


def fold_cached(name: str) -> bool:
    return os.path.exists(os.path.join(fold_dir(), name + ".pkl"))


def cached_fold(name: str, build):
    """`build()`, pickled under `name` in this code's folds."""
    path = os.path.join(fold_dir(), name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    value = build()
    write_atomic(path, pickle.dumps(value))
    return value


def window_fold_name(window: tuple, bcd: bool) -> str:
    return "window_" + hashlib.sha256(repr((window_key(window), bcd)).encode()).hexdigest()[:24]


def window_folds(pg, windows, bcd: bool, *, build: bool = True) -> list[dict] | None:
    """The windows' experts, in order (a window the fold's licenses refuse
    only loses its fast path).  Each fold is cached on its own as soon as
    it is made, so an interrupted build keeps what it did; with `build`
    False, None unless all are cached."""
    names = [window_fold_name(w, bcd) for w in windows]
    if not build and not all(map(fold_cached, names)):
        return None
    experts = (cached_fold(name, lambda w=w: fold_window(pg, w))
               for name, w in zip(names, windows, strict=True))
    return [e for e in experts if e is not None]


def fold_window(pg, window: tuple) -> dict | None:
    """A window's expert, or None where the fold's licenses refuse it."""
    try:
        return build_sequence_expert(pg, *window)
    except RuntimeError as e:
        logger.info("sequence window skipped: %s", e)
        return None


class Cpu6502TransformerMachine:
    """The 6502 as actual transformer weights, stepped position by position."""

    def __init__(self, *, bcd: bool = False, union_only: bool = False,
                 mirror_cpu_ram: bool = False, forward_hook: bool = False):
        # forward_hook: build the engine whose expert steps can hand their
        # layers to an outside runtime (torch_forward.py); a separate build,
        # since the hook's branch slows the native forwards.
        self.forward_hook = bool(forward_hook)
        self.bcd = bool(bcd)
        self.union_only = bool(union_only)
        self.mirror_cpu_ram = bool(mirror_cpu_ram)
        # Build the graph description only. The weights backend has its own
        # native execution path and does not execute the graph in Python.
        self.graph = Cpu6502Graph(bcd=bcd, wide_bits=True)
        dim_by_id = {d.id: d for d in self.graph.all_dims}
        self.dim_by_name = {
            name: dim_by_id[dim_id] for name, dim_id in self.graph.dim_id_by_name.items()
        }

        pg = self.graph.program_graph()
        self.pg = pg  # kept for the expert folds (graph/fold.py)
        build_start = time.perf_counter()
        sched = asap_schedule(pg.input_tokens, pg.output_tokens, program_graph=pg)
        self.model, _tokens, tok_to_idx, info = build_model(
            program_graph=pg, schedule=sched,
        )
        self.build_seconds = time.perf_counter() - build_start
        self.n_layers = info.n_layers
        self.slot_of = info.slot_of
        self.lives = info.lives  # residual live ranges (nes/xray.py)
        self.d_model = info.d_model
        self.state_row = tok_to_idx["STATE"]
        self.halt_row = tok_to_idx["HALT"]
        self.read_slots = {
            name: self.slot_of[self.dim_by_name[name]]
            for name in RUNTIME_READ_DIMS
        }
        self.one_slot = self.slot_of[self.graph.one]

        # The whole layer stack runs as generated straight-line C++ with an
        # exact fixed-point int32 residual (model/engine_gen.py); source
        # content is part of the cache key, so cores are construction rules
        # rather than a configuration vocabulary.
        cache_key = engine_config_key(
            bcd=bcd, union_only=union_only, mirror_cpu_ram=self.mirror_cpu_ram,
            forward_hook=self.forward_hook,
        )
        engine = GeneratedWeightsEngine.from_cache(cache_key)
        if engine is not None:
            logger.info("engine cache hit (%d experts)", engine.n_experts)
        if engine is None:
            # Without experts every opcode misses dispatch and falls to
            # step_core, so the union alone executes the whole ISA.
            experts, shapes_meta = ([], []) if union_only else self.core_experts()
            engine = GeneratedWeightsEngine(
                self.model, step_spec=self.step_spec(),
                experts=experts, cache_key=cache_key,
                shapes_meta=shapes_meta, mirror_cpu_ram=self.mirror_cpu_ram,
                forward_hook=self.forward_hook,
            )
        self.engine = engine
        # The ROM's windows land in a tiny per-ROM module attached through
        # the core's function-pointer hook (no core recompile per ROM).
        self.seq_module = None
        self.seq_windows: tuple = ()
        self.seq_key: str | None = None
        self._window_experts: list[dict] | None = None
        if not self.engine.has_machine_step:
            raise RuntimeError("generated 6502 engine is missing machine_step")
        if not self.engine.supports_direct_writes:
            raise RuntimeError("generated 6502 engine is missing direct memory writes")
        self.step_out = np.empty(22, dtype=np.int64)
        # Shape matches per injected (base, bank content): a bank switch
        # re-adds its windows from this memo instead of rescanning.
        self._region_windows: dict[tuple[int, bytes], tuple] = {}

        self.pos = 0
        self.trace_len = 0
        # RAM mirror: shape-window scans and the machine-protocol paths
        # read it (the native run loop keeps its own copy).
        self.ram = bytearray(0x10000)

    # ── Transformer plumbing ────────────────────────────────────────

    def embed_array(self, values: dict) -> np.ndarray:
        """Engine input embedding: dim values at their residual slots.

        Equivalent to a vocabulary embedding row for a token type that
        enumerated exactly these values.  There is no position encoding:
        order enters only through the LATEST tie-break.
        """
        x = np.zeros(self.d_model, dtype=np.float32)
        x[self.one_slot] = 1.0
        for dim, value in values.items():
            x[self.slot_of[dim]] = float(value)
        return x

    def forward_values(self, values: dict) -> np.ndarray:
        """One genuine transformer position: embed and run the whole layer
        stack (attention + ReGLU) in C++."""
        x = self.embed_array(values)
        self.pos += 1
        self.engine.forward(x)  # in place, whole stack
        return x

    def core_experts(self) -> tuple[list[dict], list[dict]]:
        """The core engine's experts in its build order (the forward hook's
        ids 0..), and the shapes' metadata (experts.py:build_core_experts;
        folds cached)."""
        return cached_fold(f"core_bcd{int(self.bcd)}", lambda: build_core_experts(self.pg))

    def window_experts(self, *, build: bool = True) -> list[dict] | None:
        """The attached windows' experts, in module order (the forward
        hook's ids SEQ_HOOK_IDS..; window_folds)."""
        if self._window_experts is None:
            self._window_experts = window_folds(self.pg, self.seq_windows, self.bcd, build=build)
        return self._window_experts

    def windows_folded(self) -> bool:
        """Whether every attached window's fold is cached (nothing loaded)."""
        return all(fold_cached(window_fold_name(w, self.bcd)) for w in self.seq_windows)

    def attach_sequence_module(self, windows, *, build: bool = True) -> bool:
        """Compile-or-load the per-ROM sequence module and attach its window
        dispatch to the core engine; whether it attached.  A new ROM's
        window set costs one small TU (cached by digest; folding its
        windows takes longer), which `build` False leaves to someone else:
        the windows then attach only from the cache.  The 43 MB core is
        never re-emitted for it."""
        windows = tuple(map(window_key, windows))
        self.engine.detach_seq()
        self.seq_module = None
        self.seq_windows, self.seq_key, self._window_experts = (), None, None
        if not windows:
            return False
        key = seqmod_config_key(
            core_digest=self.engine.digest, bcd=self.bcd, windows=windows
        )
        mod = SequenceModule.from_cache(key, core_digest=self.engine.digest)
        if mod is None and not build:
            return False
        self.seq_windows = windows
        if mod is None:
            experts = self.window_experts()
            if not experts:
                return False
            mod = SequenceModule(
                self.model, self.step_spec(), experts,
                core_digest=self.engine.digest, cache_key=key,
            )
        else:
            logger.info("sequence module cache hit (max window %d steps)", mod.max_k)
        self.seq_module, self.seq_key = mod, key
        heads = b"".join(
            head.to_bytes(2, "little")
            for head in sorted({head for head, *_ in windows})
        )
        self.engine.attach_seq(mod.fn_ptr, mod.ctx_ptr, heads)
        return True

    def longest_window(self, pc: int, code: bytes, ops, *, forward: bool = False,
                       taken: tuple[bool, ...] = ()) -> tuple[int, bool]:
        """How many leading instructions of `code` fold into one licensed
        window (0 if fewer than two), and whether a read of the window's own
        store is what stopped it longer (`forward` folds take such reads).
        `taken` predicts the code's branches in order; a prefix keeps the
        predictions of the branches still inside it.  A search that jumps
        to a license's own cut point when it names one, and bisects when it
        does not."""
        good, bad = 1, len(ops) + 1  # prefixes known to fold / not to
        n = len(ops)
        aliasing = False
        while good + 1 < bad:
            interior = sum(op.mode == "rel" for op in ops[: n - 1])
            try:
                build_sequence_expert(
                    self.pg, pc, code[: sum(op.length for op in ops[:n])], 1, forward,
                    taken[:interior])
                good = n
            except WindowLicenseError as e:
                aliasing |= e.aliasing
                # Every longer prefix holds the step the license refused.
                bad = min(n, e.keep + 1)
                if good < e.keep < n - 1:
                    n = e.keep  # the license names its cut: try it next
                    continue
            except RuntimeError as e:
                logger.info("window @%04X skipped: %s", pc, e)
                return 0, False
            n = (good + bad) // 2
        return (good if good >= 2 else 0), aliasing

    def fold_window(self, pc: int, code: bytes, taken: tuple[bool, ...] = ()) -> tuple | None:
        """The longest licensed window from `pc` along `code`, as
        (n instructions, window tuple), store forwarding tried when a read
        of the window's own store cut it; None below two instructions."""
        ops = decode_ops(code)[: static_window_limit(code)]
        if len(ops) < 2:
            return None
        n, aliasing = self.longest_window(pc, code, ops, taken=taken)
        forward = False
        if aliasing:
            longer, _ = self.longest_window(pc, code, ops, forward=True, taken=taken)
            if longer > n:
                n, forward = longer, True
        if n < 2:
            return None
        interior = sum(op.mode == "rel" for op in ops[: n - 1])
        return n, (pc, code[: sum(op.length for op in ops[:n])], 1, forward, taken[:interior])

    def superblock(self, read, pc: int, code: bytes, runs, taken_runs, *,
                   bias: float = 0.9, least: int = 8, limit: int = 32) -> tuple | None:
        """A window from `pc` through the conditional branches ending
        `code`'s blocks, each the way the profile says it almost always
        goes (`bias` of at least `least` runs): the blocks chained along
        those predictions, then folded as one guarded window.  None unless
        the fold reaches past the first prediction."""
        taken: tuple[bool, ...] = ()
        visited = {at for at, _length in window_layout(pc, code)}
        while code and len(decode_ops(code)) < limit:
            at, length = window_layout(pc, code, taken)[-1]  # the block's last
            last = OPCODES[code[-length]]
            if last.mode != "rel" or runs[at] < least:
                break
            rate = taken_runs[at] / runs[at]
            if max(rate, 1 - rate) < bias:
                break
            way = bool(rate >= 0.5)
            offset = code[-1] - (256 if code[-1] & 0x80 else 0)
            successor = (at + 2 + (offset if way else 0)) & 0xFFFF
            rest = block_code(read, successor, limit=limit - len(decode_ops(code)))
            if not rest or successor in visited:
                break  # (a loop's back edge: loop windows take those)
            visited |= {a for a, _l in window_layout(successor, rest)}
            code, taken = code + rest, taken + (way,)
        if not taken:
            return None
        folded = self.fold_window(pc, code, taken)
        return folded[1] if folded and folded[1][4] else None

    def learn_windows(self, *, coverage: float = 0.98, limit: int = 384,
                      profile=None, memory: bytes | None = None) -> list[tuple]:
        """Per-ROM windows for the blocks this run entered most (the
        engine's profile since the last call), covering `coverage` of all
        entries in at most `limit` heads.  Each block becomes as few windows
        as the fold's licenses allow: a window is cut just before the step
        a license refuses, and the rest of the block starts the next.  A
        block a read of its own store cuts short also tries store
        forwarding.  Where a block's last window ends at a branch the
        profile finds biased, a superblock from the same head runs on
        through it (dispatch tries it first; a failed prediction declines
        to the shorter window).  Returns (head, code, 1, forward, taken)
        for attach_sequence_module.  `profile` and `memory` default to the
        engine's (a background learner passes a running engine's).
        """
        counts, runs, taken_runs = (
            self.engine.engine.take_profile() if profile is None else profile)
        read = self.read_ram if memory is None else (lambda address: memory[address & 0xFFFF])
        order = np.argsort(counts)[::-1]
        total, covered = int(counts.sum()), 0
        windows: list[tuple] = []
        seen: set[int] = set()
        for head in order[:limit].tolist():
            if not counts[head] or covered >= coverage * total:
                break
            covered += int(counts[head])
            pc = head
            while pc not in seen:
                seen.add(pc)
                code = block_code(read, pc)
                layout = window_layout(pc, code)
                folded = self.fold_window(pc, code)
                if folded:
                    windows.append(folded[1])
                n = folded[0] if folded else 1  # an instruction no window holds runs alone
                if n >= len(layout):
                    superblock = self.superblock(read, pc, code, runs, taken_runs)
                    if superblock:
                        windows.append(superblock)
                    break  # the block is covered
                pc = layout[n][0]
        logger.info("learned %d windows (%d superblocks) from %d block heads",
                    len(windows), sum(bool(w[4]) for w in windows), len(seen))
        return windows

    def populate_shape_windows(self) -> None:
        """Fill the engine's runtime window table from the current RAM image
        (called at every boot/reseed; dispatch re-verifies bytes live, so a
        stale entry is only a missed fast path)."""
        meta = getattr(self.engine, "shapes_meta", None) or []
        if not (meta and self.engine.has_run_loop):
            return
        self.engine.clear_windows()
        for sid, head in match_shape_windows(self.ram, meta):
            self.engine.add_window(sid, head)

    def step_spec(self) -> dict:
        """Slot map for the generated C++ machine_step (engine_gen.py)."""
        spec = step_spec_base(
            lambda name: self.slot_of[self.dim_by_name[name]]
        )
        spec["outputs"] = dict(self.read_slots)
        spec["halt_head"] = head_row(self.model, self.halt_row)
        spec["state_head"] = head_row(self.model, self.state_row)
        return spec

    # ── Machine protocol used by the NES bus and test harness ────────

    def boot_image(self, memory_image, *, start_pc: int, state: dict | None = None):
        # Fresh token log: clear the engine's attention state so a machine
        # instance can be reused for another program.
        self.engine.reset()
        self.pos = 0
        self.trace_len = 0
        ram = bytearray(0x10000)
        ram[: len(memory_image)] = bytes(memory_image)
        self.ram = ram
        if self.mirror_cpu_ram:
            for addr in range(0x0800, 0x2000):
                self.ram[addr] = self.ram[addr & 0x07FF]
        graph = self.graph
        self.forward_values({
            graph.start_pc_lo: start_pc & 0xFF,
            graph.start_pc_hi: (start_pc >> 8) & 0xFF,
            graph.is_start: 1,
        })
        # Native seed: one call for the image instead of one per byte.
        self.pos += int(self.engine.seed_image(bytes(ram)))
        self.mstate = {
            "pc": start_pc & 0xFFFF, "a": 0, "x": 0, "y": 0, "s": 0xFD,
            "p": reference.FLAG_U | reference.FLAG_I,
            "ir": 0, "oplo": 0, "ophi": 0, "ea": 0, "data": 0, "halted": 0,
        }
        if state is not None:
            self.mstate.update({k: int(state[k]) for k in self.mstate if k in state})
        self.nmi_pending = False
        self.irq_pending = False
        self.machine_halted = False
        self.populate_shape_windows()

    def set_nmi(self):
        """The interrupt sequence itself then runs inside the weights."""
        self.nmi_pending = True
        # In the native run loop the pending-NMI flag lives in the engine
        # (the per-step path passes it as an argument instead).
        if self.engine.has_run_loop:
            self.engine.set_nmi()

    def set_irq(self):
        """Delivered at the next step boundary with the I flag clear."""
        self.irq_pending = True
        if self.engine.has_run_loop:
            self.engine.set_irq()

    def clear_irq(self):
        self.irq_pending = False
        if self.engine.has_run_loop:
            self.engine.clear_irq()

    def inject_write(self, addr: int, value: int):
        self.insert_write(addr & 0xFFFF, value & 0xFF)

    def inject_region(self, base: int, data: bytes):
        """Bulk injection (mapper bank switches): one WRITE token per byte
        engine-side, so the position mirror advances by len(data)."""
        base &= 0xFFFF
        data = bytes(data)
        for offset, value in enumerate(data):
            self._set_ram(base + offset, value)
        self.engine.inject_region(base, data)
        self.pos += len(data)

    def map_region(self, base: int, data: bytes):
        """Replace a mapped cartridge region without creating WRITE tokens
        (the engine's map_write stamps state in place; no sequence advance)."""
        self.engine.map_region(base, data)
        for sid, head in self.region_windows(base, data):
            self.engine.add_window(sid, head)

    def region_windows(self, base: int, data: bytes) -> tuple:
        """Shape windows of newly banked-in code, so it dispatches at once
        instead of waiting out JIT warmup; memoized per bank content."""
        meta = getattr(self.engine, "shapes_meta", None) or []
        if not (meta and self.engine.has_run_loop):
            return ()
        key = (base, bytes(data))
        hits = self._region_windows.get(key)
        if hits is None:
            hits = self._region_windows[key] = tuple(match_shape_windows(data, meta, base=base))
        return hits

    def machine_abi(self) -> tuple[int, int, int]:
        """(version, engine, table) for native callers (run_protocol.py:
        MachineAbi).  Callers driving it bypass this object's position
        and state mirrors; this object must outlive them."""
        return self.engine.machine_abi()

    def insert_write(self, addr: int, value: int):
        addr &= 0xFFFF
        value &= 0xFF
        self._set_ram(addr, value)
        self.engine.insert_write(addr, value, self.pos)
        self.pos += 1

    def _set_ram(self, addr: int, value: int) -> None:
        for target in written_addresses(addr, self.mirror_cpu_ram):
            self.ram[target] = value & 0xFF

    def machine_step(self) -> TraceRow:
        """One instruction entirely in generated C++ (embed, layers,
        readout, direct WRITE inserts, halt logits); Python only unpacks
        the 22-slot result and keeps the mirrors in sync."""
        st = self.mstate
        nmi = 1 if self.nmi_pending else 0
        self.nmi_pending = False
        irq = 1 if reference.irq_deliverable(st["p"], self.irq_pending, nmi) else 0
        if irq:
            self.irq_pending = False
        self.engine.machine_step(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"],
            st["halted"], nmi, irq, self.step_out,
        )
        o = self.step_out.tolist()  # one C call instead of 22 scalar reads
        n_writes = o[14]
        writes = []
        for i in range(n_writes):
            addr, value = o[15 + 2 * i], o[16 + 2 * i]
            self.ram[addr] = value
            writes.append(MemWrite(self.trace_len, addr, value))
        self.pos += 1 + n_writes

        halted = o[11] != 0
        row = TraceRow(
            step=self.trace_len,
            state=Cpu6502State(
                pc=o[0], a=o[1], x=o[2], y=o[3], s=o[4], p=o[5], ir=o[6],
                operand_lo=o[7], operand_hi=o[8],
                ea=o[9], data=o[10], halted=halted,
            ),
            writes=tuple(writes),
        )
        self.trace_len += 1
        st.update(
            pc=o[12], a=o[1], x=o[2], y=o[3], s=o[4], p=o[5], ir=o[6],
            oplo=o[7], ophi=o[8], ea=o[9], data=o[10],
            halted=1 if halted else 0,
        )
        if halted or o[13]:
            self.machine_halted = True
        return row

    def run_image(self, memory_image, *, start_pc: int, max_steps: int) -> Trace:
        self.boot_image(list(memory_image), start_pc=start_pc)
        trace = Trace()
        for _ in range(max_steps):
            trace.append(self.machine_step())
            if self.machine_halted:
                break
        return trace

    # ── C++ run loop ────────────────────────────────────────────────
    # Native streaming API used by Klaus, nestest, and the NES console.

    @property
    def supports_run_loop(self) -> bool:
        return self.engine.has_run_loop

    def seed_run_state(self) -> None:
        st = self.mstate
        self.engine.set_run_state(
            st["pc"], st["a"], st["x"], st["y"], st["s"], st["p"],
            st["ir"], st["oplo"], st["ophi"], st["ea"], st["data"], st["halted"],
        )

    def pull_run_state(self) -> None:
        """Copy the engine's architectural state back into the Python mirror
        (called after a native run so the machine protocol sees it)."""
        s = self.engine.get_run_state()
        self.mstate.update(
            pc=int(s["pc"]), a=int(s["a"]), x=int(s["x"]), y=int(s["y"]),
            s=int(s["s"]), p=int(s["p"]), ir=int(s["ir"]), oplo=int(s["oplo"]),
            ophi=int(s["ophi"]), ea=int(s["ea"]), data=int(s["data"]),
            halted=1 if s["halted"] else 0,
        )
        self.machine_halted = bool(s["halted"])
        self.pos = int(s["pos"])

    def run_image_arrays(self, memory_image, *, start_pc: int, max_steps: int,
                         stop_on_trap: bool = False) -> dict:
        """Boot and run up to max_steps entirely in the C++ run loop.

        Returns the run-protocol result dict: per-step
        trace arrays (pc/a/x/y/s/p/ir/oplo/ophi/ea/data/halted), write arrays
        (w_row 0-based within this call, w_addr, w_value), and the run's
        "reason"/"cycles".  Materializes every step's arrays, so it is for
        bounded runs (nestest, a Klaus prefix); use iter_run_arrays to stream
        the full suite.
        """
        if not self.supports_run_loop:
            raise RuntimeError("this engine has no C++ run loop")
        self.boot_image(list(memory_image), start_pc=start_pc)
        self.seed_run_state()
        out = self.engine.run(max_steps, stop_on_trap, 0)
        self.pull_run_state()
        return out

    def iter_run_arrays(self, memory_image, *, start_pc: int, max_steps: int,
                        chunk: int = 200_000, stop_on_trap: bool = False):
        """Stream the C++ run loop in fixed-size chunks (w_row rebased to 0
        per chunk), so the full Klaus suite verifies without materializing a
        30M-row trace.  Yields the run-protocol result dict for each chunk.

        Engine state is O(1) in the token log (memory on the exact
        nearest-address structure, constant-query heads on running-argmax
        registers), so streaming stays bounded between chunks.
        """
        if not self.supports_run_loop:
            raise RuntimeError("this engine has no C++ run loop")
        self.boot_image(list(memory_image), start_pc=start_pc)
        self.seed_run_state()
        done = 0
        while done < max_steps:
            base = int(self.engine.get_run_state()["steps"])
            out = self.engine.run(min(chunk, max_steps - done), stop_on_trap, 0)
            n = len(out["pc"])
            if n == 0:
                break
            # w_row is the engine's global step_index; rebase to this chunk.
            out["w_row"] = out["w_row"] - base
            done += n
            yield out
            if out["reason"] in ("halt", "trap"):
                break
        self.pull_run_state()

    # ── Machine API (run_protocol.py) ───────────────────────────────
    # The console sets the machine up here (image, watches) and then
    # drives the engine through its native ABI (machine_abi); run() is
    # the same loop for Python callers.

    def load_image(self, memory_image, *, start_pc: int) -> None:
        self.boot_image(list(memory_image), start_pc=start_pc)
        self.seed_run_state()

    def run(self, max_steps: int, *, stop_on_trap: bool = False,
            max_cycles: int = 0, trace: bool = True) -> dict:
        """Resume the native run loop from the engine's persistent state
        (no reboot), pausing on halt/trap/watch/cycle budget."""
        out = self.engine.run(max_steps, stop_on_trap, max_cycles, trace)
        self.pull_run_state()
        # A watch_write pause ends on the step that wrote, so this is the
        # writing opcode: the bus routes RMW writes through write_rmw.
        out["pause_ir"] = self.mstate["ir"]
        return out

    def watch_read(self, lo: int, hi: int) -> None:
        self.engine.watch_read(lo, hi)

    def watch_write(self, lo: int, hi: int) -> None:
        self.engine.watch_write(lo, hi)

    def read_ram(self, addr: int) -> int:
        return self.engine.read_ram(addr)

    def dump_ram(self) -> bytes:
        return self.engine.dump_ram()

    def get_state(self) -> dict:
        return dict(self.engine.get_run_state())


# ── Verification program (low addresses; see numerics caveat above) ──

VERIFY_START = 0x0400


def verify_program() -> bytes:
    """Loop + rewrites + stack + subroutine + arithmetic, all below $0500."""
    image = bytearray(0x10000)
    code = bytes([
        0xA9, 0x05,        # LDA #$05
        0x85, 0x10,        # STA $10
        0xA2, 0x03,        # LDX #$03
        0xE6, 0x10,        # loop: INC $10   (rewrites the same address)
        0xCA,              # DEX
        0xD0, 0xFB,        # BNE loop
        0x20, 0x20, 0x04,  # JSR $0420
        0xA5, 0x10,        # LDA $10
        0x38,              # SEC
        0xE9, 0x02,        # SBC #$02
        0x85, 0x11,        # STA $11
        0x69, 0x10,        # ADC #$10
        0x48,              # PHA
        0x68,              # PLA
        0x85, 0x12,        # STA $12
        0x4C, 0x1B, 0x04,  # trap: JMP trap ($041B)
    ])
    image[VERIFY_START:VERIFY_START + len(code)] = code
    image[0x0420:0x0426] = bytes([
        0xA0, 0x07,        # LDY #$07
        0x84, 0x13,        # STY $13
        0x60,              # RTS
    ])
    return bytes(image)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Verify the 6502 running as real transformer weights."
    )
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    machine = Cpu6502TransformerMachine(bcd=False)
    logger.info(
        "transformer: d_model=%d n_layers=%d exact-int32 (%.1fs build)",
        machine.d_model,
        machine.n_layers,
        machine.build_seconds,
    )

    image = verify_program()
    run_start = time.perf_counter()
    trace = machine.run_image(image, start_pc=VERIFY_START, max_steps=args.steps)
    run_elapsed = time.perf_counter() - run_start

    ref = reference.Cpu6502(memory=bytearray(image), bcd_enabled=False)
    ref.pc = VERIFY_START
    failures = 0
    for i, row in enumerate(trace):
        ref_row = ref.step()
        a, b = ref_row.state, row.state
        ok = (a.pc, a.a, a.x, a.y, a.s, a.p) == (b.pc, b.a, b.x, b.y, b.s, b.p)
        ok = ok and {(w.addr, w.value) for w in ref_row.writes} == {
            (w.addr, w.value) for w in row.writes
        }
        if not ok:
            failures += 1
            logger.error("step %d DIVERGED:\nREF:         %s\nTRANSFORMER: %s", i, a, b)
    status = "MATCHED" if failures == 0 else f"{failures} DIVERGENCES"
    logger.info(
        "transformer: %s  %d steps in %.1fs (%.2f steps/s)",
        status, len(trace.rows), run_elapsed, len(trace.rows) / run_elapsed,
    )
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
