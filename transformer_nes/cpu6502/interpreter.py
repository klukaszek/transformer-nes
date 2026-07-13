"""6502 CPU expressed in Percepta's computation graph DSL.

This is an instruction-level interpreter (one transformer step = one 6502
instruction) for the full official NMOS 6502 opcode set plus the stable
unofficial opcodes exercised by nestest.  Code and data share one
attention-backed 64 KiB memory image, so instruction fetch is just memory
read at PC.  CPU state is stored in the residual stream of STATE tokens; the
"current" state is the latest STATE token.  Memory writes are appended as
WRITE tokens; reads look up the latest write for an address, with non-exact
nearest-address hits masked to zero (so unwritten memory reads as 0).

Correctness contract for graph construction:
  - stepglu(one, b) equals step(b >= 0) ONLY for integer-valued b; every
    threshold expression must therefore be an integer (never `x - 9.5`).
    Likewise match_zero/match_byte require integer arguments.
  - wrap8 / wrap16 subtract at most one modulus, so their arguments must
    lie in [-256, 511] / [-65536, 131071]; splitting a full 16-bit value
    into bytes must instead be derived from values that are already bytes
    (see the JSR/BRK push-address construction).
  - Opcode/mode gates are one-hot 0/1 signals; mux chains rely on this.
  - bcd=False models the NES 2A03, which ignores the D flag: ADC/SBC (and
    the unofficial RRA/ISB) always compute in binary, and the decimal
    circuitry is left out of the graph entirely.  The unofficial RRA/ISB
    are binary-only even with bcd=True (no test suite exercises them in
    decimal mode).
  - Flags in decimal mode: V always reflects the binary sum, as on NMOS
    hardware; the reference core matches this.
  - Interrupts are graph logic, not harness magic: when a STATE token
    carries in_nmi=1 or in_irq=1 (never both; the harness gates IRQ on the
    I flag at the step boundary), the step performs the interrupt sequence
    instead of the fetched instruction -- push PCH/PCL/P (B clear, U set)
    through the three write channels, S-3, I=1, and vector through
    $FFFA/$FFFB (NMI) or $FFFE/$FFFF (IRQ) via the shared vector-read
    slots.  Every effect of the (decoded but not executed) instruction at
    PC is discarded by final is_int muxes.  The interrupt trace row
    reports the interrupted PC with ea=0 and data=0.
"""

from __future__ import annotations

from transformer_nes.cpu6502.spec import (
    FLAG_B,
    FLAG_C,
    FLAG_D,
    FLAG_I,
    FLAG_N,
    FLAG_U,
    FLAG_V,
    FLAG_Z,
    MODE_ABSOLUTE,
    MODE_ABSOLUTE_X,
    MODE_ABSOLUTE_Y,
    MODE_ACCUMULATOR,
    MODE_IMMEDIATE,
    MODE_IMPLIED,
    MODE_INDIRECT,
    MODE_INDIRECT_X,
    MODE_INDIRECT_Y,
    MODE_RELATIVE,
    MODE_ZERO_PAGE,
    MODE_ZERO_PAGE_X,
    MODE_ZERO_PAGE_Y,
    OPCODES,
)
from transformer_nes.graph import core as graph
from transformer_nes.graph.core import (
    Expression,
    InputDimension,
    auto_name,
    fetch,
    persist,
    reglu,
    stepglu,
    to_expr,
)

# ── Low-level gating helpers ───────────────────────────────────────


def match_byte(value: Expression, byte: int) -> Expression:
    """Return a 0/1 gate: 1 iff value == byte."""
    return match_zero(value - byte)


def match_zero(value: Expression) -> Expression:
    """Return a 0/1 gate: 1 iff an integer expression is exactly zero."""
    one = graph.one
    abs_diff = persist(reglu(value, value) + reglu(-value, -value))
    return stepglu(one, -abs_diff)


def multiply(a, b) -> Expression:
    """Multiply two scalar expressions using the graph DSL's internal helper."""
    from transformer_nes.graph.core import make_multiply, to_expr

    return make_multiply(to_expr(a), to_expr(b))


def mux(gate: Expression, if_one: Expression, if_zero: Expression) -> Expression:
    """Select between two linear expressions using a 0/1 gate."""
    return if_zero + multiply(if_one - if_zero, gate)


def wrap8(value: Expression) -> Expression:
    """Reduce an integer expression to an unsigned 8-bit value."""
    one = graph.one
    return value - 256 * stepglu(one, value - 256) + 256 * stepglu(one, -value - 1)


def wrap16(value: Expression) -> Expression:
    """Reduce an integer expression to an unsigned 16-bit value."""
    one = graph.one
    return value - 65536 * stepglu(one, value - 65536) + 65536 * stepglu(one, -value - 1)


def signed_byte(value: Expression) -> Expression:
    """Interpret an unsigned byte as a signed 8-bit value."""
    one = graph.one
    return value - 256 * stepglu(one, value - 128)


def memory_at(address: Expression, name: str) -> Expression:
    """Read the latest WRITE token at the given 16-bit address."""
    one = graph.one
    write_addr_expr = addr_lo + 256 * addr_hi
    # Split 64 KiB memory into 4 KiB lookup groups.  Each group uses a
    # 12-bit local-address parabola, keeping K-side scores below 2^24 so
    # adjacent addresses remain distinguishable after float32 Q/K rounding.
    group_size = 4096
    group_count = 16

    value = 0
    for group in range(group_count):
        lo = group * group_size
        hi = lo + group_size - 1

        if group == 0:
            query_in_group = stepglu(one, hi - address)
            write_in_group = stepglu(one, hi - write_addr_expr)
        elif group == group_count - 1:
            query_in_group = stepglu(one, address - lo)
            write_in_group = stepglu(one, write_addr_expr - lo)
        else:
            query_ge_lo = stepglu(one, address - lo)
            query_le_hi = stepglu(one, hi - address)
            query_in_group = multiply(query_ge_lo, query_le_hi)
            write_ge_lo = stepglu(one, write_addr_expr - lo)
            write_le_hi = stepglu(one, hi - write_addr_expr)
            write_in_group = multiply(write_ge_lo, write_le_hi)

        group_write = multiply(is_memory_write, write_in_group)
        group_value, found_addr = fetch(
            [byte_value, write_addr_expr],
            query=address - lo,
            key=write_addr_expr - lo,
            clear_key=1 - group_write,
            tie_break="latest",
        )
        group_value.lookup.memory_group = group
        exact = match_zero(address - found_addr)
        value = value + multiply(query_in_group, multiply(group_value, exact))
    result = persist(value, name=name)
    # The site's raw ingredients, for the expert folds: they rebuild the
    # read as ONE 64K-span lookup + exact mask (graph/fold.py).  The
    # 16-group form above exists for the union's f32 attention-score
    # range; expert engines serve reads exactly and never compute it.
    site = next(iter(result.terms))
    site.memory_read = {
        "address": to_expr(address),
        "byte_value": to_expr(byte_value),
        "write_addr": write_addr_expr,
        "is_write": to_expr(is_memory_write),
    }
    return result


# ── Graph helper implementations ───────────────────────────────────


def overflow(a: Expression, b: Expression, r: Expression) -> Expression:
    """Return 1 iff the sign of r differs from a and b while a and b agree."""
    one = graph.one
    a_sign = stepglu(one, a - 128)
    b_sign = stepglu(one, b - 128)
    r_sign = stepglu(one, r - 128)
    same_sign = 1 - multiply(a_sign - b_sign, a_sign - b_sign)
    diff_from_result = multiply(a_sign - r_sign, a_sign - r_sign)
    return multiply(same_sign, diff_from_result)


def extract_bits(value: Expression, name: str) -> list[Expression]:
    """Extract bits 7..0 of an 8-bit value expression using binary thresholds.

    Returns a list where bits[0] is bit 7 (MSB) and bits[7] is bit 0 (LSB).
    """
    one = graph.one
    bits = []
    remaining = value
    for i, place in enumerate((128, 64, 32, 16, 8, 4, 2, 1)):
        bit = stepglu(one, remaining - place)
        bit = persist(bit, name=f"{name}bit{7 - i}")
        bits.append(bit)
        remaining = remaining - place * bit
    return bits


def pulled_flag_bits(value: Expression, name: str) -> dict[int, Expression]:
    """All eight bits of an 8-bit value from ONE shared threshold ladder.

    ge[k] = step(value >= k) for k in 1..255; every bit is then a linear
    telescoping run sum over the ladder (eq_k = ge_k - ge_{k+1}), so eight
    extractions share 255 parallel stepglus at depth one."""
    one = graph.one
    ge = [None] + [stepglu(one, value - k) for k in range(1, 256)]
    bits: dict[int, Expression] = {}
    for b in range(8):
        result = 0
        for start in range(1 << b, 256, 1 << (b + 1)):
            result = result + ge[start]
            if start + (1 << b) < 256:
                result = result - ge[start + (1 << b)]
        bits[1 << b] = persist(result, name=f"{name}_bit{b}")
    return bits


# ── Main graph builder ─────────────────────────────────────────────


def build(*, bcd: bool = True) -> tuple[dict[str, Expression], dict[str, Expression]]:
    """Build the 6502 graph.  See the module docstring for the construction
    contract; bcd=False builds the NES 2A03 variant (no decimal mode)."""
    one = graph.one

    # ── Input dimensions ────────────────────────────────────────────
    global byte_value, is_memory_write, addr_lo, addr_hi, is_start, is_state
    byte_value = InputDimension("byte_value")
    is_memory_write = InputDimension("is_memory_write")
    addr_lo = InputDimension("addr_lo")
    addr_hi = InputDimension("addr_hi")
    is_start = InputDimension("is_start")
    is_state = InputDimension("is_state")

    in_pc_lo = InputDimension("in_pc_lo")
    in_pc_hi = InputDimension("in_pc_hi")
    in_a = InputDimension("in_a")
    in_x = InputDimension("in_x")
    in_y = InputDimension("in_y")
    in_s = InputDimension("in_s")
    in_p_c = InputDimension("in_p_c")
    in_p_z = InputDimension("in_p_z")
    in_p_i = InputDimension("in_p_i")
    in_p_d = InputDimension("in_p_d")
    in_p_b = InputDimension("in_p_b")
    in_p_u = InputDimension("in_p_u")
    in_p_v = InputDimension("in_p_v")
    in_p_n = InputDimension("in_p_n")
    in_ir = InputDimension("in_ir")
    in_oplo = InputDimension("in_oplo")
    in_ophi = InputDimension("in_ophi")
    in_ea_lo = InputDimension("in_ea_lo")
    in_ea_hi = InputDimension("in_ea_hi")
    in_data = InputDimension("in_data")
    in_halted = InputDimension("in_halted")
    # Interrupt requests: set to 1 by the harness on the STATE token *before*
    # the interrupted instruction would execute (never both; NMI wins, and
    # IRQ delivery is gated on the I flag at the harness's step boundary).
    # The entire interrupt sequence (stack pushes, I flag, vector fetch) is
    # graph logic below; the harness never constructs CPU state by hand.
    in_nmi = InputDimension("in_nmi")
    in_irq = InputDimension("in_irq")

    flag_bits = [in_p_c, in_p_z, in_p_i, in_p_d, in_p_b, in_p_u, in_p_v, in_p_n]
    flag_weights = [FLAG_C, FLAG_Z, FLAG_I, FLAG_D, FLAG_B, FLAG_U, FLAG_V, FLAG_N]

    # ── Input tokens ────────────────────────────────────────────────
    input_tokens: dict[str, Expression] = {}

    input_tokens["STATE"] = (
        in_pc_lo + 256 * in_pc_hi
        + in_a
        + in_x
        + in_y
        + in_s
        + sum(w * bit for w, bit in zip(flag_weights, flag_bits, strict=True))
        + in_ir
        + in_oplo
        + in_ophi
        + in_ea_lo + 256 * in_ea_hi
        + in_data
        + in_halted
        + in_nmi
        + in_irq
        + is_state
    )

    input_tokens["WRITE"] = is_memory_write + addr_lo + 256 * addr_hi + byte_value
    input_tokens["HALT"] = 0 * one

    start_pc_lo = InputDimension("start_pc_lo")
    start_pc_hi = InputDimension("start_pc_hi")
    input_tokens["start"] = start_pc_lo + 256 * start_pc_hi + is_start

    start_pc_lo_val, start_pc_hi_val = fetch(
        [start_pc_lo, start_pc_hi],
        query=one,
        key=is_start,
        clear_key=1 - is_start,
        tie_break="latest",
    )
    start_pc = start_pc_lo_val + 256 * start_pc_hi_val
    start_pc = persist(start_pc, name="start_pc")

    # ── Current state: latest STATE token ───────────────────────────
    state_field_names = [
        "current_pc_lo", "current_pc_hi", "current_a", "current_x", "current_y",
        "current_s", "current_p_c", "current_p_z", "current_p_i", "current_p_d",
        "current_p_b", "current_p_u", "current_p_v", "current_p_n", "current_ir",
        "current_oplo", "current_ophi", "current_ea_lo", "current_ea_hi",
        "current_data", "current_halted", "is_nmi", "is_irq",
    ]
    state_fields = [
        in_pc_lo, in_pc_hi, in_a, in_x, in_y, in_s,
        in_p_c, in_p_z, in_p_i, in_p_d, in_p_b, in_p_u, in_p_v, in_p_n,
        in_ir, in_oplo, in_ophi, in_ea_lo, in_ea_hi, in_data, in_halted,
        in_nmi, in_irq,
    ]
    latest_state = fetch(
        state_fields,
        query=one,
        key=is_state,
        clear_key=1 - is_state,
        tie_break="latest",
    )
    latest_state = tuple(
        persist(v, name=n)
        for v, n in zip(latest_state, state_field_names, strict=True)
    )

    (
        current_pc_lo, current_pc_hi, current_a, current_x, current_y, current_s,
        current_p_c, current_p_z, current_p_i, current_p_d,
        current_p_b, current_p_u, current_p_v, current_p_n,
        current_ir, current_oplo, current_ophi, current_ea_lo, current_ea_hi,
        current_data, current_halted, is_nmi, is_irq,
    ) = latest_state

    # Either interrupt runs the same sequence; only the vector differs.
    is_int = is_nmi + is_irq

    current_pc = current_pc_lo + 256 * current_pc_hi
    current_pc = persist(current_pc, name="current_pc")
    current_p = (
        FLAG_C * current_p_c
        + FLAG_Z * current_p_z
        + FLAG_I * current_p_i
        + FLAG_D * current_p_d
        + FLAG_B * current_p_b
        + FLAG_U * current_p_u
        + FLAG_V * current_p_v
        + FLAG_N * current_p_n
    )
    current_p = persist(current_p, name="current_p")
    current_ea = current_ea_lo + 256 * current_ea_hi
    current_ea = persist(current_ea, name="current_ea")

    # ── Instruction fetch ───────────────────────────────────────────
    # Code and data share the attention-backed memory, so instruction fetch
    # is three memory reads at PC, PC+1, PC+2.
    fetched_opcode = memory_at(current_pc, "fetched_opcode_raw")
    fetched_operand_lo = memory_at(wrap16(current_pc + 1), "fetched_operand_lo_raw")
    fetched_operand_hi = memory_at(wrap16(current_pc + 2), "fetched_operand_hi_raw")

    fetched_opcode = persist(fetched_opcode, name="fetched_opcode")
    # Raw operand bytes, used by the addressing machinery below.  The trace
    # *reports* them gated by instruction length (reference convention), so
    # the readout dims named fetched_operand_lo/hi are built near data_out.
    fetched_operand_lo = persist(fetched_operand_lo, name="fetched_operand_lo_all")
    fetched_operand_hi = persist(fetched_operand_hi, name="fetched_operand_hi_all")

    # ── Per-opcode gates ────────────────────────────────────────────
    opcode_gates: dict[int, Expression] = {}
    for op in OPCODES.values():
        opcode_gates[op.byte] = persist(
            match_byte(fetched_opcode, op.byte), name=f"is_op_{op.byte:02X}"
        )

    # ── Table-driven mode / operation families ──────────────────────
    all_modes = {
        MODE_IMMEDIATE, MODE_ZERO_PAGE, MODE_ZERO_PAGE_X, MODE_ZERO_PAGE_Y,
        MODE_ABSOLUTE, MODE_ABSOLUTE_X, MODE_ABSOLUTE_Y,
        MODE_INDIRECT, MODE_INDIRECT_X, MODE_INDIRECT_Y,
        MODE_RELATIVE, MODE_IMPLIED, MODE_ACCUMULATOR,
    }
    mode_gates: dict[str, Expression] = {m: 0 for m in all_modes}
    op_family_gates: dict[str, Expression] = {op.operation: 0 for op in OPCODES.values()}
    for op in OPCODES.values():
        gate = opcode_gates[op.byte]
        mode_gates[op.mode] = mode_gates[op.mode] + gate
        op_family_gates[op.operation] = op_family_gates[op.operation] + gate

    is_imm = persist(mode_gates[MODE_IMMEDIATE], name="is_imm")
    is_zpg = persist(mode_gates[MODE_ZERO_PAGE], name="is_zpg")
    is_zpx = persist(mode_gates[MODE_ZERO_PAGE_X], name="is_zpx")
    is_zpy = persist(mode_gates[MODE_ZERO_PAGE_Y], name="is_zpy")
    is_abs = persist(mode_gates[MODE_ABSOLUTE], name="is_abs")
    is_abx = persist(mode_gates[MODE_ABSOLUTE_X], name="is_abx")
    is_aby = persist(mode_gates[MODE_ABSOLUTE_Y], name="is_aby")
    is_ind = persist(mode_gates[MODE_INDIRECT], name="is_ind")
    is_idx = persist(mode_gates[MODE_INDIRECT_X], name="is_idx")
    is_idy = persist(mode_gates[MODE_INDIRECT_Y], name="is_idy")
    is_rel = persist(mode_gates[MODE_RELATIVE], name="is_rel")
    is_imp = persist(mode_gates[MODE_IMPLIED], name="is_imp")
    is_acc = persist(mode_gates[MODE_ACCUMULATOR], name="is_acc")

    # Operation-family gates
    op_load_a = op_family_gates["load_a"]
    op_load_x = op_family_gates["load_x"]
    op_load_y = op_family_gates["load_y"]
    op_store_a = op_family_gates["store_a"]
    op_store_x = op_family_gates["store_x"]
    op_store_y = op_family_gates["store_y"]
    op_adc = op_family_gates["adc"]
    op_sbc = op_family_gates["sbc"]
    op_cmp = op_family_gates["cmp"]
    op_cpx = op_family_gates["cpx"]
    op_cpy = op_family_gates["cpy"]
    op_and = op_family_gates["and"]
    op_ora = op_family_gates["ora"]
    op_eor = op_family_gates["eor"]
    op_bit = op_family_gates["bit"]
    op_asl = op_family_gates["asl"]
    op_lsr = op_family_gates["lsr"]
    op_rol = op_family_gates["rol"]
    op_ror = op_family_gates["ror"]
    op_inc = op_family_gates["inc"]
    op_dec = op_family_gates["dec"]
    op_inx = op_family_gates["inx"]
    op_iny = op_family_gates["iny"]
    op_dex = op_family_gates["dex"]
    op_dey = op_family_gates["dey"]
    op_tax = op_family_gates["tax"]
    op_tay = op_family_gates["tay"]
    op_txa = op_family_gates["txa"]
    op_tya = op_family_gates["tya"]
    op_tsx = op_family_gates["tsx"]
    op_txs = op_family_gates["txs"]
    op_pha = op_family_gates["pha"]
    op_php = op_family_gates["php"]
    op_pla = op_family_gates["pla"]
    op_plp = op_family_gates["plp"]
    op_jmp = op_family_gates["jmp"]
    op_jsr = op_family_gates["jsr"]
    op_rts = op_family_gates["rts"]
    op_rti = op_family_gates["rti"]
    op_brk = op_family_gates["brk"]
    op_bpl = op_family_gates["bpl"]
    op_bmi = op_family_gates["bmi"]
    op_bvc = op_family_gates["bvc"]
    op_bvs = op_family_gates["bvs"]
    op_bcc = op_family_gates["bcc"]
    op_bcs = op_family_gates["bcs"]
    op_bne = op_family_gates["bne"]
    op_beq = op_family_gates["beq"]
    op_clc = op_family_gates["clc"]
    op_sec = op_family_gates["sec"]
    op_cli = op_family_gates["cli"]
    op_sei = op_family_gates["sei"]
    op_clv = op_family_gates["clv"]
    op_cld = op_family_gates["cld"]
    op_sed = op_family_gates["sed"]
    op_lax = op_family_gates["lax"]
    op_sax = op_family_gates["sax"]
    op_dcp = op_family_gates["dcp"]
    op_isb = op_family_gates["isb"]
    op_slo = op_family_gates["slo"]
    op_rla = op_family_gates["rla"]
    op_sre = op_family_gates["sre"]
    op_rra = op_family_gates["rra"]

    # ── Addressing / EA / length ────────────────────────────────────
    abs_addr = fetched_operand_lo + 256 * fetched_operand_hi
    abs_addr = persist(abs_addr, name="abs_addr")

    rel_offset = persist(signed_byte(fetched_operand_lo), name="rel_offset")

    length = persist(
        1 * (is_imp + is_acc)
        + 2 * (is_imm + is_zpg + is_zpx + is_zpy + is_rel + is_idx + is_idy)
        + 3 * (is_abs + is_abx + is_aby + is_ind),
        name="length",
    )
    pc_after = persist(current_pc + length, name="pc_after")
    branch_target = persist(pc_after + rel_offset, name="branch_target")

    # Base EA for simple modes
    base_ea = mux(
        is_zpg,
        fetched_operand_lo,
        mux(
            is_zpx,
            wrap8(fetched_operand_lo + current_x),
            mux(
                is_zpy,
                wrap8(fetched_operand_lo + current_y),
                mux(
                    is_abs,
                    abs_addr,
                    mux(
                        is_abx,
                        wrap16(abs_addr + current_x),
                        mux(
                            is_aby,
                            wrap16(abs_addr + current_y),
                            mux(
                                is_rel,
                                branch_target,
                                mux(
                                    is_imm,
                                    fetched_operand_lo,
                                    0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    base_ea = persist(base_ea, name="base_ea")

    # Indirect pointers: (zp+X) for idx, zp for idy, abs for JMP (ind).
    # The effective address itself is resolved after the pointer memory
    # reads below.
    idx_ptr = persist(wrap8(fetched_operand_lo + current_x), name="idx_ptr")
    idy_ptr = persist(fetched_operand_lo, name="idy_ptr")

    # Stack read addresses (shared by PLA/PLP/RTS/RTI).  The stack pointer
    # wraps within page 1, so the offset is reduced mod 256 before adding
    # the page base.
    stack_pull_addr = persist(0x0100 + wrap8(current_s + 1), name="stack_pull_addr")
    stack_pull_addr2 = persist(0x0100 + wrap8(current_s + 2), name="stack_pull_addr2")
    stack_pull_addr3 = persist(0x0100 + wrap8(current_s + 3), name="stack_pull_addr3")

    # ── Memory read ─────────────────────────────────────────────────
    # Shared memory-read slots.  A single instruction only uses one family of
    # reads at a time, so we mux the addresses into three slots and share the
    # attention lookups across pointer reads, stack reads, and IRQ-vector reads.
    is_pointer_read = is_idx + is_idy + is_ind
    pointer_base = mux(is_idx, idx_ptr, mux(is_idy, idy_ptr, abs_addr))
    addrA = mux(
        is_pointer_read,
        pointer_base,
        mux(
            op_pla + op_plp + op_rts + op_rti,
            stack_pull_addr,
            mux(op_brk, 0xFFFE, 0),
        ),
    )
    # Second pointer byte: zero-page wrap for (zp,X)/(zp),Y; NMOS page-wrap
    # bug for JMP (abs) -- the high byte is read from the same page.
    pointer_next = mux(
        is_ind,
        256 * fetched_operand_hi + wrap8(fetched_operand_lo + 1),
        wrap8(pointer_base + 1),
    )
    addrB = mux(
        is_pointer_read,
        pointer_next,
        mux(
            op_pla + op_plp + op_rts + op_rti,
            stack_pull_addr2,
            mux(op_brk, 0xFFFF, 0),
        ),
    )
    addrC = mux(op_pla + op_plp + op_rts + op_rti, stack_pull_addr3, 0)
    # An interrupt step reads its vector through the same two slots BRK uses
    # for $FFFE/$FFFF; everything else the interrupted instruction would have
    # read is left in place and discarded by the is_int muxes below.
    addrA = mux(is_nmi, 0xFFFA, mux(is_irq, 0xFFFE, addrA))
    addrB = mux(is_nmi, 0xFFFB, mux(is_irq, 0xFFFF, addrB))
    addrA = persist(addrA, name="addrA")
    addrB = persist(addrB, name="addrB")
    addrC = persist(addrC, name="addrC")

    mem_A = memory_at(addrA, "mem_A")
    mem_B = memory_at(addrB, "mem_B")
    mem_C = memory_at(addrC, "mem_C")

    idx_lo = mem_A
    idx_hi = mem_B
    idx_ea = persist(idx_lo + 256 * idx_hi, name="idx_ea")
    idy_lo = mem_A
    idy_hi = mem_B
    idy_ea = persist(wrap16(idy_lo + 256 * idy_hi + current_y), name="idy_ea")
    ind_ptr_lo = mem_A
    ind_ptr_hi = mem_B
    ind_ea = persist(ind_ptr_lo + 256 * ind_ptr_hi, name="ind_ea")

    # Effective address: indirect modes resolved from the pointer reads above.
    # An interrupt step performs no operand access; its trace EA is 0.
    ea = mux(is_idx, idx_ea, mux(is_idy, idy_ea, mux(is_ind, ind_ea, base_ea)))
    ea = mux(is_int, 0, ea)
    ea = persist(ea, name="ea")

    # Stack/IRQ vector reads
    stack_pull_byte = mem_A
    stack_pull_byte2 = mem_B
    stack_pull_byte3 = mem_C
    pulled_word = persist(stack_pull_byte + 256 * stack_pull_byte2, name="pulled_word")
    # RTI pulls P first, then the return PC (no +1, unlike RTS).
    rti_return = persist(stack_pull_byte2 + 256 * stack_pull_byte3, name="rti_return")
    # Interrupt vector: $FFFE/$FFFF for BRK, $FFFA/$FFFB for NMI (the slot
    # addresses are muxed above, so this word is whichever vector applies).
    irq_vector_lo = mem_A
    irq_vector_hi = mem_B
    irq_vector = persist(irq_vector_lo + 256 * irq_vector_hi, name="irq_vector")

    # Operand memory read (for loads/stores/ALU with memory addressing)
    operand_mem = memory_at(ea, "operand_mem")
    operand = mux(
        is_imm,
        fetched_operand_lo,
        mux(
            is_acc,
            current_a,
            operand_mem,
        ),
    )
    operand = persist(operand, name="operand")

    # ── Bit extraction for bitwise / shift ops ────────────────────────
    a_bits = extract_bits(current_a, "a")
    operand_bits = extract_bits(operand, "operand")

    # ── ALU operations ──────────────────────────────────────────────
    carry_in = current_p_c

    # Binary ADC
    adc_raw = current_a + operand + carry_in
    adc_carry = stepglu(one, adc_raw - 256)
    adc_result = wrap8(adc_raw)
    adc_overflow = overflow(current_a, operand, adc_result)

    # Binary SBC
    sbc_operand = 255 - operand
    sbc_raw = current_a + sbc_operand + carry_in
    sbc_carry = stepglu(one, sbc_raw - 256)
    sbc_result = wrap8(sbc_raw)
    sbc_overflow = overflow(current_a, sbc_operand, sbc_result)

    if bcd:
        # Nibbles for decimal mode (bits[0] = bit 7, bits[4] = bit 3, bits[7] = bit 0)
        a_hi_nibble = 8 * a_bits[0] + 4 * a_bits[1] + 2 * a_bits[2] + 1 * a_bits[3]
        a_lo_nibble = 8 * a_bits[4] + 4 * a_bits[5] + 2 * a_bits[6] + 1 * a_bits[7]
        m_hi_nibble = 8 * operand_bits[0] + 4 * operand_bits[1] + 2 * operand_bits[2] + 1 * operand_bits[3]
        m_lo_nibble = 8 * operand_bits[4] + 4 * operand_bits[5] + 2 * operand_bits[6] + 1 * operand_bits[7]
        a_hi_nibble = persist(a_hi_nibble, name="a_hi_nibble")
        a_lo_nibble = persist(a_lo_nibble, name="a_lo_nibble")
        m_hi_nibble = persist(m_hi_nibble, name="m_hi_nibble")
        m_lo_nibble = persist(m_lo_nibble, name="m_lo_nibble")

        # Decimal ADC.  Thresholds must be integers: stepglu is only exact
        # for integer arguments (a -9.5 offset makes al == 9 yield 0.5).
        al_adc = a_lo_nibble + m_lo_nibble + carry_in
        half_carry_adc = stepglu(one, al_adc - 10)
        al_adc_mod = al_adc - 10 * half_carry_adc
        ah_adc = a_hi_nibble + m_hi_nibble + half_carry_adc
        carry_out_adc = stepglu(one, ah_adc - 10)
        ah_adc_mod = ah_adc - 10 * carry_out_adc
        adc_decimal_result = ah_adc_mod * 16 + al_adc_mod
        adc_decimal_result = persist(adc_decimal_result, name="adc_decimal_result")
        decimal_adc = mux(current_p_d, adc_decimal_result, adc_result)
        decimal_adc = persist(decimal_adc, name="decimal_adc")
        decimal_adc_carry = mux(current_p_d, carry_out_adc, adc_carry)

        # Decimal SBC (integer thresholds, as above)
        al_sbc = a_lo_nibble - m_lo_nibble - (1 - carry_in)
        borrow_sbc = 1 - stepglu(one, al_sbc)
        al_sbc_mod = al_sbc + 10 * borrow_sbc
        ah_sbc = a_hi_nibble - m_hi_nibble - borrow_sbc
        carry_out_sbc = stepglu(one, ah_sbc)
        ah_sbc_mod = ah_sbc + 10 * (1 - carry_out_sbc)
        sbc_decimal_result = ah_sbc_mod * 16 + al_sbc_mod
        sbc_decimal_result = persist(sbc_decimal_result, name="sbc_decimal_result")
        decimal_sbc = mux(current_p_d, sbc_decimal_result, sbc_result)
        decimal_sbc = persist(decimal_sbc, name="decimal_sbc")
        decimal_sbc_carry = mux(current_p_d, carry_out_sbc, sbc_carry)
    else:
        # 2A03: the D flag is ignored, arithmetic is always binary.
        decimal_adc = persist(adc_result, name="decimal_adc")
        decimal_adc_carry = adc_carry
        decimal_sbc = persist(sbc_result, name="decimal_sbc")
        decimal_sbc_carry = sbc_carry

    # Compare operations (A/M/X/Y - operand)
    cmp_raw = current_a - operand
    cmp_carry = stepglu(one, cmp_raw)
    cmp_result = wrap8(cmp_raw)
    cmp_n = stepglu(one, cmp_result - 128)
    cmp_z = 1 - stepglu(one, cmp_result - 1)

    cpx_raw = current_x - operand
    cpx_carry = stepglu(one, cpx_raw)
    cpx_result = wrap8(cpx_raw)
    cpx_n = stepglu(one, cpx_result - 128)
    cpx_z = 1 - stepglu(one, cpx_result - 1)

    cpy_raw = current_y - operand
    cpy_carry = stepglu(one, cpy_raw)
    cpy_result = wrap8(cpy_raw)
    cpy_n = stepglu(one, cpy_result - 128)
    cpy_z = 1 - stepglu(one, cpy_result - 1)

    # Logical operations
    and_result = 0
    ora_result = 0
    eor_result = 0
    for i in range(8):
        a_bit = a_bits[7 - i]
        m_bit = operand_bits[7 - i]
        bit_and = multiply(a_bit, m_bit)
        bit_or = a_bit + m_bit - bit_and
        bit_xor = a_bit + m_bit - 2 * bit_and
        and_result = and_result + (1 << i) * bit_and
        ora_result = ora_result + (1 << i) * bit_or
        eor_result = eor_result + (1 << i) * bit_xor
    and_result = persist(and_result, name="and_result")
    ora_result = persist(ora_result, name="ora_result")
    eor_result = persist(eor_result, name="eor_result")

    # BIT
    bit_n = operand_bits[0]
    bit_v = operand_bits[1]
    bit_z = 1 - stepglu(one, and_result - 1)

    # Shifts (using arithmetic and the extracted LSB for LSR/ROR carry)
    asl_result = wrap8(2 * operand)
    asl_carry = operand_bits[0]
    asl_n = stepglu(one, asl_result - 128)
    asl_z = 1 - stepglu(one, asl_result - 1)

    lsr_result = operand * 0.5 - operand_bits[7] * 0.5
    lsr_result = persist(lsr_result, name="lsr_result")
    lsr_carry = operand_bits[7]
    lsr_n = 0
    lsr_z = 1 - stepglu(one, lsr_result - 1)

    rol_result = wrap8(2 * operand + carry_in)
    rol_carry = operand_bits[0]
    rol_n = stepglu(one, rol_result - 128)
    rol_z = 1 - stepglu(one, rol_result - 1)

    ror_result = operand * 0.5 - operand_bits[7] * 0.5 + carry_in * 128
    ror_result = persist(ror_result, name="ror_result")
    ror_carry = operand_bits[7]
    ror_n = stepglu(one, ror_result - 128)
    ror_z = 1 - stepglu(one, ror_result - 1)

    # Increments / decrements
    inc_result = wrap8(operand + 1)
    inc_n = stepglu(one, inc_result - 128)
    inc_z = 1 - stepglu(one, inc_result - 1)

    dec_result = wrap8(operand - 1)
    dec_n = stepglu(one, dec_result - 128)
    dec_z = 1 - stepglu(one, dec_result - 1)

    inx_result = wrap8(current_x + 1)
    iny_result = wrap8(current_y + 1)
    dex_result = wrap8(current_x - 1)
    dey_result = wrap8(current_y - 1)

    # ── Unofficial opcode ALU ───────────────────────────────────────
    # SAX stores A & X.
    x_bits = extract_bits(current_x, "x")
    sax_result = 0
    for j in range(8):
        sax_result = sax_result + (1 << (7 - j)) * multiply(a_bits[j], x_bits[j])
    sax_result = persist(sax_result, name="sax_result")

    # SLO/RLA/SRE combine A with the shifted memory value.  The shifted
    # value's bits are the operand's bits re-indexed (None = constant 0),
    # so no additional bit extraction is needed.
    asl_bits = operand_bits[1:] + [None]           # (operand << 1) & 0xFF
    rla_bits = operand_bits[1:] + [carry_in]       # (operand << 1) | C
    lsr_bits = [None] + operand_bits[:-1]          # operand >> 1

    slo_result = 0   # A | asl
    rla_result = 0   # A & rol
    sre_result = 0   # A ^ lsr
    for j in range(8):
        weight = 1 << (7 - j)
        a_bit = a_bits[j]
        if asl_bits[j] is None:
            slo_result = slo_result + weight * a_bit
        else:
            both = multiply(a_bit, asl_bits[j])
            slo_result = slo_result + weight * (a_bit + asl_bits[j] - both)
        if rla_bits[j] is not None:
            rla_result = rla_result + weight * multiply(a_bit, rla_bits[j])
        if lsr_bits[j] is None:
            sre_result = sre_result + weight * a_bit
        else:
            both = multiply(a_bit, lsr_bits[j])
            sre_result = sre_result + weight * (a_bit + lsr_bits[j] - 2 * both)
    slo_result = persist(slo_result, name="slo_result")
    rla_result = persist(rla_result, name="rla_result")
    sre_result = persist(sre_result, name="sre_result")

    # DCP: DEC memory then CMP against A.
    dcp_raw = current_a - dec_result
    dcp_carry = stepglu(one, dcp_raw)
    dcp_cmp = wrap8(dcp_raw)
    dcp_n = stepglu(one, dcp_cmp - 128)
    dcp_z = 1 - stepglu(one, dcp_cmp - 1)

    # ISB: INC memory then SBC; RRA: ROR memory then ADC with the shifted-out
    # carry.  Both are modeled in binary only -- neither Klaus (official
    # opcodes) nor nestest (2A03, no BCD) exercises them in decimal mode.
    isb_inv = 255 - inc_result
    isb_raw = current_a + isb_inv + carry_in
    isb_carry = stepglu(one, isb_raw - 256)
    isb_result = persist(wrap8(isb_raw), name="isb_result")
    isb_overflow = overflow(current_a, isb_inv, isb_result)

    rra_carry_in = operand_bits[7]
    rra_raw = current_a + ror_result + rra_carry_in
    rra_carry = stepglu(one, rra_raw - 256)
    rra_result = persist(wrap8(rra_raw), name="rra_result")
    rra_overflow = overflow(current_a, ror_result, rra_result)

    # ── New register values ─────────────────────────────────────────
    new_a = current_a
    new_a = mux(op_load_a + op_lax, operand, new_a)
    new_a = mux(op_and, and_result, new_a)
    new_a = mux(op_ora, ora_result, new_a)
    new_a = mux(op_eor, eor_result, new_a)
    new_a = mux(op_adc, decimal_adc, new_a)
    new_a = mux(op_sbc, decimal_sbc, new_a)
    new_a = mux(op_pla, stack_pull_byte, new_a)
    new_a = mux(op_txa, current_x, new_a)
    new_a = mux(op_tya, current_y, new_a)
    new_a = mux(op_slo, slo_result, new_a)
    new_a = mux(op_rla, rla_result, new_a)
    new_a = mux(op_sre, sre_result, new_a)
    new_a = mux(op_rra, rra_result, new_a)
    new_a = mux(op_isb, isb_result, new_a)
    # Shifts only write A in accumulator mode (memory modes leave A alone).
    shift_result = mux(op_asl, asl_result,
        mux(op_lsr, lsr_result,
            mux(op_rol, rol_result,
                mux(op_ror, ror_result, 0))))
    new_a = mux(is_acc, shift_result, new_a)
    new_a = mux(is_int, current_a, new_a)
    new_a = persist(new_a, name="new_a")

    new_x = current_x
    new_x = mux(op_load_x + op_lax, operand, new_x)
    new_x = mux(op_inx, inx_result, new_x)
    new_x = mux(op_dex, dex_result, new_x)
    new_x = mux(op_tax, current_a, new_x)
    new_x = mux(op_tsx, current_s, new_x)
    new_x = mux(is_int, current_x, new_x)
    new_x = persist(new_x, name="new_x")

    new_y = current_y
    new_y = mux(op_load_y, operand, new_y)
    new_y = mux(op_iny, iny_result, new_y)
    new_y = mux(op_dey, dey_result, new_y)
    new_y = mux(op_tay, current_a, new_y)
    new_y = mux(is_int, current_y, new_y)
    new_y = persist(new_y, name="new_y")

    new_s = current_s
    new_s = mux(op_pha + op_php, wrap8(current_s - 1), new_s)
    new_s = mux(op_jsr, wrap8(current_s - 2), new_s)
    new_s = mux(op_brk, wrap8(current_s - 3), new_s)
    new_s = mux(op_pla + op_plp, wrap8(current_s + 1), new_s)
    new_s = mux(op_rts, wrap8(current_s + 2), new_s)
    new_s = mux(op_rti, wrap8(current_s + 3), new_s)
    new_s = mux(op_txs, current_x, new_s)
    # NMI pushes three bytes, exactly like BRK.
    new_s = mux(is_int, wrap8(current_s - 3), new_s)
    new_s = persist(new_s, name="new_s")

    # ── Flags ───────────────────────────────────────────────────────
    # NZ source selection
    nz_alu = mux(
        op_load_a + op_adc + op_sbc + op_and + op_ora + op_eor + op_pla + op_txa + op_tya
        + op_lax + op_slo + op_rla + op_sre + op_rra + op_isb,
        new_a,
        mux(
            op_load_x + op_inx + op_dex + op_tax + op_tsx,
            new_x,
            mux(
                op_load_y + op_iny + op_dey + op_tay,
                new_y,
                0,
            ),
        ),
    )
    nz_alu = persist(nz_alu, name="nz_alu")
    n_alu = stepglu(one, nz_alu - 128)
    z_alu = 1 - stepglu(one, nz_alu - 1)

    # For memory-modifying operations (inc/dec/shifts), NZ comes from the result
    nz_mem = mux(
        op_inc, inc_result,
        mux(op_dec, dec_result,
            mux(op_asl, asl_result,
                mux(op_lsr, lsr_result,
                    mux(op_rol, rol_result,
                        mux(op_ror, ror_result, 0))))))
    nz_mem = persist(nz_mem, name="nz_mem")
    n_mem = stepglu(one, nz_mem - 128)
    z_mem = 1 - stepglu(one, nz_mem - 1)

    # Branch/flag ops
    bpl_taken = multiply(1 - current_p_n, op_bpl)
    bmi_taken = multiply(current_p_n, op_bmi)
    bvc_taken = multiply(1 - current_p_v, op_bvc)
    bvs_taken = multiply(current_p_v, op_bvs)
    bcc_taken = multiply(1 - current_p_c, op_bcc)
    bcs_taken = multiply(current_p_c, op_bcs)
    bne_taken = multiply(1 - current_p_z, op_bne)
    beq_taken = multiply(current_p_z, op_beq)
    branch_taken = (
        bpl_taken + bmi_taken + bvc_taken + bvs_taken
        + bcc_taken + bcs_taken + bne_taken + beq_taken
    )
    branch_taken = persist(branch_taken, name="branch_taken")

    # Carry flag updates
    carry_op = (
        op_adc + op_sbc + op_cmp + op_cpx + op_cpy + op_asl + op_lsr + op_rol + op_ror
        + op_clc + op_sec + op_dcp + op_slo + op_rla + op_sre + op_rra + op_isb
    )
    new_p_c = (
        multiply(current_p_c, 1 - carry_op)
        + multiply(op_adc, decimal_adc_carry)
        + multiply(op_sbc, decimal_sbc_carry)
        + multiply(op_cmp, cmp_carry)
        + multiply(op_cpx, cpx_carry)
        + multiply(op_cpy, cpy_carry)
        + multiply(op_asl, asl_carry)
        + multiply(op_lsr, lsr_carry)
        + multiply(op_rol, rol_carry)
        + multiply(op_ror, ror_carry)
        + multiply(op_clc, 0)
        + multiply(op_sec, 1)
        + multiply(op_dcp, dcp_carry)
        + multiply(op_slo, asl_carry)
        + multiply(op_rla, rol_carry)
        + multiply(op_sre, lsr_carry)
        + multiply(op_rra, rra_carry)
        + multiply(op_isb, isb_carry)
    )
    new_p_c = persist(new_p_c, name="new_p_c")

    # Zero flag updates
    writes_z = (
        op_load_a + op_load_x + op_load_y + op_adc + op_sbc + op_and + op_ora + op_eor
        + op_pla + op_inx + op_iny + op_dex + op_dey + op_tax + op_tay + op_txa + op_tya + op_tsx
        + op_inc + op_dec + op_asl + op_lsr + op_rol + op_ror
        + op_lax + op_slo + op_rla + op_sre + op_rra + op_isb
    )
    new_p_z = mux(
        writes_z,
        mux(op_inc + op_dec + op_asl + op_lsr + op_rol + op_ror, z_mem, z_alu),
        current_p_z,
    )
    new_p_z = mux(op_cmp, cmp_z, mux(op_cpx, cpx_z, mux(op_cpy, cpy_z, mux(op_dcp, dcp_z, new_p_z))))
    new_p_z = persist(new_p_z, name="new_p_z")

    # Negative flag updates
    new_p_n = mux(
        writes_z,
        mux(op_inc + op_dec + op_asl + op_lsr + op_rol + op_ror, n_mem, n_alu),
        current_p_n,
    )
    new_p_n = mux(op_cmp, cmp_n, mux(op_cpx, cpx_n, mux(op_cpy, cpy_n, mux(op_dcp, dcp_n, new_p_n))))
    new_p_n = persist(new_p_n, name="new_p_n")

    # Overflow flag updates
    writes_v = op_adc + op_sbc + op_bit + op_clv + op_rra + op_isb
    new_p_v = mux(
        writes_v,
        mux(op_adc, adc_overflow,
            mux(op_sbc, sbc_overflow,
                mux(op_bit, bit_v,
                    mux(op_rra, rra_overflow,
                        mux(op_isb, isb_overflow,
                            mux(op_clv, 0, current_p_v)))))),
        current_p_v,
    )
    new_p_v = persist(new_p_v, name="new_p_v")

    # Decimal flag
    new_p_d = mux(op_cld, 0, mux(op_sed, 1, current_p_d))
    new_p_d = persist(new_p_d, name="new_p_d")

    # Interrupt flag
    new_p_i = mux(op_cli, 0, mux(op_sei + op_brk, 1, current_p_i))
    new_p_i = persist(new_p_i, name="new_p_i")

    # Break flag (cleared by PLP/RTI; PHP pushes B=1 but does not set live P)
    new_p_b = mux(op_plp + op_rti, 0, current_p_b)
    new_p_b = persist(new_p_b, name="new_p_b")

    # Unused flag always 1
    new_p_u = one

    # BIT special: N and V come from operand, Z from A & operand
    new_p_n = mux(op_bit, bit_n, new_p_n)
    new_p_z = mux(op_bit, bit_z, new_p_z)
    new_p_n = persist(new_p_n, name="new_p_n_bit")
    new_p_z = persist(new_p_z, name="new_p_z_bit")

    # PLP/RTI update all flags from the pulled byte (B cleared, U set)
    op_pull_p = op_plp + op_rti
    pull_bits = pulled_flag_bits(stack_pull_byte, "pullp")
    new_p_c = mux(op_pull_p, pull_bits[FLAG_C], new_p_c)
    new_p_z = mux(op_pull_p, pull_bits[FLAG_Z], new_p_z)
    new_p_i = mux(op_pull_p, pull_bits[FLAG_I], new_p_i)
    new_p_d = mux(op_pull_p, pull_bits[FLAG_D], new_p_d)
    new_p_v = mux(op_pull_p, pull_bits[FLAG_V], new_p_v)
    new_p_n = mux(op_pull_p, pull_bits[FLAG_N], new_p_n)
    # NMI: the interrupted instruction's flag effects are discarded; the
    # sequence only sets I (NMOS interrupts do not touch D).  These muxes
    # come last so they also override the PLP/RTI pull path above.
    new_p_c = mux(is_int, current_p_c, new_p_c)
    new_p_z = mux(is_int, current_p_z, new_p_z)
    new_p_i = mux(is_int, 1, new_p_i)
    new_p_d = mux(is_int, current_p_d, new_p_d)
    new_p_b = mux(is_int, current_p_b, new_p_b)
    new_p_v = mux(is_int, current_p_v, new_p_v)
    new_p_n = mux(is_int, current_p_n, new_p_n)
    new_p_c = persist(new_p_c, name="new_p_c_final")
    new_p_z = persist(new_p_z, name="new_p_z_final")
    new_p_i = persist(new_p_i, name="new_p_i_final")
    new_p_d = persist(new_p_d, name="new_p_d_final")
    new_p_b = persist(new_p_b, name="new_p_b_final")
    new_p_v = persist(new_p_v, name="new_p_v_final")
    new_p_n = persist(new_p_n, name="new_p_n_final")

    new_p = (
        FLAG_C * new_p_c
        + FLAG_Z * new_p_z
        + FLAG_I * new_p_i
        + FLAG_D * new_p_d
        + FLAG_B * new_p_b
        + FLAG_U * new_p_u
        + FLAG_V * new_p_v
        + FLAG_N * new_p_n
    )
    new_p = persist(new_p, name="new_p")

    # ── PC update ───────────────────────────────────────────────────
    next_pc = pc_after
    next_pc = mux(op_jmp + op_jsr, ea, next_pc)
    next_pc = mux(op_rts, wrap16(pulled_word + 1), next_pc)
    next_pc = mux(op_rti, rti_return, next_pc)
    next_pc = mux(op_brk, irq_vector, next_pc)
    next_pc = mux(branch_taken, branch_target, next_pc)
    # NMI vectors through $FFFA/$FFFB (read into irq_vector above); this mux
    # is last so it overrides whatever the interrupted instruction decoded to.
    next_pc = mux(is_int, irq_vector, next_pc)
    next_pc = persist(next_pc, name="next_pc")

    # ── Primary memory write for this instruction ─────────────────────
    # Accumulator mode occurs only on the four shift opcodes, so subtracting
    # is_acc leaves exactly the memory-mode read-modify-write shifts.
    shifts_mem = op_asl + op_lsr + op_rol + op_ror - is_acc
    unofficial_store = op_sax + op_dcp + op_isb + op_slo + op_rla + op_sre + op_rra
    is_store = (
        op_store_a + op_store_x + op_store_y + op_inc + op_dec + shifts_mem
        + op_pha + op_php + op_jsr + op_brk + unofficial_store
    )
    is_store = mux(is_int, 1, is_store)
    is_store = persist(is_store, name="is_store")

    write_addr = mux(
        op_store_a + op_store_x + op_store_y + op_inc + op_dec + shifts_mem + unofficial_store,
        ea,
        mux(
            op_pha + op_php + op_jsr + op_brk,
            wrap16(0x0100 + current_s),
            0,
        ),
    )
    write_addr = mux(is_int, 0x0100 + current_s, write_addr)
    write_addr = persist(write_addr, name="write_addr_out")

    # JSR and BRK both push PC+2 (JSR: last byte of the 3-byte instruction;
    # BRK: the byte after its padding byte).  The bytes are derived from the
    # PC bytes so wrap8 stays within its valid domain.
    push_ret_lo = persist(wrap8(current_pc_lo + 2), name="push_ret_lo")
    push_ret_carry = stepglu(one, current_pc_lo - 254)
    push_ret_hi = persist(wrap8(current_pc_hi + push_ret_carry), name="push_ret_hi")

    php_push_value = current_p + FLAG_B * (1 - current_p_b) + FLAG_U * (1 - current_p_u)
    write_value = mux(
        op_store_a, current_a,
        mux(op_store_x, current_x,
            mux(op_store_y, current_y,
                mux(op_inc, inc_result,
                    mux(op_dec, dec_result,
                        mux(op_asl, asl_result,
                            mux(op_lsr, lsr_result,
                                mux(op_rol, rol_result,
                                    mux(op_ror, ror_result,
                                        mux(op_pha, current_a,
                                            mux(op_php, php_push_value,
                                                mux(op_jsr + op_brk, push_ret_hi, 0))))))))))))
    write_value = mux(op_sax, sax_result, write_value)
    write_value = mux(op_dcp, dec_result, write_value)
    write_value = mux(op_isb, inc_result, write_value)
    write_value = mux(op_slo, asl_result, write_value)
    write_value = mux(op_rla, rol_result, write_value)
    write_value = mux(op_sre, lsr_result, write_value)
    write_value = mux(op_rra, ror_result, write_value)
    # NMI pushes PCH first (the interrupted PC, not PC+2 as BRK pushes).
    write_value = mux(is_int, current_pc_hi, write_value)
    write_value = persist(write_value, name="write_value_out")

    # Second and third stack pushes (JSR pushes 2 bytes, BRK and NMI push 3).
    # The push addresses are shared: NMI uses the same S-1/S-2 slots as
    # JSR/BRK, so only the gates and pushed values need is_int muxes.
    is_store2 = persist(mux(is_int, 1, op_jsr + op_brk), name="is_store2")
    write_addr2 = persist(0x0100 + wrap8(current_s - 1), name="write_addr2_out")
    write_value2 = persist(mux(is_int, current_pc_lo, push_ret_lo), name="write_value2_out")

    # NMI/IRQ push P with B clear and U set (BRK/PHP push with B set).
    int_push_p = current_p - FLAG_B * current_p_b + FLAG_U * (1 - current_p_u)
    is_store3 = persist(mux(is_int, 1, op_brk), name="is_store3")
    write_addr3 = persist(0x0100 + wrap8(current_s - 2), name="write_addr3_out")
    write_value3 = persist(mux(is_int, int_push_p, php_push_value), name="write_value3_out")

    # ── Trace-context outputs (reference-aligned, 2026-07-09) ────────
    # The trace reports exactly what reference.py reports, field for field:
    # operands only when the instruction fetches them (raw on an NMI step,
    # which reports the not-executed instruction bytes), and data as the
    # operation's *primary result* -- the value a load loads, a store
    # stores, an RMW writes, a compare computes -- 0 for pure control
    # flow.  Previously these reported the raw pc+1/pc+2/[ea] reads, which
    # kept those memory lookups alive in every routed expert even when the
    # instruction never used them.
    has_operand_lo = length - 1  # 0 for 1-byte instructions, else >= 1
    has_operand_hi = is_abs + is_abx + is_aby + is_ind
    report_operand_lo = stepglu(fetched_operand_lo, has_operand_lo + is_int - 1)
    report_operand_lo = persist(report_operand_lo, name="fetched_operand_lo")
    report_operand_hi = stepglu(fetched_operand_hi, has_operand_hi + is_int - 1)
    report_operand_hi = persist(report_operand_hi, name="fetched_operand_hi")

    # The operation-family gates are a mutually exclusive one-hot, so data
    # is a *parallel* gated sum (depth 2), not a mux chain.  Families with
    # data == 0 (jmp/jsr/rts/rti/brk, branches, flag ops, nop) are simply
    # absent.  Shift families report the shift result in both accumulator
    # and memory modes, exactly as the reference handlers return it.
    data_ref = (
        multiply(op_store_a + op_pha, current_a)
        + multiply(op_store_x, current_x)
        + multiply(op_store_y, current_y)
        + multiply(op_php, current_p)
        + multiply(op_inc + op_isb, inc_result)
        + multiply(op_dec + op_dcp, dec_result)
        + multiply(op_asl + op_slo, asl_result)
        + multiply(op_lsr + op_sre, lsr_result)
        + multiply(op_rol + op_rla, rol_result)
        + multiply(op_ror + op_rra, ror_result)
        + multiply(op_sax, sax_result)
        + multiply(
            op_load_a + op_lax + op_adc + op_sbc + op_and + op_ora + op_eor
            + op_pla + op_txa + op_tya,
            new_a,
        )
        + multiply(op_load_x + op_inx + op_dex + op_tax + op_tsx, new_x)
        + multiply(op_load_y + op_iny + op_dey + op_tay, new_y)
        + multiply(op_txs, new_s)
        + multiply(op_plp, new_p)
        + multiply(op_cmp, cmp_result)
        + multiply(op_cpx, cpx_result)
        + multiply(op_cpy, cpy_result)
        + multiply(op_bit, operand)
    )
    data_out = mux(is_int, 0, data_ref)
    data_out = persist(data_out, name="data_out")

    # ── Halt ────────────────────────────────────────────────────────
    # BRK halts only if the IRQ vector is 0 (uninitialized memory).  An
    # interrupt never halts, even if the interrupted byte happens to be $00.
    halted_now = multiply(op_brk, match_byte(irq_vector, 0))
    halted_now = multiply(halted_now, 1 - is_int)
    new_halted = current_halted + halted_now
    new_halted = persist(new_halted, name="new_halted")

    # ── Output tokens ───────────────────────────────────────────────
    H = 1e5
    output_tokens: dict[str, Expression] = {}

    output_tokens["STATE"] = H * (1 - new_halted)
    output_tokens["HALT"] = H * new_halted

    auto_name(locals())
    return input_tokens, output_tokens
