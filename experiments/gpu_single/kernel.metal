// The 6502 transformer as one persistent Metal kernel: a single
// threadgroup (ideally one simdgroup) runs whole instructions in a loop
// -- opcode dispatch, STATE embed, the routed model's layers, readout,
// RAM writes, next state -- returning only at the step budget or halt.
//
// Expert routing: each step reads the fetched byte ram[st_pc] (masked by
// the written bit, exactly the graph's memory read) and runs that
// opcode's rebuilt expert model; NMI and non-expert bytes run the union
// (descriptor 0).  A descriptor is 12 ints: layer range, d_model,
// one_slot, embed/readout slot tables, STATE/HALT head rows.  Expert
// memory heads are the folds' fused 64K-span reads (group base -1).
//
// The linear algebra is pre-scheduled at pack time into per-lane sparse
// streams (packed 16-bit dst/col pair plus coeff, 8 bytes per term), whole
// destination rows per lane (no races), loads balanced.  Streams are
// walked in batches of four with every load issued before the first use:
// Apple GPUs are in-order, so an unbatched load->use loop stalls a full
// memory latency per term -- the measured wall of the CSR variants.
// Extended columns: [0,D) reads the residual, [D,..) reads ho
// (attention) or act (FFN).  BARRIER()/DBARRIER()/TG_* come from the
// Python driver (simdgroup barriers when threads <= 32).

#include <metal_stdlib>
using namespace metal;

inline float csr_dot(device const int* cols, device const float* vals,
                     int b, int e, threadgroup const float* v) {
    float a = 0.0f;
    for (int i = b; i < e; ++i) a += vals[i] * v[cols[i]];
    return a;
}

struct Trip4 {
    int d0, c0, d1, c1, d2, c2, d3, c3;
    float v0, v1, v2, v3;
};

inline Trip4 load4(device const int* TR, int t) {
    Trip4 b;
    const uint p0 = as_type<uint>(TR[2 * t]);
    const uint p1 = as_type<uint>(TR[2 * t + 2]);
    const uint p2 = as_type<uint>(TR[2 * t + 4]);
    const uint p3 = as_type<uint>(TR[2 * t + 6]);
    b.d0 = int(p0 >> 16); b.c0 = int(p0 & 0xffffu);
    b.d1 = int(p1 >> 16); b.c1 = int(p1 & 0xffffu);
    b.d2 = int(p2 >> 16); b.c2 = int(p2 & 0xffffu);
    b.d3 = int(p3 >> 16); b.c3 = int(p3 & 0xffffu);
    b.v0 = as_type<float>(TR[2 * t + 1]);
    b.v1 = as_type<float>(TR[2 * t + 3]);
    b.v2 = as_type<float>(TR[2 * t + 5]);
    b.v3 = as_type<float>(TR[2 * t + 7]);
    return b;
}

#define STREAM_APPLY(B, DST, SRC)                                          \
            DST[B.d0] += B.v0 * SRC(B.c0);                                 \
            DST[B.d1] += B.v1 * SRC(B.c1);                                 \
            DST[B.d2] += B.v2 * SRC(B.c2);                                 \
            DST[B.d3] += B.v3 * SRC(B.c3);

// Seg bounds and the first sparse batch are preloaded at the layer top:
// every table address is a pure function of li, so those loads fly while
// the layer's first phase computes (an in-order core doesn't block on a
// load until first use).  Inside the walk, batch n+1's loads issue
// before batch n applies -- the load->use stall pays once per phase at
// most, not once per batch.
#define STREAM_PHASE(TR, NZ, T0, E, DST, SRC)                              \
    if (NZ) {                                                              \
        int t = (T0);                                                      \
        for (; t + 4 <= (E); t += 4) {                                     \
            const Trip4 b = load4((TR), t);                                \
            STREAM_APPLY(b, DST, SRC)                                      \
        }                                                                  \
        for (; t < (E); ++t) {                                             \
            const uint p = as_type<uint>((TR)[2 * t]);                      \
            const int d = int(p >> 16), c = int(p & 0xffffu);              \
            DST[d] += as_type<float>((TR)[2 * t + 1]) * SRC(c);            \
        }                                                                  \
        BARRIER();                                                         \
    }

kernel void run6502(
    constant int* P              [[buffer(0)]],
    device const int* I          [[buffer(1)]],
    device const float* F        [[buffer(2)]],
    device float* ram            [[buffer(3)]],
    device float* wr             [[buffer(4)]],
    device float* regstate       [[buffer(5)]],
    device int* regany           [[buffer(6)]],
    device int* st               [[buffer(7)]],
    device int* io               [[buffer(8)]],
    device int* trace            [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint T   [[threads_per_threadgroup]])
{
    threadgroup float x[TG_D];
    threadgroup float xa[TG_D];
    threadgroup float acc[2 * TG_LIVE];
    threadgroup float act[TG_LIVE];
    threadgroup float ho[TG_HO];
    threadgroup float rs[TG_REG];
    threadgroup float rv[2 * TG_REG];
    threadgroup float outk[TG_OUT];
    threadgroup int rany[TG_REG];
    threadgroup int halted_tg;
    threadgroup int e_tg;
    threadgroup int k_tg;

    const int NREG = P[3];
    for (int r = tid; r < NREG; r += T) {
        rs[r] = regstate[3 * r];
        rv[2 * r] = regstate[3 * r + 1];
        rv[2 * r + 1] = regstate[3 * r + 2];
        rany[r] = regany[r];
    }
    if (tid == 0) halted_tg = 0;
    BARRIER();

    const int max_steps = io[0], do_trace = io[1];
    int step = 0;
    int seq_cache = -1;
    int seq_head = 0, seq_k = 0, seq_e = 0, seq_len = 0;
    int seq_guard = 0, seq_guard_min = 0;
    int seq_span_lo = 0, seq_span_len = 0;
    int seq_spec = -1, cur_spec = -1;
    int seq_pguard = 0;
    while (step < max_steps) {
        // ── dispatch + STATE embed ──
        for (int i = tid; i < TG_D; i += T) x[i] = 0.0f;
        if (tid == 0) {
            // Sequence windows first (pc == head, bytes re-verified from
            // RAM, whole window inside the step budget): one forward
            // retires K instructions.  Else the routed opcode expert:
            // fetched byte == the graph's memory read at pc (unwritten
            // reads mask to 0).  NMI runs the union.
            const int pc = st[0];
            int e = 0, k = 1, found = 0;
            cur_spec = -1;
            if (st[12] == 0) {
                if (seq_cache >= 0 && seq_head != pc)
                    seq_cache = -1;
                const int cached_counter =
                    seq_guard == SEQ_GUARD_X ? st[2]
                    : (seq_guard == SEQ_GUARD_Y ? st[3] : 0);
                bool cached_guard_ok = seq_guard == 0
                    || cached_counter == 0 || cached_counter >= seq_guard_min;
                if (seq_guard == SEQ_GUARD_S)
                    cached_guard_ok = st[4] == seq_guard_min;
                if (seq_pguard && (st[5] & (seq_pguard >> 8))
                        != (seq_pguard & 0xFF))
                    cached_guard_ok = false;
                if (seq_guard == SEQ_GUARD_STORE_ABS) {
                    const int alo = seq_head + seq_guard_min;
                    const int lo = wr[alo] != 0.0f ? (int)ram[alo] : 0;
                    const int hi = wr[alo + 1] != 0.0f
                        ? (int)ram[alo + 1] : 0;
                    const int target = lo | (hi << 8);
                    cached_guard_ok = target < 0x2000
                        && (target < seq_head
                            || target >= seq_head + seq_len);
                }
                if (seq_cache >= 0 && seq_head == pc
                        && step + seq_k <= max_steps && cached_guard_ok) {
                    e = seq_e;
                    k = seq_k;
                    cur_spec = seq_spec;
                    found = 1;
                }
                if (!found && P[2]) {
                    const int hr = (I + P[18])[pc];
                    const int sb = (hr >> 8) - 1;
                    const int se = sb + (hr & 0xFF);
                    for (int s = sb; !found && s < se; ++s) {
                        device const int* SQ = I + P[17] + SEQ_ROW * s;
                        if (step + SQ[1] > max_steps) continue;
                        const int counter =
                            SQ[6] == SEQ_GUARD_X ? st[2]
                            : (SQ[6] == SEQ_GUARD_Y ? st[3] : 0);
                        bool guard_ok = SQ[6] == 0
                            || counter == 0 || counter >= SQ[7];
                        if (SQ[6] == SEQ_GUARD_S)
                            guard_ok = st[4] == SQ[7];
                        if (SQ[6] == SEQ_GUARD_STORE_ABS) {
                            const int alo = pc + SQ[7];
                            const int lo = wr[alo] != 0.0f
                                ? (int)ram[alo] : 0;
                            const int hi = wr[alo + 1] != 0.0f
                                ? (int)ram[alo + 1] : 0;
                            const int target = lo | (hi << 8);
                            guard_ok = target < 0x2000
                                && (target < pc || target >= pc + SQ[3]);
                        }
                        if (!guard_ok)
                            continue;
                        if (SQ[11] && (st[5] & (SQ[11] >> 8))
                                != (SQ[11] & 0xFF))
                            continue;
                        device const int* wb = I + SQ[4];
                        bool ok = true;
                        for (int j = 0; j < SQ[5]; ++j) {
                            const int check = wb[j];
                            const int a = (pc + (check >> 8)) & 0xFFFF;
                            const int b = wr[a] != 0.0f
                                ? (int)ram[a] : 0;
                            if (b != (check & 0xFF)) {
                                ok = false; break;
                            }
                        }
                        if (ok) {
                            e = SQ[2];
                            k = SQ[1];
                            found = 1;
                            seq_cache = s;
                            seq_head = SQ[0];
                            seq_k = SQ[1];
                            seq_e = SQ[2];
                            seq_len = SQ[3];
                            seq_guard = SQ[6];
                            seq_guard_min = SQ[7];
                            seq_span_lo = SQ[8];
                            seq_span_len = SQ[9];
                            seq_spec = SQ[10];
                            cur_spec = SQ[10];
                            seq_pguard = SQ[11];
                        }
                    }
                }
                if (!found) {
                    const int opc = (wr[pc] != 0.0f) ? (int)ram[pc] : 0;
                    e = (I + P[16])[opc];
                }
            }
            e_tg = e;
            k_tg = k;
        }
        BARRIER();
        device const int* XD = I + P[15] + 12 * e_tg;
        const int lay0 = XD[0], NL = XD[1], D = XD[2];

        if (tid == 0) {
            const int pc = st[0], a = st[1], xr = st[2], y = st[3];
            const int sp = st[4], p = st[5], ir = st[6], oplo = st[7];
            const int ophi = st[8], ea = st[9], data = st[10];
            const int hl = st[11], nmi = st[12];
            device const int* SL = I + XD[4];
            const float vals[31] = {
                (float)(pc & 0xFF), (float)((pc >> 8) & 0xFF),
                (float)a, (float)xr, (float)y, (float)sp,
                (float)(p & 1), (float)((p >> 1) & 1), (float)((p >> 2) & 1),
                (float)((p >> 3) & 1), (float)((p >> 4) & 1),
                (float)((p >> 5) & 1), (float)((p >> 6) & 1),
                (float)((p >> 7) & 1),
                (float)ir, (float)oplo, (float)ophi,
                (float)(ea & 0xFF), (float)((ea >> 8) & 0xFF),
                (float)data, (float)hl, (float)nmi, 1.0f,
                // Served A bits, MSB first: state carries its own bits.
                (float)((a >> 7) & 1), (float)((a >> 6) & 1),
                (float)((a >> 5) & 1), (float)((a >> 4) & 1),
                (float)((a >> 3) & 1), (float)((a >> 2) & 1),
                (float)((a >> 1) & 1), (float)(a & 1)};
            for (int i = 0; i < 31; ++i) x[SL[i]] = vals[i];
            x[XD[3]] = 1.0f;
        }
        BARRIER();

        // Probe modes: 2 = null model (no layers), 3 = barriers only.
        for (int li = 0; li < (do_trace == 3 ? P[1] : 0); ++li) {
            BARRIER(); BARRIER(); BARRIER();
            BARRIER(); BARRIER(); BARRIER();
        }
        // ── layers (the routed model's range) ──
        for (int li = lay0; li < (do_trace >= 2 ? lay0 : lay0 + NL); ++li) {
            device const int* LT = I + P[13] + 12 * li;
            const int n_lookup = LT[0], n_live = LT[1];
            const int mem0 = LT[2], nmem = LT[3], nreg = LT[4];
            const int reg_c0 = LT[5], reg_c1 = LT[6], reg_gate = LT[7];
            const int reg_v0 = LT[8], reg_pairs = LT[9];
            const int reg_rid = LT[10], reg_ho = LT[11];
            device const int* SGA = I + P[5] + li * (T + 1);
            device const int* SGG = I + P[9] + li * (T + 1);
            device const int* SGF = I + P[7] + li * (T + 1);
            const int a_nz = SGA[T] > SGA[0];
            const int a_t0 = SGA[tid], a_e = SGA[tid + 1];
            const int g_nz = SGG[T] > SGG[0];
            const int g_t0 = SGG[tid], g_e = SGG[tid + 1];
            const int f_nz = SGF[T] > SGF[0];
            const int f_t0 = SGF[tid], f_e = SGF[tid + 1];
            device const int* TRA = I + P[4];
            device const int* TRG = I + P[8];
            device const int* TRF = I + P[6];

            // phase A: xa = x; lookups fill every ho slot from x.
            for (int i = tid; i < D; i += T) xa[i] = x[i];
            if (n_lookup) {
                for (int mi = tid; mi < nmem; mi += T) {
                    const int row = mem0 + mi;
                    device const int* mp = I + P[10];
                    const float q = csr_dot(I + P[11], F + P[12],
                                            mp[row], mp[row + 1], x);
                    device const int* MM = I + P[14] + 3 * row;
                    const int qi = (int)rint(q);
                    const int base = MM[0];
                    // base -1: fused 64K-span read (expert folds) -- qi IS
                    // the address; else the union's 4K group dispatch.
                    float h0 = 0.0f, h1 = 0.0f;
                    if (qi >= 0 && qi < (base < 0 ? 65536 : 4096)) {
                        const int addr = base < 0 ? qi : base + qi;
                        if (wr[addr] != 0.0f) {
                            h0 = ram[addr];
                            h1 = (float)addr;
                        }
                    }
                    ho[MM[1]] = h0;
                    ho[MM[1] + 1] = h1;
                    // Served bits (MSB first): pure functions of the read
                    // byte, computed here so the graph never ladders them.
                    const int nv = MM[2];
                    const int v = (int)h0;
                    for (int j = 0; j + 2 < nv; ++j)
                        ho[MM[1] + 2 + j] = (float)((v >> (7 - j)) & 1);
                }
                for (int ri = tid; ri < nreg; ri += T) {
                    const float s = 2.0f * x[reg_c0] - x[reg_c1];
                    const bool live = x[reg_gate] == 0.0f;
                    const int rid = reg_rid + ri;
                    if (live && (!rany[rid] || s >= rs[rid])) {
                        rany[rid] = 1;
                        rs[rid] = s;
                        rv[2 * rid] = x[reg_v0 + 2 * ri];
                        rv[2 * rid + 1] = ri < reg_pairs
                            ? x[reg_v0 + 2 * ri + 1] : 0.0f;
                    }
                    const float g = rany[rid] ? 1.0f : 0.0f;
                    ho[reg_ho + 2 * ri] = g * rv[2 * rid];
                    ho[reg_ho + 2 * ri + 1] = g * rv[2 * rid + 1];
                }
            }
            BARRIER();

            // phase B: attention adds.
#define SRC_ATTN(c) ((c) < D ? x[(c)] : ho[(c) - D])
            STREAM_PHASE(TRA, a_nz, a_t0, a_e, xa, SRC_ATTN)

            // phase C: x = xa; ReGLU gate/value accumulate, combine.
            for (int i = tid; i < D; i += T) x[i] = xa[i];
            for (int i = tid; i < 2 * n_live; i += T) acc[i] = 0.0f;
            BARRIER();
            if (n_live) {
#define SRC_GATE(c) (xa[(c)])
                STREAM_PHASE(TRG, g_nz, g_t0, g_e, acc, SRC_GATE)
                for (int j = tid; j < n_live; j += T) {
                    const float g = acc[2 * j];
                    act[j] = g > 0.0f ? g * acc[2 * j + 1] : 0.0f;
                }
                BARRIER();
            }

            // phase D: FFN adds.
#define SRC_FFN(c) ((c) < D ? xa[(c)] : act[(c) - D])
            STREAM_PHASE(TRF, f_nz, f_t0, f_e, x, SRC_FFN)
        }

        // ── readout, writes, next state (K rows for windows).  The
        // slot gather runs on all lanes -- the RO loads are the last
        // serial device chains -- then lane 0 commits in step order.
        // Speculated interior branches (cur_spec >= 0): each verified
        // step's next_pc readout must equal the trace's assumed pc; on
        // the first violation only the exact prefix commits -- st takes
        // that step's state, later rows/writes never apply. ──
        {
            device const int* RO = I + XD[5];
            for (int i = tid; i < 22 * k_tg; i += T)
                outk[i] = rint(x[RO[i]]);
        }
        BARRIER();
        if (tid == 0) {
            device const int* SP = cur_spec >= 0 ? I + cur_spec : 0;
            int out[22];
            int committed = k_tg;
            for (int k = 0; k < k_tg; ++k) {
                for (int i = 0; i < 22; ++i)
                    out[i] = (int)outk[22 * k + i];
                if (do_trace == 1) {
                    device int* tr = trace + 22 * (step + k);
                    for (int i = 0; i < 22; ++i) tr[i] = out[i];
                }
                const int wf[3][3] = {{13, 14, 15}, {16, 17, 18},
                                      {19, 20, 21}};
                for (int w = 0; w < 3; ++w) {
                    if (out[wf[w][0]] & 0xFF) {
                        const int addr = out[wf[w][1]] & 0xFFFF;
                        ram[addr] = (float)(out[wf[w][2]] & 0xFF);
                        wr[addr] = 1.0f;
                        if (seq_cache >= 0 && addr >= seq_span_lo
                                && addr < seq_span_lo + seq_span_len)
                            seq_cache = -1;
                    }
                }
                if (SP && k + 1 < k_tg && SP[k] >= 0
                        && (out[1] & 0xFFFF) != SP[k]) {
                    committed = k + 1;
                    break;
                }
            }
            float ss = 0.0f, hs = 0.0f;
            if (committed == k_tg) {
                for (int i = 0; i < XD[8]; ++i)
                    ss += (F + XD[7])[i] * x[(I + XD[6])[i]];
                for (int i = 0; i < XD[11]; ++i)
                    hs += (F + XD[10])[i] * x[(I + XD[9])[i]];
            }
            k_tg = committed;
            st[0] = out[1] & 0xFFFF;
            st[1] = out[2] & 0xFF;
            st[2] = out[3] & 0xFF;
            st[3] = out[4] & 0xFF;
            st[4] = out[5] & 0xFF;
            st[5] = out[6] & 0xFF;
            st[6] = out[8] & 0xFF;
            st[7] = out[9] & 0xFF;
            st[8] = out[10] & 0xFF;
            st[9] = out[11] & 0xFFFF;
            st[10] = out[12] & 0xFF;
            st[11] = out[7] & 0xFF;
            // Probe modes skip the layers, so the readout is garbage;
            // a garbage halt would end the timing loop at step one.
            if ((st[11] || hs > ss) && do_trace < 2) halted_tg = 1;
        }
        DBARRIER();
        step += k_tg;
        if (halted_tg) break;
    }

    for (int r = tid; r < NREG; r += T) {
        regstate[3 * r] = rs[r];
        regstate[3 * r + 1] = rv[2 * r];
        regstate[3 * r + 2] = rv[2 * r + 1];
        regany[r] = rany[r];
    }
    if (tid == 0) {
        io[2] = step;
        io[3] = halted_tg;
    }
}
