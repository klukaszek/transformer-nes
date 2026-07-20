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

inline float reglu_at(threadgroup const float* v, int base) {
    const float g = v[base];
    return g > 0.0f ? g * v[base + 1] : 0.0f;
}

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

template <typename PTR>
inline Trip4 load4(PTR TR, int t) {
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
// The <=3-term tail is branchless load-then-apply: thin expert layers
// put 1-3 terms on a lane, so the old per-term loop was 1-3 SERIAL
// device stalls per walk (a stalled apply blocks the next load's issue
// on an in-order core).  Clamped indices make the extra loads harmless
// duplicates.  probe8 (replay only): pin every trip load to the segment
// head -- apply work runs but the streaming loads collapse to one warm
// line, attributing walk cost to loads vs apply.
#define STREAM_PHASE(TR, NZ, T0, E, DST, SRC)                              \
    if (NZ) {                                                              \
        int t = (T0);                                                      \
        for (; t + 4 <= (E); t += 4) {                                     \
            const Trip4 b = load4((TR), probe8 ? (T0) : t);                \
            STREAM_APPLY(b, DST, SRC)                                      \
        }                                                                  \
        const int rem = (E) - t;                                           \
        if (rem > 0) {                                                     \
            const int ta = probe8 ? (T0) : t;                              \
            const int tb = probe8 ? (T0) : t + (rem > 1);                  \
            const int tc = probe8 ? (T0) : t + 2 * (rem > 2);              \
            const uint pa = as_type<uint>((TR)[2 * ta]);                   \
            const float va = as_type<float>((TR)[2 * ta + 1]);             \
            const uint pb = as_type<uint>((TR)[2 * tb]);                   \
            const float vb = as_type<float>((TR)[2 * tb + 1]);             \
            const uint pc = as_type<uint>((TR)[2 * tc]);                   \
            const float vc = as_type<float>((TR)[2 * tc + 1]);             \
            DST[int(pa >> 16)] += va * SRC(int(pa & 0xffffu));             \
            if (rem > 1)                                                   \
                DST[int(pb >> 16)] += vb * SRC(int(pb & 0xffffu));         \
            if (rem > 2)                                                   \
                DST[int(pc >> 16)] += vc * SRC(int(pc & 0xffffu));         \
        }                                                                  \
        BARRIER();                                                         \
    }

// Branch-free parallel STATE embed: value i is (stg[src] >> sh) & msk
// (src -1 is the constant 1.0 for is_state).  A switch on tid would
// serialize under SIMT; this is one uniform round per lane.
constant int ESRC[31] = {0, 0, 1, 2, 3, 4,
                         5, 5, 5, 5, 5, 5, 5, 5,
                         6, 7, 8, 9, 9, 10, 11, 12, -1,
                         1, 1, 1, 1, 1, 1, 1, 1};
constant int ESH[31] = {0, 8, 0, 0, 0, 0,
                        0, 1, 2, 3, 4, 5, 6, 7,
                        0, 0, 0, 0, 8, 0, 0, 0, 0,
                        7, 6, 5, 4, 3, 2, 1, 0};
constant int EMSK[31] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                         1, 1, 1, 1, 1, 1, 1, 1,
                         0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0,
                         1, 1, 1, 1, 1, 1, 1, 1};

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
    // xa carries the layer's gate/value accumulators at xa[D + slot]:
    // merged layers stream them beside the attention adds, and the FFN
    // walk applies ReGLU at the read site (no act phase).
    threadgroup float x[TG_D];
    threadgroup float xa[TG_D + 2 * TG_LIVE];
    threadgroup float ho[TG_HO];
    threadgroup float rs[TG_REG];
    threadgroup float rv[2 * TG_REG];
    threadgroup float outk[TG_OUT];
    threadgroup float hstage[2 * TG_HEAD];
    threadgroup int rany[TG_REG];
    threadgroup int halted_tg;
    threadgroup int e_tg;
    threadgroup int k_tg;
    threadgroup int found_tg;
    threadgroup int chk_tg;
    // Machine state lives in threadgroup memory for the whole dispatch:
    // its device round trips were unhidden-latency landmarks on every
    // forward's dispatch/embed/commit path.
    threadgroup int stg[13];

    const int NREG = P[3];
    for (int r = tid; r < NREG; r += T) {
        rs[r] = regstate[3 * r];
        rv[2 * r] = regstate[3 * r + 1];
        rv[2 * r + 1] = regstate[3 * r + 2];
        rany[r] = regany[r];
    }
    for (int i = tid; i < 13; i += T) stg[i] = st[i];
    if (tid == 0) halted_tg = 0;
    BARRIER();

    const int max_steps = io[0], do_trace = io[1];
    int step = 0;
    int fwds = 0;
    int lays = 0;
    int low_fwds = 0;
    int seq_cache = -1;
    int seq_head = 0, seq_k = 0, seq_e = 0, seq_len = 0;
    int seq_guard = 0, seq_guard_min = 0;
    int seq_span_lo = 0, seq_span_len = 0;
    int seq_spec = -1, cur_spec = -1;
    int seq_pguard = 0;
    while (step < max_steps) {
        // ── dispatch + STATE embed.  Sequence windows first (pc == head,
        // bytes re-verified from RAM, whole window inside the step
        // budget): one forward retires K instructions.  Else the routed
        // opcode expert: fetched byte == the graph's memory read at pc
        // (unwritten reads mask to 0).  NMI runs the union.  EVERY lane
        // runs the fetch chain eagerly (uniform loads) while lane 0
        // checks the cached window: the common no-candidate path costs
        // ONE barrier and no lane-0 serial chain. ──
        const int pc = stg[0];
        int e_op = 0, scan_cand = 0;
        if (stg[12] == 0) {
            const int opc = (wr[pc] != 0.0f) ? (int)ram[pc] : 0;
            e_op = (I + P[16])[opc];
            scan_cand = P[2] ? (I + P[18])[pc] : 0;
        }
        cur_spec = -1;
        if (tid == 0) {
            int e = 0, k = 1, found = 0;
            if (stg[12] == 0) {
                if (seq_cache >= 0 && seq_head != pc)
                    seq_cache = -1;
                const int cached_counter =
                    seq_guard == SEQ_GUARD_X ? stg[2]
                    : (seq_guard == SEQ_GUARD_Y ? stg[3] : 0);
                bool cached_guard_ok = seq_guard == 0
                    || cached_counter == 0 || cached_counter >= seq_guard_min;
                if (seq_guard == SEQ_GUARD_S)
                    cached_guard_ok = stg[4] == seq_guard_min;
                if (seq_pguard && (stg[5] & (seq_pguard >> 8))
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
            } else {
                found = 1;  // NMI runs the union (e = 0, k = 1)
            }
            e_tg = e;
            k_tg = k;
            found_tg = found;
        }
        BARRIER();
        int e_route = e_op, k_route = 1;
        if (found_tg) {
            e_route = e_tg;
            k_route = k_tg;
        } else if (scan_cand) {
            // Cold-path candidate scan, uniform on ALL lanes: guards
            // recompute redundantly from uniform device/state reads, and
            // the byte checks -- the measured dispatch wall as serial
            // lane-0 load chains -- split across lanes (chk_tg reduces).
            // All control flow here is uniform, so barriers are legal.
            int found = 0;
            {
                const int sb = (scan_cand >> 8) - 1;
                const int se = sb + (scan_cand & 0xFF);
                for (int s = sb; !found && s < se; ++s) {
                    device const int* SQ = I + P[17] + SEQ_ROW * s;
                    if (step + SQ[1] > max_steps) continue;
                    const int counter =
                        SQ[6] == SEQ_GUARD_X ? stg[2]
                        : (SQ[6] == SEQ_GUARD_Y ? stg[3] : 0);
                    bool guard_ok = SQ[6] == 0
                        || counter == 0 || counter >= SQ[7];
                    if (SQ[6] == SEQ_GUARD_S)
                        guard_ok = stg[4] == SQ[7];
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
                    if (guard_ok && SQ[11] && (stg[5] & (SQ[11] >> 8))
                            != (SQ[11] & 0xFF))
                        guard_ok = false;
                    if (!guard_ok)
                        continue;
                    if (tid == 0) chk_tg = 0;
                    BARRIER();
                    device const int* wb = I + SQ[4];
                    bool myok = true;
                    for (int j = tid; j < SQ[5]; j += T) {
                        const int check = wb[j];
                        const int a = (pc + (check >> 8)) & 0xFFFF;
                        const int b = wr[a] != 0.0f ? (int)ram[a] : 0;
                        if (b != (check & 0xFF)) myok = false;
                    }
                    if (!myok) chk_tg = 1;
                    BARRIER();
                    if (chk_tg) continue;
                    found = 1;
                    cur_spec = SQ[10];
                    e_route = SQ[2];
                    k_route = SQ[1];
                    if (tid == 0) {
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
                        seq_pguard = SQ[11];
                    }
                }
            }
        }
        device const int* XD = I + P[15] + 12 * e_route;
        const int lay0 = XD[0], NL = XD[1], D = XD[2];
#ifdef LOWERED_MODELS
        const int low_m = lowered_model_of(e_route);
        const bool low_route = low_m >= 0
            && !(do_trace >= 2 && do_trace < 5);
#else
        const bool low_route = false;
#endif

        // Embed on ALL lanes (the 31 SL loads were a serial lane-0
        // device chain): one branch-free table round per lane.  Lowered
        // routes skip the zero wave and embed entirely -- their bodies
        // read stg[] directly and never touch x.
        if (!low_route) {
            for (int i = tid; i < TG_D; i += T) x[i] = 0.0f;
            if (tid < 31) {
                const int s = ESRC[tid];
                const float v = s < 0
                    ? 1.0f : (float)((stg[s] >> ESH[tid]) & EMSK[tid]);
                x[(I + XD[4])[tid]] = v;
                if (tid == 0) x[XD[3]] = 1.0f;
            }
            BARRIER();
        }

        // Probe modes: 2 = null model (no layers), 3 = barriers only.
        for (int li = 0; li < (do_trace == 3 ? P[1] : 0); ++li) {
            BARRIER(); BARRIER(); BARRIER();
            BARRIER(); BARRIER(); BARRIER();
        }
        // ── layers (the routed model's range).  Probes 5-7 replay the
        // stack once more AFTER the gather (results discarded, real
        // trajectory kept): the adjacent bench delta isolates pure
        // layer-stack cost.  6 skips lookups on the replay, 7 skips the
        // stream walks -- deltas vs 5 attribute the layer cost. ──
#ifdef LOWERED_MODELS
        float ss_low = 0.0f, hs_low = 0.0f;
#endif
        for (int pass = 0; ; ++pass) {
        const int skip_lu = pass && do_trace == 6;
        const int skip_walk = pass && do_trace == 7;
        const int probe8 = pass && do_trace == 8;
#ifdef LOWERED_MODELS
        // Lowered routes (scalargen.py): the WHOLE forward as scalar
        // expressions on lane 0 -- no zero/embed wave, no walks, no
        // barriers; outk and the head sums come straight from the body.
        // The route is uniform, so the branch is too.  Probes 2-4 skip
        // layers (interpreter path); replay passes rerun the body
        // verbatim -- inputs are unchanged until commit, so the stores
        // rewrite identical values.
        if (low_route) {
            if (pass == 0) low_fwds += 1;
            if (tid == 0)
                lowered_stack(low_m, outk, stg, ram, wr, ss_low, hs_low);
        } else {
#endif
        for (int li = lay0;
             li < (do_trace >= 2 && do_trace < 5 ? lay0 : lay0 + NL);
             ++li) {
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

            // phase A: xa = x, acc slots zeroed; lookups fill ho from x.
            for (int i = tid; i < D; i += T) xa[i] = x[i];
            for (int i = tid; i < 2 * n_live; i += T) xa[D + i] = 0.0f;
            if (n_lookup && !skip_lu) {
                for (int mi = tid; mi < nmem; mi += T) {
                    const int row = mem0 + mi;
                    device const int* MM = I + P[14] + 4 * row;
                    // MM[3] >= 0: constant-address read, no query walk.
                    int qi = MM[3];
                    if (qi < 0) {
                        device const int* mp = I + P[10];
                        const float q = csr_dot(I + P[11], F + P[12],
                                                mp[row], mp[row + 1], x);
                        qi = (int)rint(q);
                    }
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
            STREAM_PHASE(TRA, a_nz && !skip_walk, a_t0, a_e, xa, SRC_ATTN)

            // phase C: x = xa.  An unmerged gate walk shares this wave --
            // the copy writes x, the walk touches only xa -- so the wave
            // ends at the walk's own barrier.
            for (int i = tid; i < D; i += T) x[i] = xa[i];
            if (!g_nz) { BARRIER(); }
#define SRC_GATE(c) (xa[(c)])
            STREAM_PHASE(TRG, g_nz && !skip_walk, g_t0, g_e, xa, SRC_GATE)

            // phase D: FFN adds; ReGLU applied at the read site.
#define SRC_FFN(c) ((c) < D ? xa[(c)] : reglu_at(xa, D + 2 * ((c) - D)))
            STREAM_PHASE(TRF, f_nz && !skip_walk, f_t0, f_e, x, SRC_FFN)
        }

        // ── readout, writes, next state (K rows for windows).  The
        // slot gather and the state/halt head products run on all lanes
        // in one wave -- RO loads and the head dots were the last serial
        // lane-0 device chains -- then lane 0 commits in step order.
        // Speculated interior branches (cur_spec >= 0): each verified
        // step's next_pc readout must equal the trace's assumed pc; on
        // the first violation only the exact prefix commits -- st takes
        // that step's state, later rows/writes never apply. ──
        if (pass == 0) {
            device const int* RO = I + XD[5];
            for (int i = tid; i < 22 * k_route; i += T)
                outk[i] = rint(x[RO[i]]);
            for (int i = tid; i < XD[8]; i += T)
                hstage[i] = (F + XD[7])[i] * x[(I + XD[6])[i]];
            for (int i = tid; i < XD[11]; i += T)
                hstage[TG_HEAD + i] = (F + XD[10])[i] * x[(I + XD[9])[i]];
        }
        BARRIER();
#ifdef LOWERED_MODELS
        }
#endif
        if (do_trace < 5 || pass == 1) break;
        }
        if (tid == 0) {
            device const int* SP = cur_spec >= 0 ? I + cur_spec : 0;
            int committed = k_route;
            for (int k = 0; k < k_route; ++k) {
                if (do_trace == 1) {
                    device int* tr = trace + 22 * (step + k);
                    for (int i = 0; i < 22; ++i)
                        tr[i] = (int)outk[22 * k + i];
                }
                // Only the write triples (and a speculated next_pc) are
                // read per step; the final state comes from the last
                // committed row below.
                const int wf[3][3] = {{13, 14, 15}, {16, 17, 18},
                                      {19, 20, 21}};
                for (int w = 0; w < 3; ++w) {
                    if ((int)outk[22 * k + wf[w][0]] & 0xFF) {
                        const int addr = (int)outk[22 * k + wf[w][1]] & 0xFFFF;
                        ram[addr] = (float)((int)outk[22 * k + wf[w][2]] & 0xFF);
                        wr[addr] = 1.0f;
                        if (seq_cache >= 0 && addr >= seq_span_lo
                                && addr < seq_span_lo + seq_span_len)
                            seq_cache = -1;
                    }
                }
                if (SP && k + 1 < k_route && SP[k] >= 0
                        && ((int)outk[22 * k + 1] & 0xFFFF) != SP[k]) {
                    committed = k + 1;
                    break;
                }
            }
            float ss = 0.0f, hs = 0.0f;
            if (committed == k_route) {
#ifdef LOWERED_MODELS
                if (low_route) { ss = ss_low; hs = hs_low; } else {
#endif
                for (int i = 0; i < XD[8]; ++i) ss += hstage[i];
                for (int i = 0; i < XD[11]; ++i) hs += hstage[TG_HEAD + i];
#ifdef LOWERED_MODELS
                }
#endif
            }
            k_tg = committed;
            threadgroup const float* fin = outk + 22 * (committed - 1);
            stg[0] = (int)fin[1] & 0xFFFF;
            stg[1] = (int)fin[2] & 0xFF;
            stg[2] = (int)fin[3] & 0xFF;
            stg[3] = (int)fin[4] & 0xFF;
            stg[4] = (int)fin[5] & 0xFF;
            stg[5] = (int)fin[6] & 0xFF;
            stg[6] = (int)fin[8] & 0xFF;
            stg[7] = (int)fin[9] & 0xFF;
            stg[8] = (int)fin[10] & 0xFF;
            stg[9] = (int)fin[11] & 0xFFFF;
            stg[10] = (int)fin[12] & 0xFF;
            stg[11] = (int)fin[7] & 0xFF;
            // Probe modes skip the layers, so the readout is garbage;
            // a garbage halt would end the timing loop at step one.
            if ((stg[11] || hs > ss) && do_trace < 2) halted_tg = 1;
        }
        DBARRIER();
        step += k_tg;
        fwds += 1;
        lays += NL;
        if (halted_tg) break;
    }

    for (int r = tid; r < NREG; r += T) {
        regstate[3 * r] = rs[r];
        regstate[3 * r + 1] = rv[2 * r];
        regstate[3 * r + 2] = rv[2 * r + 1];
        regany[r] = rany[r];
    }
    for (int i = tid; i < 13; i += T) st[i] = stg[i];
    if (tid == 0) {
        io[2] = step;
        io[3] = halted_tg;
        io[4] = fwds;
        io[5] = lays;
        io[6] = low_fwds;
    }
}
