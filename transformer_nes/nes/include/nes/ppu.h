#pragma once

// 2C02 catch-up machine: CPU accesses and mapper writes carry absolute dots.
// Per rendered line: slots at 1..256, sprites/horizontal copy at 257,
// vertical copy at 280..304 (pre-render), and prefetch at 321..336.

#include "nes/chr.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace transformer_nes::nes {

using std::array;
using std::int64_t;
using std::optional;
using std::size_t;
using std::span;
using std::uint16_t;
using std::uint8_t;
using std::vector;

constexpr int64_t NEVER = std::numeric_limits<int64_t>::max();

struct PpuEvent {
  enum Kind : uint8_t { A12Rise, ScanlineStart, Idle };
  Kind kind;
  int64_t dot;
};

// Receives PPU bus events. A mapper is the live watcher; a probe consumes
// its private IRQ-effect horizon during a dry run and reports the fire dot.
class PpuWatcher {
public:
  virtual ~PpuWatcher() = default;
  virtual void ppu_event(const PpuEvent&) {}
};

class IrqProbe : public PpuWatcher {
public:
  int64_t fire_dot = -1;
};

class Ppu {
public:
  static constexpr size_t frame_width = 256;
  static constexpr size_t frame_height = 240;
  static constexpr int dots_per_line = 341;
  static constexpr int vblank_line = 241;
  static constexpr int prerender_line = 261;
  static constexpr int64_t frame_dots = 341 * 262;
  using Frame = array<uint8_t, frame_width * frame_height>;

  Ppu(vector<uint8_t> chr_data, bool writable, bool vertical_mirroring);

  // ── Device protocol (registers $2000-$3FFF, mirrored every 8) ──
  // `dot` is the absolute PPU clock of the access; the PPU catches up
  // to it first.

  [[nodiscard]] int read(int address, int64_t dot);
  void write(int address, int value, int64_t dot);
  // A $2002 value that stays valid until lease_expiry (a dot): the next
  // status change a dry run finds before `horizon`.
  [[nodiscard]] optional<int> read_lease(int address, int64_t horizon);
  [[nodiscard]] int64_t lease_expiry() const { return lease_expiry_; }
  // Neural source/stall request; destination state is sampled at delivery.
  [[nodiscard]] array<int, 2> dma_request(int page, int parity) const;
  void oam_dma(span<const uint8_t, 256> page, int64_t dot);

  // ── Clock ──

  void catch_up(int64_t dot);
  [[nodiscard]] int64_t position() const { return r_.pos; }
  [[nodiscard]] int64_t frame_start() const { return r_.frame_start; }
  // Dot of the next frame's line 0, assuming the odd-frame skip is
  // decided by the current rendering state.
  [[nodiscard]] int64_t next_frame_start() const;
  [[nodiscard]] int64_t vblank_dot() const {
    return r_.frame_start + int64_t{vblank_line} * dots_per_line + 1;
  }
  // Dry run from the current position: the dot of the first $2002
  // status change (track_status) or of the probe's IRQ, whichever comes
  // first before `until`; -1 if neither.  Leaves the PPU untouched.
  int64_t preview(int64_t until, IrqProbe* probe, bool track_status);

  [[nodiscard]] bool take_nmi_edge() { return std::exchange(r_.nmi_edge, false); }
  // An NMI enabled while the vblank flag is already set: the CPU takes
  // it after the NEXT instruction, not at the current boundary.
  [[nodiscard]] bool take_delayed_nmi() { return std::exchange(r_.nmi_delayed, false); }
  [[nodiscard]] bool delayed_nmi_pending() const { return r_.nmi_delayed; }
  [[nodiscard]] bool rendering_enabled() const { return (mask & 0x18) != 0; }
  // The last frame whose visible lines finished (copied at line 240).
  [[nodiscard]] const Frame& completed_frame() const { return completed_; }
  // Which render model drew each 8-pixel slot of that frame (32 per line):
  // 0 the full graph, k the k-th folded expert, unrendered_slot otherwise.
  static constexpr uint8_t unrendered_slot = 0xFF;
  using ExpertMap = array<uint8_t, frame_height * 32>;
  [[nodiscard]] const ExpertMap& completed_expert_map() const { return completed_experts_; }

  // Whether a register access since the last call may have moved the
  // mapper-IRQ schedule: fetch tables, sprite size, rendering enable,
  // 8x16 OAM, or an A12 rise driven by a CPU access.
  [[nodiscard]] bool take_schedule_change() { return std::exchange(schedule_changed_, false); }

  // ── Graph memory (ppu2c02/space.py) ──
  //
  // Graph byte image: CHR $0000, nametables $2000, palette $3F00,
  // OAM $4000, alternate CHR $5000, MMC5 whole-CHR $10000.
  // Epochs/dirty chunks track delivery; -1 shadow bytes have not been sent.
  struct GraphMemorySync {
    array<int16_t, 0x1000 + 0x20 + 0x100> shadow; // pages, palette, OAM; -1 unsent
    array<size_t, 16> chr_source;                 // per 1 KiB CHR slot: source offset
    // Written since the last delivery, per 64-byte chunk: of the four
    // nametable source pages, and of CHR (by source offset).
    uint64_t pages_dirty = ~0ULL;
    vector<uint64_t> chr_dirty;
    bool chr_any = false, full_sent = false;
    uint64_t palette = ~0ULL, oam = ~0ULL, chr_map = ~0ULL;
    uint64_t seen = ~0ULL; // memory_epoch_ at the last full delivery
    GraphMemorySync() { reset(); }
    void reset() {
      shadow.fill(-1);
      chr_source.fill(std::numeric_limits<size_t>::max());
      pages_dirty = ~0ULL;
      std::fill(chr_dirty.begin(), chr_dirty.end(), 0);
      chr_any = full_sent = false;
      palette = oam = chr_map = seen = ~0ULL;
    }
  };
  void mark_page(size_t offset) { // offset within the four source pages
    graph_sync_.pages_dirty |= 1ULL << ((offset >> 6) & 63);
    ++memory_epoch_;
  }
  void mark_pages() {
    graph_sync_.pages_dirty = ~0ULL;
    ++memory_epoch_;
  }
  void mark_chr(size_t source) {
    graph_sync_.chr_dirty.resize((chr.size() >> 6) / 64 + 1);
    graph_sync_.chr_dirty[source >> 12] |= 1ULL << ((source >> 6) & 63);
    graph_sync_.chr_any = true;
    ++memory_epoch_;
  }
  // The source page (0-3) each logical nametable reads, packed 2 bits
  // per table (MMC5's $5105 layout; plain mirroring is its special cases).
  [[nodiscard]] int nt_map() const;
  // Whole-CHR bytes the PPU graphs need mapped (MMC5), else 0.
  [[nodiscard]] size_t graph_chr_full() const { return mmc5.active() ? chr.size() : 0; }

  // Token/expert IDs from ppu2c02/machine.py and control.py.
  // The host owns clocks and storage; neural readouts define state/effects.
  struct RenderDispatch {
    array<array<uint8_t, 8>, 2> inner{}; // per background table, fine x
    uint8_t sprites = 0;
    uint8_t blank = 0;
    array<uint8_t, 2> first{}, last{}; // slots 0 and 31, per table
  };
  // The sprite unit's models (ppu2c02/machine.py:sprite_dispatch):
  // evaluation: full, size experts from index 0, remaining-batch experts;
  // overflow: full, size experts; debug fetch: frozen-latch size experts;
  // fetch: full, then experts per size and count 0-8 without CHR latches.
  struct SpriteDispatch {
    array<uint8_t, 5> evaluate{};
    array<uint8_t, 3> overflow{};
    array<uint8_t, 2> debug_fetch{};
    array<uint8_t, 19> fetch{}, fetch_bus{}, fetch_status{};
  };
  struct PpuEngine {
    void* ctx = nullptr;
    void (*write)(void*, int, int) = nullptr;
    void (*step)(void*, int, const int*, int*) = nullptr;
    // The full model on a copy of the engine, every layer's residual
    // stream written out (trace_rows x trace_width ints).
    void (*trace)(void*, const int*, int*) = nullptr;
    int trace_rows = 0, trace_width = 0;
    array<RenderDispatch, 2> dispatch{}; // [CHR latches active]
    SpriteDispatch sprites{};
    int compose_model = 0;
    vector<int> render_status;
    array<array<uint8_t, 8>, 2> registers{}; // [is_write][register]
    void* control_ctx = nullptr;
    void (*control_step)(void*, int, const int*, int*) = nullptr;
    array<int, 6> control_models{};
    array<int, 3> bus_models{};
    int background_span_model = 0;
    array<array<int, 8>, 2> access_bus_models{};
    int view_model = 0;
    int access_delay_model = 0;
    int dma_request_model = 0, dma_transfer_model = 0;
  };
  // The render graph's token layout (ppu2c02/render.py INPUTS, OUTPUTS).
  static constexpr int render_inputs = 43, render_outputs = 31;
  static constexpr int in_bg = 19, in_sprites = 35;
  static constexpr int out_tile = 8, out_v = 16, out_fetched = 20, out_hit = 21, out_latch = 29;
  // The register graph's (ppu2c02/registers.py INPUTS, OUTPUTS).
  static constexpr int access_inputs = 35, access_outputs = 21;
  static constexpr int in_access_banks = 13, in_access_state = 21;
  // Finite event control (ppu2c02/control.py INPUTS, OUTPUTS).
  static constexpr int control_inputs = 23, control_outputs = 15, out_control_skip = 13;
  // The sprite graphs' (ppu2c02/sprites.py EVALUATE_*, FETCH_*).
  static constexpr int evaluate_inputs = 4, evaluate_outputs = 9;
  static constexpr int fetch_inputs = 23, fetch_outputs = 90;
  static constexpr int compose_inputs = 19, compose_outputs = 10;
  static constexpr int in_index = 4, in_latch = 12, in_banks = 15;
  static constexpr int out_sprite_x = 64, out_sprite_fetch = 72, out_sprite_a12 = 80,
                       out_sprite_latch = 88;
  void set_engine(PpuEngine engine);
  // The X-ray's pixel dissection: the next time the render graph draws
  // `slot` of `line`, also trace that slot through the full model.
  struct SlotTrace {
    int line = -1, slot = -1, model = 0;
    vector<int> residual; // trace_rows x trace_width
  };
  void request_slot_trace(int line, int slot) {
    slot_trace_request_ = {line, slot};
    slot_trace_.reset();
  }
  [[nodiscard]] std::optional<SlotTrace> take_slot_trace() { return std::exchange(slot_trace_, {}); }

  void set_watcher(PpuWatcher* watcher, bool wants_a12) {
    watcher_ = watcher;
    wants_a12_ = wants_a12;
  }

  // ── Mapper-side control ──

  void set_chr_banks(const ChrBanks& banks_1k) {
    const size_t count = chr.size() >> 10;
    bool moved = false;
    for (size_t i = 0; i < chr_bank.size(); ++i) {
      const size_t bank = (banks_1k[i] % count) * 0x400;
      moved |= chr_bank[i] != bank;
      chr_bank[i] = bank;
    }
    if (moved)
      chr_map_changed();
  }
  void set_chr_latch_banks(const ChrLatchBanks& banks_4k, bool wide_low) {
    chr_map_changed();
    chr_latch_active = true;
    chr_latch.wide_low = wide_low;
    chr_latch.configure(banks_4k, chr.size());
  }
  void set_mmc5_state(int exram_mode, int nametable_map, int fill_tile, int fill_color,
                      int chr_upper) {
    mmc5.configure(exram_mode, fill_tile, fill_color, chr_upper);
    refresh_mmc5_view();
    mirroring = nametable_map;
    mark_pages();
  }
  void set_mmc5_exram(int offset, int value) {
    mmc5.write_exram(offset, value);
    mark_page(0x800 + static_cast<size_t>(offset & 0x3FF));
  }
  void set_mmc5_chr(const ChrBanks& sprite, const ChrBanks& background) {
    mmc5.configure_chr(sprite, background);
    chr_map_changed();
  }

  // ── State ──

  // Full CHR data (ROM of any 1 KiB-multiple size, or 8 KiB CHR RAM);
  // the PPU's $0000-$1FFF window maps through eight 1 KiB bank slots.
  vector<uint8_t> chr;
  ChrBanks chr_bank{};
  ChrLatch chr_latch;
  bool chr_latch_active = false;
  Mmc5Ppu mmc5;
  bool chr_writable;
  // Nametable arrangement, runtime-switchable by mappers (MMC1
  // encoding): 0/1 single-screen low/high, 2 vertical, 3 horizontal.
  // With MMC5 active it is the $5105 map.
  int mirroring;
  array<uint8_t, 0x800> nametables{};
  array<uint8_t, 0x20> palette{};
  array<uint8_t, 0x100> oam{};

  int ctrl = 0, mask = 0;
  uint8_t oam_addr = 0;
  uint16_t t = 0; // loopy t (15 bits)
  int fine_x = 0;
  bool write_latch = false; // loopy w
  int read_buffer = 0;
  bool sprite_limit = true; // hardware 8-per-scanline cap (debug toggle)
  // Every graph-rendered slot through the full render model, none through
  // folded experts (the X-ray's ablations live in the full model).
  bool force_full_render = false;

private:
  enum class Ev : uint8_t {
    Slot,
    Scanline,
    Sprites,
    Prefetch,
    VCopy,
    OddSkip,
    FrameDone,
    VBlank,
    Clear,
    Evaluate
  };
  struct EventAt {
    int dot;
    Ev ev;
    int slot;
    int run = 1; // consecutive background slots in this immutable timeline
  };

  // Everything rendering mutates: the dry-run preview snapshots and
  // restores exactly this (plus the MMC2 latch).
  struct Render {
    int64_t pos = 0;         // absolute dot the PPU has reached
    int64_t line_start = 0;  // absolute dot of the current line's dot 0
    int64_t frame_start = 0; // absolute dot of the current frame's line 0
    int line = 0;
    int line_len = dots_per_line;
    int next_event = 0; // index into the current line's event table
    bool odd = false;   // frame parity (odd frames may skip a dot)
    uint16_t v = 0;     // loopy v (15 bits)
    bool vblank = false, nmi_edge = false, nmi_delayed = false;
    // $2002 read one dot before the flag sets: no flag, no NMI this frame.
    bool vblank_suppressed = false;
    int64_t vblank_set_dot = -1; // this frame's flag-set dot, once set
    int64_t hit_dot = NEVER;      // sprite-0 hit, visible to reads at >= this dot
    int64_t overflow_dot = NEVER; // sprite overflow, likewise
    bool a12_high = false;
    int64_t a12_low_since = 0;
    // The line's fetched tiles (0-1 prefetched on the prior line) as
    // palette indices, 8 per tile: 0 transparent, else palette*4+value.
    // Colors resolve at output, so mid-line palette writes still land.
    array<uint8_t, 34 * 8 + 8> bg{};
    // Next line's sprite pixels: bits 0-4 palette index (16+palette*4+
    // value), bit 6 behind background, bit 7 sprite 0; 0 = transparent.
    // Padding transports right-edge rows without a native clipping rule.
    array<uint8_t, 256 + 8> sprites{};
    array<uint8_t, 33> sprite_slots{}; // occupied tiles, plus a zero sentinel
#ifdef TRANSFORMER_NES_PPU_REFERENCE
    bool sprite0_on_line = false;
#endif
    // Sprite evaluation (dot 65) for the next line: OAM indices in order.
    array<uint8_t, 64> selected{};
    int selected_count = 0;
    const vector<EventAt>* events = nullptr; // the current line's timeline
  };

  struct RegisterFile {
    int32_t t, v, fine_x, w, ctrl, mask, oam_addr, read_buffer;
  };
  struct RegisterAccess {
    int reg, value;
    bool is_write;
    int status;
    bool rendering;
  };
  struct RegisterResult {
    int read_value = 0;
    bool clear_vblank = false, nmi_rise = false, nmi_fall = false;
  };
  enum class ControlEvent { Access, VerticalCopy, SpriteLine, OddSkip, VBlank, Clear };
  bool control(ControlEvent event, int64_t dot, const RegisterResult& effects);
  int access(int reg, int value, bool is_write, int64_t dot);
  RegisterResult graph_transition(const RegisterAccess& request, const RegisterFile& before);
  void access_effects(const RegisterAccess& request, const RegisterFile& before,
                      const RegisterResult& result, int64_t dot);
  void write_physical(int addr, int value);
  bool rendering_line() const { return r_.line < 240 || r_.line == prerender_line; }
  const size_t* bg_banks() const {
    return mmc5.active() ? mmc5.live_background_banks() : chr_bank.data();
  }
  const size_t* sprite_banks() const {
    return mmc5.active() ? mmc5.sprite_banks() : chr_bank.data();
  }
  int status(int64_t dot) const;

  static const vector<EventAt>& events_for(int line);
  void process(const EventAt& e);
  void advance_line();
  void emit(PpuEvent::Kind kind, int64_t dot);
  enum class BusEvent { Pin, Background, Sprite };
  bool bus_event(BusEvent event, bool high, int64_t dot);
  bool a12(bool high, int64_t dot) { return bus_event(BusEvent::Pin, high, dot); }
  void refresh_mmc5_view();
  bool start_sprite_line();
  void compose_sprite_rows(const int* fetched, int count);
  // The sprite unit as graph steps (engine_.sprites).
  void evaluate_sprites_graph(int64_t dot);
  void load_sprites_graph(int64_t dot);
  void mark_status_change(int64_t dot);

  // Selection is a neural function of line, OAM and control inputs. Share
  // its first batch across live rendering and both kinds of lookahead.
  struct SpriteRows {
    array<array<uint8_t, evaluate_outputs>, frame_height> rows{};
    std::bitset<frame_height> valid;
    uint64_t oam_epoch = ~0ULL;
    int model = -1;
  } sprite_rows_;

  Render r_;
  Frame frame_{};
  Frame completed_{};
  static ExpertMap empty_expert_map() {
    ExpertMap map;
    map.fill(unrendered_slot);
    return map;
  }
  ExpertMap experts_ = empty_expert_map(), completed_experts_ = empty_expert_map();
  PpuWatcher* watcher_ = nullptr;
  bool wants_a12_ = false;
  int64_t lease_expiry_ = NEVER;
  GraphMemorySync graph_sync_;
  PpuEngine engine_;
  std::pair<int, int> slot_trace_request_{-1, -1};
  std::optional<SlotTrace> slot_trace_;
  // Change counts of the graph-memory view's regions.
  // chr_map_epoch_ counts changes to what the CHR windows map (banks,
  // latches, the ctrl bits that pick MMC5's background set).
  // memory_epoch_ counts every change of any kind, so an unchanged
  // delivery costs one compare.
  uint64_t palette_epoch_ = 0, oam_epoch_ = 0, chr_map_epoch_ = 0, memory_epoch_ = 0;
  void palette_changed() {
    ++palette_epoch_;
    ++memory_epoch_;
  }
  void oam_changed() {
    ++oam_epoch_;
    ++memory_epoch_;
  }
  void chr_map_changed() {
    ++chr_map_epoch_;
    ++memory_epoch_;
  }
  void sync_graph_memory();
  int page_byte(int page, int offset) const;
  void graph_step(int model, const int* in, int* out);
  void render_slot(int slot, int64_t dot);
  void background_bus(int64_t dot);
  int preview_background(int count, int64_t dot);
  RegisterFile register_file() const;
  bool schedule_changed_ = false;
#ifdef TRANSFORMER_NES_PPU_REFERENCE
#include "tests/ppu_reference.h"
#endif
  // Dry-run preview state: no frame writes, events go to the probe,
  // and the loop stops at the first status change or probe IRQ.
  bool dry_ = false;
  bool track_status_ = false;
  IrqProbe* probe_ = nullptr;
  int64_t dry_stop_ = -1;
};

} // namespace transformer_nes::nes
