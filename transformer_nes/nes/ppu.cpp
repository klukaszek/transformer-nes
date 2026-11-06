// NES PPU (2C02): loopy registers, tile-slot rendering, and timed events
// on an absolute dot clock.  See ppu.h for the line timeline.

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <tuple>
#include <utility>
#include <vector>

#include "nes/chr.h"
#include "nes/ppu.h"

namespace transformer_nes::nes {

#ifdef TRANSFORMER_NES_PPU_REFERENCE
#include "tests/ppu_reference.inc"
#endif

Ppu::Ppu(vector<uint8_t> chr_data, bool writable, bool vertical_mirroring)
    : chr(std::move(chr_data)), chr_writable(writable), mirroring(vertical_mirroring ? 2 : 3) {
  if (chr.size() < 0x2000)
    chr.resize(0x2000, 0);
  for (size_t i = 0; i < chr_bank.size(); ++i)
    chr_bank[i] = i * 0x400;
}

int Ppu::status(int64_t dot) const {
  return (r_.vblank ? 0x80 : 0) | (r_.hit_dot <= dot ? 0x40 : 0) |
         (r_.overflow_dot <= dot ? 0x20 : 0);
}

// ── Registers ──

int Ppu::read(int addr, int64_t dot) {
  catch_up(dot);
  return access((addr - 0x2000) & 7, 0, false, dot);
}

void Ppu::write(int addr, int value, int64_t dot) {
  const int reg = (addr - 0x2000) & 7;
  // The access circuit supplies its scheduling delay before mutation.
  int delay;
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx)
    delay = reference_write_delay(reg);
  else
#endif
  {
    if (!engine_.ctx)
      throw std::runtime_error("PPU transformer engine is required");
    const int in[17] = {reg, 1};
    engine_.control_step(engine_.control_ctx, engine_.access_delay_model, in, &delay);
  }
  catch_up(dot + delay);
  (void)access(reg, value & 0xFF, true, dot);
}

// A register token computes the register file and memory transition;
// the host then applies effects that depend on the access's dot.
int Ppu::access(int reg, int value, bool is_write, int64_t dot) {
  const RegisterFile before = register_file();
  const RegisterAccess request{reg, value, is_write, status(dot),
                               rendering_enabled() && rendering_line()};
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  const RegisterResult result = engine_.ctx ? graph_transition(request, before) : transition(request);
#else
  const RegisterResult result = graph_transition(request, before);
#endif
  access_effects(request, before, result, dot);
  return result.read_value;
}

// Apply neural timing effects, then deliver mapper-visible A12 edges,
// CHR latches and renderer invalidations.
void Ppu::access_effects(const RegisterAccess& request, const RegisterFile& before,
                         const RegisterResult& result, int64_t dot) {
  if (result.clear_vblank || result.nmi_rise || result.nmi_fall)
    control(ControlEvent::Access, dot, result);
  const int in[]{request.reg, request.is_write, request.rendering, before.w,
                 before.ctrl, before.mask, ctrl, mask, (r_.v >> 12) & 7,
                 before.v & 31, (before.v >> 5) & 31, (before.v >> 10) & 3,
                 (before.v >> 12) & 7, chr_latch_active, chr_latch.wide_low,
                 chr_latch.latch_fe[0], chr_latch.latch_fe[1]};
  int out[7];
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx)
    reference_access_bus(in, out);
  else
#endif
    engine_.control_step(engine_.control_ctx, engine_.access_bus_models[request.is_write][request.reg], in, out);
  if (out[0] && a12(out[1] != 0, dot))
    schedule_changed_ = true;
  chr_latch.latch_fe[0] = out[2] != 0;
  chr_latch.latch_fe[1] = out[3] != 0;
  if (out[4]) {
    refresh_mmc5_view();
    chr_map_changed();
  }
  schedule_changed_ |= out[5] != 0;
}

// Commit the register expert's state and memory-write readouts.
Ppu::RegisterResult Ppu::graph_transition(const RegisterAccess& request,
                                          const RegisterFile& before) {
  const int map = nt_map();
  int in[access_inputs] = {1,
                           request.reg,
                           request.value,
                           request.is_write,
                           request.status,
                           request.rendering,
                           map & 3,
                           (map >> 2) & 3,
                           (map >> 4) & 3,
                           (map >> 6) & 3,
                           chr_writable,
                           chr_latch_active && chr_latch.latch_fe[0],
                           chr_latch_active && chr_latch.latch_fe[1]};
  if (mmc5.active())
    for (int s = 0; s < 8; ++s)
      in[in_access_banks + s] = static_cast<int>(chr_bank[static_cast<size_t>(s)] >> 10);
  int* field = in + in_access_state;
  for (const int32_t loopy : {before.t, before.v}) {
    *field++ = loopy & 0x1F;
    *field++ = (loopy >> 5) & 0x1F;
    *field++ = (loopy >> 10) & 3;
    *field++ = (loopy >> 12) & 7;
  }
  for (const int32_t value :
       {before.fine_x, before.w, before.ctrl, before.mask, before.oam_addr, before.read_buffer})
    *field++ = value;
  int out[access_outputs];
  graph_step(engine_.registers[request.is_write][request.reg], in, out);
  auto join = [](const int* f) { return f[0] | (f[1] << 5) | (f[2] << 10) | (f[3] << 12); };
  t = static_cast<uint16_t>(join(out));
  r_.v = static_cast<uint16_t>(join(out + 4));
  fine_x = out[8];
  write_latch = out[9] != 0;
  ctrl = out[10];
  mask = out[11];
  oam_addr = static_cast<uint8_t>(out[12]);
  read_buffer = out[13];
  if (out[15])
    write_physical(out[16], out[17]);
  return {out[14], out[18] != 0, out[19] != 0, out[20] != 0};
}

// A graph-memory write (the physical space of ppu2c02/space.py) into
// PPU state: CHR RAM through the window, CIRAM pages, MMC5 ExRAM (page
// 2), palette, OAM.  The fill-mode page holds no memory.
void Ppu::write_physical(int addr, int value) {
  const auto byte = static_cast<uint8_t>(value);
  if (addr < 0x2000) {
    if (chr_writable) {
      const size_t source = chr_bank[(addr >> 10) & 7] + (addr & 0x3FF);
      chr[source] = byte;
      mark_chr(source);
    }
  } else if (addr < 0x2800) {
    nametables[static_cast<size_t>(addr - 0x2000)] = byte;
    mark_page(static_cast<size_t>(addr - 0x2000));
  } else if (addr < 0x2C00) {
    mmc5.write_exram(addr - 0x2800, byte);
    mark_page(static_cast<size_t>(addr - 0x2000));
  } else if (addr >= 0x4000) {
    oam_changed();
    oam[static_cast<size_t>(addr - 0x4000)] = byte;
  } else if (addr >= 0x3F00) {
    palette_changed();
    palette[static_cast<size_t>(addr - 0x3F00)] = byte & 0x3F;
  }
}

void Ppu::set_engine(PpuEngine engine) {
  engine_ = engine;
  sprite_rows_.valid.reset();
  graph_sync_.reset(); // the new engine's memory starts empty
  refresh_mmc5_view();
}

int Ppu::nt_map() const {
  if (mmc5.active())
    return mirroring & 0xFF; // the $5105 map itself
  static constexpr int maps[4] = {0x00, 0x55, 0x44, 0x50};
  return maps[mirroring & 3];
}

// A nametable source page's content as the PPU reads it.
int Ppu::page_byte(int page, int offset) const {
  if (page < 2)
    return nametables[static_cast<size_t>(page * 0x400 + offset)];
  return mmc5.active() ? mmc5.page_view(page, offset) : 0;
}

// Deliver the graph-memory view's changes since the engine last saw
// it.  Nametable pages go by written 64-byte chunk, palette and OAM by
// change count; CHR goes by 1 KiB slot, resent when the slot's source
// moves and otherwise by written chunk.  CHR is mapped as the background
// fetches see it (with both latch windows under MMC2/MMC4), and for
// MMC5 the whole CHR ROM, which its sprite and $2007 reads bank into.
void Ppu::sync_graph_memory() {
  GraphMemorySync& sync = graph_sync_;
  if (sync.seen == memory_epoch_)
    return;
  auto write = [this](int addr, int value) { engine_.write(engine_.ctx, addr, value); };
  const bool chr_full = mmc5.active();
  auto deliver = [&](size_t index, int addr, uint8_t value) {
    if (sync.shadow[index] == value)
      return;
    sync.shadow[index] = value;
    write(addr, value);
  };
  for (uint64_t dirty = std::exchange(sync.pages_dirty, 0); dirty; dirty &= dirty - 1)
    for (int a = std::countr_zero(dirty) * 64, end = a + 64; a < end; ++a)
      deliver(static_cast<size_t>(a), 0x2000 + a, static_cast<uint8_t>(page_byte(a >> 10, a & 0x3FF)));
  if (sync.palette != palette_epoch_) {
    sync.palette = palette_epoch_;
    for (int a = 0; a < 0x20; ++a)
      deliver(static_cast<size_t>(0x1000 + a), 0x3F00 + a, palette[static_cast<size_t>(a)]);
  }
  if (sync.oam != oam_epoch_) {
    sync.oam = oam_epoch_;
    for (int a = 0; a < 0x100; ++a)
      deliver(static_cast<size_t>(0x1020 + a), 0x4000 + a, oam[static_cast<size_t>(a)]);
  }
  auto dirty = [&](size_t source) {
    return (sync.chr_dirty[source >> 12] >> ((source >> 6) & 63)) & 1;
  };
  if (sync.chr_any || sync.chr_map != chr_map_epoch_) {
    sync.chr_map = chr_map_epoch_;
    const bool latches = chr_latch_active;
    for (int s = 0; s < (latches ? 16 : 8); ++s) {
      const int slot = s & 7;
      size_t source;
      if (latches)
        source = chr_latch.banks[(slot >> 2) * 2 + (s >> 3)] + (slot & 3) * 0x400;
      else
        source = bg_banks()[slot];
      const int base = (s < 8 ? 0 : 0x5000) + slot * 0x400;
      const bool moved = std::exchange(sync.chr_source[static_cast<size_t>(s)], source) != source;
      if (!moved && !sync.chr_any)
        continue;
      for (int i = 0; i < 0x400; i += 64)
        if (moved || dirty(source + static_cast<size_t>(i)))
          for (int j = i; j < i + 64; ++j)
            write(base + j, chr[source + static_cast<size_t>(j)]);
    }
  }
  if (chr_full && (!sync.full_sent || sync.chr_any))
    for (size_t i = 0; i < chr.size(); i += 64)
      if (!sync.full_sent || (sync.chr_any && dirty(i)))
        for (size_t j = i; j < i + 64; ++j)
          write(0x10000 + static_cast<int>(j), chr[j]);
  sync.full_sent |= chr_full;
  if (std::exchange(sync.chr_any, false))
    std::fill(sync.chr_dirty.begin(), sync.chr_dirty.end(), 0);
  sync.seen = memory_epoch_;
}

// One tile slot through the render graph: slots 0-31 output pixels and
// fetch tile slot+2; 32-33 are the next line's prefetches.
// One token through the PPU engine, its memory first brought up to
// date with the render view.
void Ppu::graph_step(int model, const int* in, int* out) {
  if (!engine_.ctx)
    throw std::logic_error("PPU transformer engine is required");
  sync_graph_memory();
  engine_.step(engine_.ctx, model, in, out);
}

void Ppu::background_bus(int64_t dot) {
  bus_event(BusEvent::Background, false, dot);
}

// Neural run composition retains the first-pair stamp for an early IRQ stop.
int Ppu::preview_background(int count, int64_t dot) {
  if (!rendering_enabled() || !wants_a12_)
    return count;
  const int age = static_cast<int>(std::clamp<int64_t>(dot - r_.a12_low_since, -1, 11));
  const int in[]{r_.a12_high, age, ctrl, mmc5.active() && mmc5.extended_attributes(), count};
  int out[6];
  engine_.control_step(engine_.control_ctx, engine_.background_span_model, in, out);
  r_.a12_high = out[0] != 0;
  if (out[5])
    r_.a12_low_since = dot;
  if (out[2])
    emit(PpuEvent::A12Rise, dot + out[3]);
  if (dry_stop_ >= 0)
    return 1;
  if (out[1])
    r_.a12_low_since = dot + out[4];
  return count;
}

void Ppu::render_slot(int slot, int64_t dot) {
  // IRQ probes observe fetch clocks, not pixels or scroll readouts.
  // The background pattern table is a control pin, independent of CHR.
  if (dry_ && (!track_status_ || r_.hit_dot != NEVER || (mask & 0x18) != 0x18)) {
    if (rendering_enabled())
      background_bus(dot);
    return;
  }
  const bool visible = slot < 32;
  const bool extended = mmc5.active() && mmc5.extended_attributes();
  const int map = nt_map();
  int in[render_inputs] = {1,
                           slot,
                           r_.hit_dot != NEVER,
                           fine_x,
                           ctrl,
                           mask,
                           map & 3,
                           (map >> 2) & 3,
                           (map >> 4) & 3,
                           (map >> 6) & 3,
                           chr_latch_active && chr_latch.latch_fe[0],
                           chr_latch_active && chr_latch.latch_fe[1],
                           chr_latch.wide_low,
                           extended,
                           mmc5.chr_upper(),
                           r_.v & 0x1F,
                           (r_.v >> 5) & 0x1F,
                           (r_.v >> 10) & 3,
                           (r_.v >> 12) & 7};
  const bool sprites = visible && r_.sprite_slots[static_cast<size_t>(slot)];
  if (visible) {
    std::copy_n(r_.bg.data() + slot * 8, 16, in + in_bg);
    if (sprites)
      std::copy_n(r_.sprites.data() + slot * 8, 8, in + in_sprites);
  }
  // A folded expert only when its control pins hold. Disabled rendering
  // still computes palette output through the blank model. Ordinary slots
  // select by pattern table/fine x; unsupported active modes use the union.
  int model = 0;
  if (!(mask & 0x18) && !force_full_render) {
    model = engine_.dispatch[chr_latch_active ? 1 : 0].blank;
  } else if ((mask & 0x08) && !(mask & 0x01) && !extended && !force_full_render) {
    const RenderDispatch& d = engine_.dispatch[chr_latch_active ? 1 : 0];
    const int table = (ctrl & 0x10) ? 1 : 0;
    if (slot == 0 || slot == 31)
      model = sprites ? 0 : slot == 0 ? d.first[table] : d.last[table];
    else
      model = sprites ? d.sprites : d.inner[table][fine_x];
  }
  int out[render_outputs];
  if (dry_ && !force_full_render)
    graph_step(engine_.render_status[static_cast<size_t>(model)], in, out + out_tile);
  else
    graph_step(model, in, out);
  if (engine_.trace && !dry_ && slot_trace_request_ == std::pair{r_.line, slot}) {
    SlotTrace trace{r_.line, slot, model,
                    vector<int>(static_cast<size_t>(engine_.trace_rows * engine_.trace_width))};
    engine_.trace(engine_.ctx, in, trace.residual.data());
    slot_trace_ = std::move(trace);
    slot_trace_request_ = {-1, -1};
  }
  if (visible && r_.line < 240) {
    if (!dry_) {
      experts_[static_cast<size_t>(r_.line * 32 + slot)] = static_cast<uint8_t>(model);
      uint8_t* row = frame_.data() + static_cast<size_t>(r_.line) * 256 + slot * 8;
      for (int i = 0; i < 8; ++i)
        row[i] = static_cast<uint8_t>(out[i]);
    }
    for (int i = 0; i < 8 && r_.hit_dot == NEVER; ++i)
      if (out[out_hit + i]) {
        r_.hit_dot = dot + i;
        mark_status_change(dot + i);
      }
  }
  if (out[out_fetched]) { // the tile, v, the latches, and the fetch's A12 edge
    background_bus(dot);
    const int index = visible ? slot + 2 : slot - 32;
    for (int c = 0; c < 8; ++c)
      r_.bg[static_cast<size_t>(index * 8 + c)] = static_cast<uint8_t>(out[out_tile + c]);
    const int* v = out + out_v;
    r_.v = static_cast<uint16_t>(v[0] | (v[1] << 5) | (v[2] << 10) | (v[3] << 12));
    const bool latch_lo = out[out_latch] != 0, latch_hi = out[out_latch + 1] != 0;
    if (chr_latch_active &&
        (chr_latch.latch_fe[0] != latch_lo || chr_latch.latch_fe[1] != latch_hi)) {
      chr_latch.latch_fe[0] = latch_lo;
      chr_latch.latch_fe[1] = latch_hi;
      chr_map_changed();
    }
  }
}

optional<int> Ppu::read_lease(int addr, int64_t horizon) {
  if (((addr - 0x2000) & 7) != 2 || r_.vblank)
    return std::nullopt;
  const int64_t now = r_.pos;
  int64_t expiry = NEVER;
  if (r_.hit_dot > now)
    expiry = r_.hit_dot;
  if (r_.overflow_dot > now)
    expiry = std::min(expiry, r_.overflow_dot);
  // The machine may finish an instruction or two past its horizon
  // before the console drops leases; look that far too.
  const int64_t until = std::min(horizon + 3 * 16, expiry);
  if (const int64_t change = preview(until, nullptr, true); change >= 0)
    expiry = std::min(expiry, change);
  lease_expiry_ = expiry;
  return status(now);
}

array<int, 2> Ppu::dma_request(int page, int parity) const {
  const int in[]{page, parity, 0, 0};
  array<int, 2> out;
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx) {
    int packet[7];
    reference_dma(in, packet);
    return {packet[0], packet[1]};
  }
#endif
  if (!engine_.control_step)
    throw std::logic_error("PPU transformer engine is required");
  engine_.control_step(engine_.control_ctx, engine_.dma_request_model, in, out.data());
  return out;
}

void Ppu::oam_dma(span<const uint8_t, 256> page, int64_t dot) {
  catch_up(dot);
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx) {
    reference_oam_dma(page);
    return;
  }
#endif
  if (!engine_.control_step)
    throw std::logic_error("PPU transformer engine is required");
  const int in[]{0, 0, oam_addr, ctrl};
  int out[5];
  engine_.control_step(engine_.control_ctx, engine_.dma_transfer_model, in, out);
  std::copy_n(page.begin(), out[1], oam.begin() + out[0]);
  std::copy_n(page.begin() + out[2], out[3], oam.begin());
  oam_changed();
  schedule_changed_ |= out[4] != 0;
}

// ── The register file ──

Ppu::RegisterFile Ppu::register_file() const {
  return {t, r_.v, fine_x, write_latch, ctrl, mask, oam_addr, read_buffer};
}

// ── Clock ──

const vector<Ppu::EventAt>& Ppu::events_for(int line) {
  static const auto tables = [] {
    array<vector<EventAt>, 5> t{};
    auto slots = [](vector<EventAt>& v, bool scanline) {
      for (int k = 0; k < 32; ++k) {
        v.push_back({1 + 8 * k, Ev::Slot, k});
        if (k == 0 && scanline)
          v.push_back({3, Ev::Scanline, 0});
      }
      v.push_back({257, Ev::Sprites, 0});
    };
    slots(t[0], true); // visible
    t[0].insert(t[0].begin() + 9, {65, Ev::Evaluate, 0}); // after slot 7 (dot 57)
    t[0].push_back({321, Ev::Prefetch, 0});
    t[1].push_back({0, Ev::FrameDone, 0});
    t[2].push_back({1, Ev::VBlank, 0});
    t[4].push_back({1, Ev::Clear, 0}); // pre-render
    slots(t[4], false);
    t[4].push_back({280, Ev::VCopy, 0});
    t[4].push_back({321, Ev::Prefetch, 0});
    t[4].push_back({339, Ev::OddSkip, 0});
    for (auto& table : t)
      for (int i = static_cast<int>(table.size()) - 2; i >= 0; --i) {
        auto& e = table[static_cast<size_t>(i)];
        const auto& next = table[static_cast<size_t>(i + 1)];
        if (e.ev == Ev::Slot && next.ev == Ev::Slot && next.dot == e.dot + 8)
          e.run = next.run + 1;
      }
    return t;
  }();
  if (line < 240)
    return tables[0];
  if (line == 240)
    return tables[1];
  if (line == vblank_line)
    return tables[2];
  if (line == prerender_line)
    return tables[4];
  return tables[3];
}

void Ppu::catch_up(int64_t dot) {
  if (!r_.events)
    r_.events = &events_for(r_.line);
  while (r_.pos < dot) {
    const auto& events = *r_.events;
    if (r_.next_event < static_cast<int>(events.size())) {
      const EventAt& e = events[static_cast<size_t>(r_.next_event)];
      const int64_t at = r_.line_start + e.dot;
      if (at >= dot) {
        r_.pos = dot;
        return;
      }
      r_.pos = at;
      bool span = dry_ && !track_status_ && e.ev == Ev::Slot;
#ifdef TRANSFORMER_NES_PPU_REFERENCE
      span = span && engine_.ctx; // keep the independent literal-clock oracle
#endif
      if (span) {
        const int count = static_cast<int>(std::min<int64_t>(e.run, 1 + (dot - at - 1) / 8));
        const int consumed = preview_background(count, at);
        r_.next_event += consumed;
        r_.pos = at + 8 * (consumed - 1);
      } else {
        ++r_.next_event;
        process(e);
      }
      if (dry_stop_ >= 0)
        return;
      continue;
    }
    const int64_t end = r_.line_start + r_.line_len;
    if (end > dot) {
      r_.pos = dot;
      return;
    }
    r_.pos = end;
    advance_line();
  }
}

void Ppu::advance_line() {
  r_.line_start += r_.line_len;
  r_.line_len = dots_per_line;
  r_.next_event = 0;
  if (++r_.line == 262) {
    r_.line = 0;
    r_.frame_start = r_.line_start;
    r_.odd = !r_.odd;
  }
  r_.events = &events_for(r_.line);
}

int64_t Ppu::next_frame_start() const {
  if (r_.line == prerender_line && r_.pos - r_.line_start >= 339)
    return r_.line_start + r_.line_len;
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx)
    return r_.frame_start + frame_dots - (r_.odd && rendering_enabled() ? 1 : 0);
#endif
  // Ask the same event expert used at dot 339. Its skip readout depends
  // only on parity and mask; no preview or live state update is needed.
  const auto event = static_cast<size_t>(ControlEvent::OddSkip);
  const int in[control_inputs] = {static_cast<int>(event), mask, ctrl, r_.odd};
  int out[control_outputs];
  engine_.control_step(engine_.control_ctx, engine_.control_models[event], in, out);
  return r_.frame_start + frame_dots - out[out_control_skip];
}

void Ppu::process(const EventAt& e) {
  const int64_t dot = r_.pos;
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx && reference_process(e))
    return;
#endif
  switch (e.ev) {
  case Ev::Slot:
    render_slot(e.slot, dot);
    break;
  case Ev::Scanline:
    emit(rendering_enabled() ? PpuEvent::ScanlineStart : PpuEvent::Idle, dot);
    break;
  case Ev::Evaluate:
    r_.selected_count = 0;
    if (rendering_enabled())
      evaluate_sprites_graph(dot);
    break;
  case Ev::Sprites:
    load_sprites_graph(dot);
    break;
  case Ev::Prefetch:
    if (rendering_enabled()) {
      render_slot(32, dot);
      render_slot(33, dot + 8);
      a12(false, dot + 16);
    }
    break;
  case Ev::VCopy:
    control(ControlEvent::VerticalCopy, dot, {});
    break;
  case Ev::OddSkip:
    control(ControlEvent::OddSkip, dot, {});
    break;
  case Ev::FrameDone:
    if (!dry_) {
      completed_ = frame_;
      completed_experts_ = experts_;
      experts_.fill(unrendered_slot);
    }
    emit(PpuEvent::Idle, dot);
    break;
  case Ev::VBlank:
    control(ControlEvent::VBlank, dot, {});
    break;
  case Ev::Clear:
    control(ControlEvent::Clear, dot, {});
    break;
  }
}

// ── Rendering ──

// Dot 257: the next line's sprite buffer starts empty, v takes t's
// horizontal bits and OAMADDR resets; false when not rendering.
bool Ppu::start_sprite_line() {
  r_.sprites.fill(0);
  r_.sprite_slots.fill(0);
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  r_.sprite0_on_line = false;
#endif
  return control(ControlEvent::SpriteLine, r_.pos, {});
}

// Pack bounded clock context, run the event circuit, and apply its state
// and timestamp effects. Absolute clocks never enter the integer model.
bool Ppu::control(ControlEvent event, int64_t dot, const RegisterResult& effects) {
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx)
    return reference_control(event, dot, effects);
#endif
  const int age = r_.vblank_set_dot < 0 ? -1 :
      static_cast<int>(std::min<int64_t>(dot - r_.vblank_set_dot, 3));
  int in[control_inputs] = {static_cast<int>(event), mask, ctrl, r_.odd,
                static_cast<int>(dot - r_.frame_start), age,
                r_.vblank, r_.vblank_suppressed, r_.nmi_edge, r_.nmi_delayed};
  int* field = in + 10;
  for (const int value : {int(t), int(r_.v)}) {
    *field++ = value & 31;
    *field++ = (value >> 5) & 31;
    *field++ = (value >> 10) & 3;
    *field++ = (value >> 12) & 7;
  }
  *field++ = oam_addr;
  *field++ = effects.clear_vblank;
  *field++ = effects.nmi_rise;
  *field++ = effects.nmi_fall;
  *field = r_.hit_dot != NEVER || r_.overflow_dot != NEVER;
  int out[control_outputs];
  engine_.control_step(engine_.control_ctx, engine_.control_models[static_cast<size_t>(event)], in, out);
  r_.v = static_cast<uint16_t>(out[0] | (out[1] << 5) | (out[2] << 10) | (out[3] << 12));
  oam_addr = static_cast<uint8_t>(out[4]);
  r_.vblank = out[5] != 0;
  r_.vblank_suppressed = out[6] != 0;
  r_.nmi_edge = out[7] != 0;
  r_.nmi_delayed = out[8] != 0;
  if (out[9])
    r_.vblank_set_dot = dot;
  if (out[10])
    r_.vblank_set_dot = -1;
  if (out[11])
    r_.hit_dot = r_.overflow_dot = NEVER;
  if (out[12])
    mark_status_change(dot);
  if (event == ControlEvent::OddSkip)
    r_.line_len = dots_per_line - out[out_control_skip];
  return out[14] != 0;
}

// Transport ordered eight-pixel windows through the composition circuit.
// Both bytes and occupancy return as state, including right-edge padding.
void Ppu::compose_sprite_rows(const int* fetched, int count) {
  for (int k = 0; k < count; ++k) {
    const size_t x = static_cast<size_t>(fetched[out_sprite_x + k]);
    int in[compose_inputs] = {static_cast<int>(x), r_.sprite_slots[x >> 3], r_.sprite_slots[(x >> 3) + 1]};
    std::copy_n(fetched + k * 8, 8, in + 3);
    std::copy_n(r_.sprites.data() + x, 8, in + 11);
    int out[compose_outputs];
    engine_.control_step(engine_.control_ctx, engine_.compose_model, in, out);
    std::copy_n(out, 8, r_.sprites.data() + x);
    r_.sprite_slots[x >> 3] = out[8];
    r_.sprite_slots[(x >> 3) + 1] = out[9];
  }
}

// The selected OAM indices and diagonal overflow candidate come from
// transformer tokens. Unlimited sprites repeat selection after each
// eight-index batch; the ordinary path is one folded evaluation.
void Ppu::evaluate_sprites_graph(int64_t dot) {
  if (dry_ && ((!track_status_ && !wants_a12_) ||
               (track_status_ && r_.overflow_dot != NEVER &&
                (r_.hit_dot != NEVER || (mask & 0x18) != 0x18))))
    return; // only status changes and mapper bus clocks matter to a preview
  int in[evaluate_inputs] = {1, r_.line, ctrl, 0};
  int out[evaluate_outputs];
  auto& rows = sprite_rows_;
  const int model = engine_.sprites.evaluate[(ctrl & 0x20) ? 2 : 1];
  if (rows.oam_epoch != oam_epoch_ || rows.model != model) {
    rows.valid.reset();
    rows.oam_epoch = oam_epoch_;
    rows.model = model;
  }
  const auto line = static_cast<size_t>(r_.line);
  auto& row = rows.rows[line];
  if (rows.valid[line]) {
    std::copy(row.begin(), row.end(), out);
  } else {
    graph_step(model, in, out);
    std::copy_n(out, row.size(), row.begin());
    rows.valid[line] = true;
  }
  r_.selected_count = out[0];
  for (int k = 0; k < out[0]; ++k)
    r_.selected[static_cast<size_t>(k)] = static_cast<uint8_t>(out[1 + k]);
  if ((!dry_ || track_status_) && out[0] == 8 && r_.overflow_dot == NEVER) {
    for (int start = out[8] + 1; start < 64; start += 8) {
      const int probe[4] = {1, r_.line, ctrl, start};
      int found;
      graph_step(engine_.sprites.overflow[(ctrl & 0x20) ? 2 : 1], probe, &found);
      if (found < 64) {
        r_.overflow_dot = dot + 48 + 2 * found;
        mark_status_change(r_.overflow_dot);
        break;
      }
    }
  }
  while (!dry_ && !sprite_limit && out[0] == 8 && out[8] < 63) {
    in[3] = out[8] + 1;
    graph_step(engine_.sprites.evaluate[(ctrl & 0x20) ? 4 : 3], in, out);
    for (int k = 0; k < out[0]; ++k)
      r_.selected[static_cast<size_t>(r_.selected_count++)] = static_cast<uint8_t>(out[1 + k]);
  }
}

// The sprite fetch through the fetch graph: the selected sprites' rows
// and where they go, the latches the fetches flip, and the pattern table
// every fetch reads, for its A12 edge.
void Ppu::load_sprites_graph(int64_t dot) {
  if (dry_ && ((!track_status_ && !wants_a12_) ||
               (track_status_ && (r_.hit_dot != NEVER || (mask & 0x18) != 0x18))))
    return;
  if (!start_sprite_line())
    return;
  const int total = r_.line < 240 ? r_.selected_count : 0;
  const int count = std::min(total, 8);
  int in[fetch_inputs] = {1, r_.line, ctrl, count};
  for (int k = 0; k < count; ++k)
    in[in_index + k] = r_.selected[static_cast<size_t>(k)];
  in[in_latch] = chr_latch_active && chr_latch.latch_fe[0];
  in[in_latch + 1] = chr_latch_active && chr_latch.latch_fe[1];
  in[in_latch + 2] = chr_latch.wide_low;
  if (mmc5.active())
    for (int s = 0; s < 8; ++s)
      in[in_banks + s] = static_cast<int>(sprite_banks()[s] >> 10);
  const bool bus_only = dry_ && !track_status_;
  const bool status_only = dry_ && track_status_;
  const auto& models = bus_only      ? engine_.sprites.fetch_bus
                       : status_only ? engine_.sprites.fetch_status
                                     : engine_.sprites.fetch;
  int out[fetch_outputs];
  const int model = chr_latch_active ? models[0] : models[1 + ((ctrl & 0x20) ? 9 : 0) + count];
  if (status_only) {
    int selected[27];
    graph_step(model, in, selected);
    std::copy_n(selected, 8, out);
    out[out_sprite_x] = selected[8];
    std::copy_n(selected + 9, 18, out + out_sprite_fetch);
  } else {
    graph_step(model, in, bus_only ? out + out_sprite_fetch : out);
  }
  for (int k = 0; k < 8; ++k)
    if (out[out_sprite_fetch + k]) {
      bus_event(BusEvent::Sprite, out[out_sprite_a12 + k] != 0, dot + 8 * k);
    }
  const bool latch_lo = out[out_sprite_latch] != 0, latch_hi = out[out_sprite_latch + 1] != 0;
  if (chr_latch_active &&
      (chr_latch.latch_fe[0] != latch_lo || chr_latch.latch_fe[1] != latch_hi)) {
    chr_latch.latch_fe[0] = latch_lo;
    chr_latch.latch_fe[1] = latch_hi;
    chr_map_changed();
  }
  if (bus_only)
    return;
  compose_sprite_rows(out, status_only ? std::min(count, 1) : count);
  if (status_only)
    return;
  // Debug rows past the hardware cap affect pixels only: no A12 edges
  // and no latch flips, even within a batch.
  for (int base = 8; base < total; base += 8) {
    const int extra = std::min(total - base, 8);
    in[3] = extra;
    for (int k = 0; k < extra; ++k)
      in[in_index + k] = r_.selected[static_cast<size_t>(base + k)];
    in[in_latch] = chr_latch_active && chr_latch.latch_fe[0];
    in[in_latch + 1] = chr_latch_active && chr_latch.latch_fe[1];
    graph_step(chr_latch_active ? engine_.sprites.debug_fetch[(ctrl & 0x20) ? 1 : 0]
                               : models[1 + ((ctrl & 0x20) ? 9 : 0) + extra], in, out);
    compose_sprite_rows(out, extra);
  }
}

// ── Events ──

// Deliver the neural bus packet. The host supplies only bounded clock
// context and applies the returned timestamp and edge effects.
bool Ppu::bus_event(BusEvent event, bool high, int64_t dot) {
  if (!wants_a12_)
    return false;
  const int age = static_cast<int>(std::clamp<int64_t>(dot - r_.a12_low_since, -1, 11));
  const int in[]{static_cast<int>(event), r_.a12_high, age, high, ctrl,
                 mmc5.active() && mmc5.extended_attributes()};
  int out[4];
#ifdef TRANSFORMER_NES_PPU_REFERENCE
  if (!engine_.ctx)
    reference_bus(in, out);
  else
#endif
    engine_.control_step(engine_.control_ctx, engine_.bus_models[static_cast<size_t>(event)], in, out);
  r_.a12_high = out[0] != 0;
  if (out[1])
    r_.a12_low_since = dot;
  if (out[2])
    emit(PpuEvent::A12Rise, dot + out[3]);
  return out[2] != 0;
}

void Ppu::refresh_mmc5_view() {
  if (!mmc5.active())
    return;
  const int in[]{mmc5.exram_mode(), mmc5.fill_tile(), mmc5.fill_color(), ctrl};
  int out[6];
  if (!engine_.ctx) {
#ifdef TRANSFORMER_NES_PPU_REFERENCE
    reference_view(in, out);
#else
    // A mapper may configure storage before the PPU engine attaches.
    // set_engine refreshes the descriptor before any graph can read it.
    return;
#endif
  } else {
    engine_.control_step(engine_.control_ctx, engine_.view_model, in, out);
  }
  mmc5.apply_view(out);
}

void Ppu::emit(PpuEvent::Kind kind, int64_t dot) {
  PpuWatcher* watcher = dry_ ? probe_ : watcher_;
  if (!watcher)
    return;
  watcher->ppu_event({kind, dot});
  if (dry_ && probe_->fire_dot >= 0 && dry_stop_ < 0)
    dry_stop_ = probe_->fire_dot;
}

void Ppu::mark_status_change(int64_t dot) {
  if (dry_ && track_status_ && dry_stop_ < 0)
    dry_stop_ = dot;
}

int64_t Ppu::preview(int64_t until, IrqProbe* probe, bool track_status) {
  const Render saved = r_;
  const ChrLatch latch = chr_latch;
  const uint8_t saved_oam_addr = oam_addr;
  dry_ = true;
  probe_ = probe;
  track_status_ = track_status;
  dry_stop_ = -1;
  // Forward tokens only read graph memory. Preview latch changes select
  // between already-mapped FD/FE windows; they do not change those bytes.
  // The same engine can therefore serve the preview without a memory copy.
  catch_up(until);
  const int64_t found = dry_stop_;
  r_ = saved;
  chr_latch = latch;
  oam_addr = saved_oam_addr;
  dry_ = track_status_ = false;
  probe_ = nullptr;
  dry_stop_ = -1;
  return found;
}

} // namespace transformer_nes::nes
