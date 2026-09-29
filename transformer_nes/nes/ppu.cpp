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

namespace {

// A pattern plane byte (MSB = leftmost pixel) as eight byte lanes, one
// per pixel, leftmost in the lowest address.
constexpr auto plane = [] {
  array<uint64_t, 256> t{};
  for (int b = 0; b < 256; ++b)
    for (int c = 0; c < 8; ++c)
      if (b & (0x80 >> c))
        t[static_cast<size_t>(b)] |= uint64_t{1} << (8 * c);
  return t;
}();

} // namespace

Ppu::Ppu(vector<uint8_t> chr_data, bool writable, bool vertical_mirroring)
    : chr(std::move(chr_data)), chr_writable(writable), mirroring(vertical_mirroring ? 2 : 3) {
  if (chr.size() < 0x2000)
    chr.resize(0x2000, 0);
  for (size_t i = 0; i < chr_bank.size(); ++i)
    chr_bank[i] = i * 0x400;
}

// ── Memory ──

size_t Ppu::palette_index(int addr) {
  size_t offset = (addr - 0x3F00) & 0x1F;
  if (offset == 0x10 || offset == 0x14 || offset == 0x18 || offset == 0x1C)
    offset -= 0x10;
  return offset;
}

size_t Ppu::nt_offset_for(int addr, int mirr) {
  const size_t a = (addr - 0x2000) & 0xFFF;
  const size_t table = a >> 10, offset = a & 0x3FF;
  size_t phys;
  switch (mirr) {
  case 0:
    phys = 0;
    break; // single-screen low
  case 1:
    phys = 1;
    break; // single-screen high
  case 2:
    phys = table & 1;
    break; // vertical
  default:
    phys = table >> 1;
    break; // horizontal
  }
  return phys * 0x400 + offset;
}

void Ppu::map_nametables() {
  nt_mirroring_ = mirroring;
  for (int table = 0; table < 4; ++table)
    nt_base_[static_cast<size_t>(table)] =
        static_cast<uint16_t>(nt_offset_for(0x2000 + table * 0x400, mirroring));
}

std::pair<uint8_t, uint8_t> Ppu::fetch_pattern(const size_t* banks, int pattern_addr) {
  if (!chr_latch_active) {
    const uint8_t* bank = chr.data() + banks[(pattern_addr >> 10) & 7] + (pattern_addr & 0x3FF);
    return {bank[0], bank[8]};
  }
  // The FD/FE tile itself still fetches through the old bank: both
  // pattern bytes are read before either fetch flips its latch.
  const uint8_t lo = chr[chr_index(banks, pattern_addr)];
  const uint8_t hi = chr[chr_index(banks, pattern_addr + 8)];
  const bool flipped_lo = chr_latch.after_fetch(pattern_addr);
  if (chr_latch.after_fetch(pattern_addr + 8) || flipped_lo)
    chr_map_changed();
  return {lo, hi};
}

int Ppu::vram_peek(int addr) {
  addr &= 0x3FFF;
  if (addr < 0x2000)
    return chr[chr_index(chr_bank.data(), static_cast<size_t>(addr))];
  if (addr < 0x3F00)
    return nt_read(addr);
  return palette[palette_index(addr)];
}

void Ppu::vram_write(int addr, int value) {
  addr &= 0x3FFF;
  if (addr < 0x2000) {
    if (chr_writable) {
      const size_t source = chr_bank[(addr >> 10) & 7] + (addr & 0x3FF);
      chr[source] = static_cast<uint8_t>(value);
      mark_chr(source);
    }
  } else if (addr < 0x3F00) {
    const int a = (addr - 0x2000) & 0xFFF;
    if (mmc5.active()) {
      mmc5.write_nametable(a >> 10, a & 0x3FF, value, nametables.data());
      mark_page(static_cast<size_t>(((nt_map() >> (2 * (a >> 10))) & 3) * 0x400 + (a & 0x3FF)));
    } else {
      const size_t offset = nt_offset_for(addr, mirroring);
      nametables[offset] = static_cast<uint8_t>(value);
      mark_page(offset);
    }
  } else {
    palette_changed();
    palette[palette_index(addr)] = static_cast<uint8_t>(value & 0x3F); // 6-bit RAM
  }
}

// With rendering off the PPU outputs the backdrop -- or, while v points
// into palette RAM, that palette entry (the NESDEV backdrop override).
uint8_t Ppu::backdrop() const {
  const int a = r_.v & 0x3FFF;
  const uint8_t gmask = (mask & 0x01) ? 0x30 : 0x3F;
  return (a >= 0x3F00 ? palette[palette_index(a)] : palette[0]) & gmask;
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
  // PPUMASK reaches the renderer one dot late (the rendering-enable
  // latch), which decides e.g. whether this frame skips its odd dot.
  catch_up(reg == 1 ? dot + 1 : dot);
  (void)access(reg, value & 0xFF, true, dot);
}

// One register access in two parts: the register-file transition --
// natively, or through an attached external step (the register graph,
// ppu2c02/registers.py, which computes exactly this) -- and then the
// effects that depend on the dot rather than on the access.
int Ppu::access(int reg, int value, bool is_write, int64_t dot) {
  const RegisterFile before = register_file();
  const RegisterAccess request{reg, value, is_write, status(dot),
                               rendering_enabled() && rendering_line()};
  RegisterResult result;
  if (register_step_.access) {
    deliver_memory_changes(!is_write && reg == 7 && (r_.v & 0x3FFF) < 0x2000);
    result = external_transition(request, before);
  } else {
    result = transition(request);
  }
  access_effects(request, before, dot);
  return result.read_value;
}

// The register file's own semantics: loopy t/v/x/w, ctrl, mask, the OAM
// address, and the $2007 read buffer.  A memory write is applied here and
// reported at its graph-memory address (graph_address).
Ppu::RegisterResult Ppu::transition(const RegisterAccess& request) {
  RegisterResult out;
  const int value = request.value;
  switch (request.reg + (request.is_write ? 8 : 0)) {
  case 8: // PPUCTRL
    ctrl = value;
    t = static_cast<uint16_t>((t & ~0x0C00) | ((value & 3) << 10));
    break;
  case 9: // PPUMASK
    mask = value;
    break;
  case 2: // PPUSTATUS
    out.read_value = request.status;
    write_latch = false;
    break;
  case 11: // OAMADDR
    oam_addr = static_cast<uint8_t>(value);
    break;
  case 12: // OAMDATA
    out.mem_addr = 0x4000 + oam_addr;
    out.mem_value = value;
    oam[oam_addr++] = static_cast<uint8_t>(value);
    oam_changed();
    break;
  case 4: // OAMDATA
    out.read_value = oam[oam_addr];
    break;
  case 13: // PPUSCROLL (x then y, into t)
    if (write_latch) {
      t = static_cast<uint16_t>((t & ~0x73E0) | ((value & 7) << 12) | ((value & 0xF8) << 2));
    } else {
      fine_x = value & 7;
      t = static_cast<uint16_t>((t & ~0x001F) | (value >> 3));
    }
    write_latch = !write_latch;
    break;
  case 14: // PPUADDR (high then low); the second write loads v
    if (write_latch) {
      t = static_cast<uint16_t>((t & 0x7F00) | value);
      r_.v = t;
    } else {
      t = static_cast<uint16_t>((t & 0x00FF) | ((value & 0x3F) << 8));
    }
    write_latch = !write_latch;
    break;
  case 15: { // PPUDATA
    const int a = r_.v & 0x3FFF;
    // CHR ROM and MMC5's fill-mode page take no write.
    const int32_t phys = graph_address(a);
    if ((a >= 0x2000 || chr_writable) && !(phys >= 0x2C00 && phys < 0x3000)) {
      out.mem_addr = phys;
      out.mem_value = a >= 0x3F00 ? value & 0x3F : value;
    }
    vram_write(a, value);
    step_vram_address(request.rendering);
    break;
  }
  case 7: { // PPUDATA (buffered below $3F00, direct for palette)
    const int a = r_.v & 0x3FFF;
    if (a >= 0x3F00) {
      out.read_value = palette[palette_index(a)];
      // The buffer still loads from the nametable "underneath".
      read_buffer = vram_peek(a - 0x1000);
    } else {
      out.read_value = read_buffer;
      read_buffer = vram_peek(a);
    }
    step_vram_address(request.rendering);
    break;
  }
  default: // write-only registers read as open bus; model as 0
    break;
  }
  return out;
}

// $2007 steps v by 1/32 -- except during rendering, where the access
// collides with the fetch pipeline and bumps coarse x and y instead.
void Ppu::step_vram_address(bool rendering) {
  if (rendering) {
    increment_x();
    increment_y();
  } else {
    r_.v = static_cast<uint16_t>((r_.v + increment()) & 0x7FFF);
  }
}

// Everything an access does that depends on the dot or on the rest of
// the console: the vblank/NMI races, mapper-visible A12 edges and CHR
// latches, and the renderer's caches.
void Ppu::access_effects(const RegisterAccess& request, const RegisterFile& before,
                         int64_t dot) {
  const int reg = request.reg;
  if (reg == 2 && !request.is_write) {
    // The vblank race (NESDEV PPU frame timing): a read one dot before
    // the flag sets reads it clear and cancels this frame's flag and
    // NMI; a read on the setting dot or the next reads it set but still
    // cancels the NMI; later reads behave normally.
    if (dot == vblank_dot() && r_.vblank_set_dot < 0)
      r_.vblank_suppressed = true;
    if (r_.vblank_set_dot >= 0 && dot - r_.vblank_set_dot <= 2)
      r_.nmi_edge = false;
    r_.vblank = false;
  }
  if (reg == 0 && request.is_write) {
    const bool rising = (ctrl & 0x80) && !(before.ctrl & 0x80);
    const bool falling = !(ctrl & 0x80) && (before.ctrl & 0x80);
    // The NMI output lags by a dot: enabling on the flag's last dot
    // raises no pulse.
    if (rising && r_.vblank && dot != r_.frame_start + int64_t{prerender_line} * dots_per_line + 1)
      r_.nmi_delayed = true;
    // Disabling within two dots of the flag setting drops the NMI line
    // before the CPU's edge detector latches it.
    if (falling && r_.vblank_set_dot >= 0 && dot - r_.vblank_set_dot <= 2)
      r_.nmi_edge = false;
  }
  // Outside rendering the PPU address bus shows v, so loading or
  // stepping it can raise A12 (an MMC3 clock).
  const bool v_loaded = reg == 6 && request.is_write && before.w && !rendering_enabled();
  const bool v_stepped = reg == 7 && !request.rendering;
  if ((v_loaded || v_stepped) && a12((r_.v & 0x1000) != 0, dot))
    schedule_changed_ = true;
  // A $2007 read below $2000 is a pattern fetch the MMC2/MMC4 latches see.
  if (reg == 7 && !request.is_write && chr_latch_active && (before.v & 0x3FFF) < 0x2000 &&
      chr_latch.after_fetch(static_cast<size_t>(before.v & 0x3FFF)))
    chr_map_changed();
  if (before.ctrl != ctrl)
    chr_map_changed(); // MMC5's background set follows the sprite size
  if (((before.ctrl ^ ctrl) & 0x38) || ((before.mask ^ mask) & 0x18) ||
      (reg == 4 && request.is_write && (ctrl & 0x20)))
    schedule_changed_ = true;
  if ((reg == 4 && request.is_write) || ((before.ctrl ^ ctrl) & 0x20))
    sprites_indexed_ = false;
}

// The transition through the attached step (its memory is already up
// to date): load the register file it returns and apply its write.
Ppu::RegisterResult Ppu::external_transition(const RegisterAccess& request,
                                             const RegisterFile& before) {
  const auto [after, result] = register_step_.access(request, nt_map(), chr_writable, before);
  t = static_cast<uint16_t>(after.t);
  r_.v = static_cast<uint16_t>(after.v);
  fine_x = after.fine_x;
  write_latch = after.w != 0;
  ctrl = after.ctrl;
  mask = after.mask;
  oam_addr = static_cast<uint8_t>(after.oam_addr);
  read_buffer = after.read_buffer;
  if (result.mem_addr >= 0)
    write_physical(result.mem_addr, result.mem_value);
  return result;
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

void Ppu::set_register_step(RegisterStep step) {
  register_step_ = std::move(step);
  register_sync_.reset(); // the new step's memory starts empty
}

void Ppu::set_render_engine(RenderEngine engine) {
  render_ = engine;
  render_sync_.reset();
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

// Deliver the graph-memory view's changes since `sync` last saw it.
// Nametable pages go by written 64-byte chunk, palette and OAM by change
// count; CHR goes by 1 KiB slot, resent when the slot's source moves and
// otherwise by written chunk.  The register view maps CHR as $2007 reads
// see it (current banks and latches); the render view maps the
// background banks, both latch windows, and for MMC5 the whole CHR ROM.
template <class Write>
void Ppu::sync_graph_memory(GraphMemorySync& sync, ChrView view, bool chr_window, bool chr_full,
                            Write&& write) {
  if (chr_window && sync.seen == memory_epoch_)
    return;
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
  if (!chr_window)
    return; // CHR changes wait for the next delivery that wants the window
  auto dirty = [&](size_t source) {
    return (sync.chr_dirty[source >> 12] >> ((source >> 6) & 63)) & 1;
  };
  if (sync.chr_any || sync.chr_map != chr_map_epoch_) {
    sync.chr_map = chr_map_epoch_;
    const bool latches = view == ChrView::render && chr_latch_active;
    for (int s = 0; s < (latches ? 16 : 8); ++s) {
      const int slot = s & 7;
      size_t source;
      if (view == ChrView::registers)
        source = chr_index(chr_bank.data(), static_cast<size_t>(slot) * 0x400);
      else if (latches)
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
void Ppu::render_slot(int slot, int64_t dot) {
  sync_graph_memory(render_sync_, ChrView::render, true, mmc5.active(),
                    [this](int addr, int value) { render_.write(render_.ctx, addr, value); });
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
  const bool sprites = visible && ((r_.sprite_slots >> slot) & 1);
  if (visible) {
    std::copy_n(r_.bg.data() + slot * 8, 16, in + in_bg);
    if (sprites)
      std::copy_n(r_.sprites.data() + slot * 8, 8, in + in_sprites);
  }
  // A folded expert when its pins hold: background on (so rendering), no
  // greyscale, no extended attributes; by slot, table and fine x.  Edge
  // slots with sprites have none.
  int model = 0;
  if ((mask & 0x08) && !(mask & 0x01) && !extended && !force_full_render) {
    const RenderDispatch& d = render_.dispatch[chr_latch_active ? 1 : 0];
    const int table = (ctrl & 0x10) ? 1 : 0;
    if (slot == 0 || slot == 31)
      model = sprites ? 0 : slot == 0 ? d.first[table] : d.last[table];
    else
      model = sprites ? d.sprites : d.inner[table][fine_x];
  }
  int out[render_outputs];
  render_.step(render_.ctx, model, in, out);
  if (render_.trace && !dry_ && slot_trace_request_ == std::pair{r_.line, slot}) {
    SlotTrace trace{r_.line, slot, model,
                    vector<int>(static_cast<size_t>(render_.trace_rows * render_.trace_width))};
    render_.trace(render_.ctx, in, trace.residual.data());
    slot_trace_ = std::move(trace);
    slot_trace_request_ = {-1, -1};
  }
  if (visible && r_.line < 240) {
    experts_[static_cast<size_t>(r_.line * 32 + slot)] = static_cast<uint8_t>(model);
    uint8_t* row = frame_.data() + static_cast<size_t>(r_.line) * 256 + slot * 8;
    for (int i = 0; i < 8; ++i)
      row[i] = static_cast<uint8_t>(out[i]);
    for (int i = 0; i < 8 && r_.hit_dot == NEVER; ++i)
      if (out[out_hit + i]) {
        r_.hit_dot = dot + i;
        mark_status_change(dot + i);
      }
  }
  if (out[out_fetched]) { // the tile, v, the latches, and the fetch's A12 edge
    a12(false, dot);
    if (!extended)
      a12((ctrl & 0x10) != 0, dot + 4);
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

void Ppu::oam_dma(span<const uint8_t, 256> page, int64_t dot) {
  catch_up(dot);
  for (size_t i = 0; i < page.size(); ++i)
    oam[(oam_addr + i) & 0xFF] = page[i];
  sprites_indexed_ = false;
  oam_changed();
  if (ctrl & 0x20)
    schedule_changed_ = true;
}

// ── The register file as an external step ──

Ppu::RegisterFile Ppu::register_file() const {
  return {t, r_.v, fine_x, write_latch, ctrl, mask, oam_addr, read_buffer};
}

// A PPU address's place in graph memory (ppu2c02/space.py).
int32_t Ppu::graph_address(int addr) const {
  addr &= 0x3FFF;
  if (addr < 0x2000)
    return addr;
  if (addr < 0x3F00) {
    const int table = (addr >> 10) & 3;
    return 0x2000 + 0x400 * ((nt_map() >> (2 * table)) & 3) + (addr & 0x3FF);
  }
  return static_cast<int32_t>(0x3F00 + palette_index(addr));
}

// Deliver the register view's changes since the last access to the
// step.  The CHR window is only read through a $2007 read below $2000,
// so bank switches (several a frame under MMC3) are delivered just
// before one.
void Ppu::deliver_memory_changes(bool chr_window) {
  sync_graph_memory(register_sync_, ChrView::registers, chr_window, false,
                    [this](int addr, int value) { register_step_.write(addr, value); });
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
      ++r_.next_event;
      process(e);
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
  return r_.frame_start + frame_dots - (r_.odd && rendering_enabled() ? 1 : 0);
}

void Ppu::process(const EventAt& e) {
  const int64_t dot = r_.pos;
  switch (e.ev) {
  case Ev::Slot:
    if (render_.ctx && !dry_) {
      render_slot(e.slot, dot);
      break;
    }
    if (r_.line < 240)
      output_pixels(e.slot, dot);
    if (rendering_enabled()) {
      fetch_tile(e.slot + 2, dot);
      increment_x();
      if (e.slot == 31)
        increment_y();
    }
    break;
  case Ev::Scanline:
    emit(rendering_enabled() ? PpuEvent::ScanlineStart : PpuEvent::Idle, dot);
    break;
  case Ev::Evaluate:
    r_.selected_count = 0;
    if (rendering_enabled())
      evaluate_sprites(dot);
    break;
  case Ev::Sprites:
    load_sprites(dot);
    break;
  case Ev::Prefetch:
    if (render_.ctx && !dry_) {
      if (rendering_enabled()) {
        render_slot(32, dot);
        render_slot(33, dot + 8);
        a12(false, dot + 16);
      }
      break;
    }
    if (rendering_enabled()) {
      fetch_tile(0, dot);
      increment_x();
      fetch_tile(1, dot + 8);
      increment_x();
      a12(false, dot + 16); // dots 337-340: nametable fetches
    }
    break;
  case Ev::VCopy:
    if (rendering_enabled())
      r_.v = static_cast<uint16_t>((r_.v & ~0x7BE0) | (t & 0x7BE0));
    break;
  case Ev::OddSkip:
    if (r_.odd && rendering_enabled())
      r_.line_len = dots_per_line - 1;
    break;
  case Ev::FrameDone:
    if (!dry_) {
      completed_ = frame_;
      completed_experts_ = experts_;
      experts_.fill(native_slot);
    }
    emit(PpuEvent::Idle, dot);
    break;
  case Ev::VBlank:
    if (std::exchange(r_.vblank_suppressed, false))
      break;
    r_.vblank = true;
    r_.vblank_set_dot = dot;
    if (ctrl & 0x80)
      r_.nmi_edge = true;
    mark_status_change(dot);
    break;
  case Ev::Clear:
    sprites_indexed_ = false;
    if (r_.vblank || r_.hit_dot != NEVER || r_.overflow_dot != NEVER)
      mark_status_change(dot);
    r_.vblank = false;
    r_.vblank_set_dot = -1;
    r_.hit_dot = r_.overflow_dot = NEVER;
    break;
  }
}

// ── Rendering ──

void Ppu::increment_x() {
  if ((r_.v & 0x001F) == 31)
    r_.v = static_cast<uint16_t>((r_.v & ~0x001F) ^ 0x0400);
  else
    ++r_.v;
}

void Ppu::increment_y() {
  if ((r_.v & 0x7000) != 0x7000) {
    r_.v = static_cast<uint16_t>(r_.v + 0x1000);
    return;
  }
  int v = r_.v & ~0x7000;
  int y = (v & 0x03E0) >> 5;
  if (y == 29) {
    y = 0;
    v ^= 0x0800;
  } else if (y == 31) {
    y = 0;
  } else {
    ++y;
  }
  r_.v = static_cast<uint16_t>((v & ~0x03E0) | (y << 5));
}

// A background tile slot: nametable, attribute, then the two pattern
// bytes at v.  Pattern fetches drive A12 (MMC3's scanline clock) and
// the MMC2/MMC4 latches in true fetch order.
void Ppu::fetch_tile(int index, int64_t dot) {
  const int v = r_.v;
  a12(false, dot);
  const int tile_no = nt_read(0x2000 | (v & 0x0FFF));
  const int fine_y = (v >> 12) & 7;
  int palette_sel;
  uint8_t lo, hi;
  if (mmc5.active() && mmc5.extended_attributes()) {
    const int offset = v & 0x3FF;
    const size_t base = mmc5.extended_chr_base(offset) + static_cast<size_t>(tile_no) * 16 + fine_y;
    lo = chr[base % chr.size()];
    hi = chr[(base + 8) % chr.size()];
    palette_sel = mmc5.exram(offset) >> 6;
  } else {
    const int attr = nt_read(0x23C0 | (v & 0x0C00) | ((v >> 4) & 0x38) | ((v >> 2) & 0x07));
    palette_sel = (attr >> (((v >> 4) & 4) | (v & 2))) & 3;
    const int pattern = ((ctrl & 0x10) ? 0x1000 : 0) + tile_no * 16 + fine_y;
    a12((pattern & 0x1000) != 0, dot + 4);
    std::tie(lo, hi) = fetch_pattern(bg_banks(), pattern);
  }
  // Eight 2-bit pixels, then the palette offset on the opaque ones.
  uint64_t pixels = plane[lo] | (plane[hi] << 1);
  const uint64_t opaque = (pixels | (pixels >> 1)) & 0x0101010101010101ULL;
  pixels += opaque * static_cast<uint64_t>(palette_sel * 4);
  std::memcpy(r_.bg.data() + static_cast<size_t>(index) * 8, &pixels, 8);
}

// Pixels 8*slot .. 8*slot+7 of the current line, output at dots
// 8*slot+1 .. 8*slot+8 from the fetched tiles and the sprite line.
void Ppu::output_pixels(int slot, int64_t dot) {
  // A dry run only needs pixels to find sprite-0 hits.
  if (dry_ && !(track_status_ && r_.sprite0_on_line))
    return;
  uint8_t* row = frame_.data() + static_cast<size_t>(r_.line) * 256 + slot * 8;
  const bool bg_on = mask & 0x08, sprites_on = mask & 0x10;
  if (!bg_on && !sprites_on) {
    if (!dry_)
      std::memset(row, backdrop(), 8);
    return;
  }
  const uint8_t gmask = (mask & 0x01) ? 0x30 : 0x3F;
  // Background palette indices under this slot; the left-edge mask
  // (and a disabled background) show index 0, the backdrop.
  array<uint8_t, 8> bg{};
  if (bg_on) {
    std::memcpy(bg.data(), r_.bg.data() + slot * 8 + fine_x, 8);
    if (slot == 0 && !(mask & 0x02))
      bg.fill(0);
  }
  const bool sprites_here = sprites_on && ((r_.sprite_slots >> slot) & 1);
  if (!sprites_here) {
    if (!dry_)
      for (int i = 0; i < 8; ++i)
        row[i] = palette[bg[static_cast<size_t>(i)]] & gmask;
    return;
  }
  const uint8_t* sprite = r_.sprites.data() + slot * 8;
  const bool sprite_clip = slot == 0 && !(mask & 0x04);
  for (int i = 0; i < 8; ++i) {
    const uint8_t b = bg[static_cast<size_t>(i)];
    uint8_t index = b;
    const uint8_t s = sprite_clip ? 0 : sprite[i];
    if (s) {
      // Sprite 0 hits on opaque-over-opaque, whatever its priority,
      // never at x=255.
      if ((s & 0x80) && b && slot * 8 + i != 255 && r_.hit_dot == NEVER) {
        r_.hit_dot = dot + i;
        mark_status_change(dot + i);
      }
      if (!((s & 0x40) && b))
        index = s & 0x1F;
    }
    if (!dry_)
      row[i] = palette[index] & gmask;
  }
}

// Dot 257: copy t's horizontal bits into v, then evaluate OAM for the
// next line and run the eight sprite-slot fetches (empty slots fetch
// tile $FF, which still drives A12).  Sprites claim pixels lowest index
// first; only the claiming sprite's priority bit decides front/behind.
void Ppu::load_sprites(int64_t dot) {
  if (r_.any_sprites)
    r_.sprites.fill(0);
  r_.any_sprites = r_.sprite0_on_line = false;
  r_.sprite_slots = 0;
  if (!rendering_enabled())
    return;
  r_.v = static_cast<uint16_t>((r_.v & ~0x041F) | (t & 0x041F));
  oam_addr = 0;
  const int h = (ctrl & 0x20) ? 16 : 8;
  const bool draw = r_.line < 239; // the next line is visible
  const int count = r_.line < 240 ? r_.selected_count : 0;
  for (int slot = 0; slot < count; ++slot) {
    const int i = r_.selected[static_cast<size_t>(slot)];
    int row = r_.line - oam[static_cast<size_t>(i) * 4];
    const int tile = oam[static_cast<size_t>(i) * 4 + 1];
    const int attr = oam[static_cast<size_t>(i) * 4 + 2];
    const int x = oam[static_cast<size_t>(i) * 4 + 3];
    if (row < 0 || row >= h) // OAM or sprite size changed since evaluation
      continue;
    if (attr & 0x80)
      row = h - 1 - row;
    const int addr = h == 16
                         ? ((tile & 1) << 12) | ((tile & 0xFE) << 4) | ((row & 8) << 1) | (row & 7)
                         : ((ctrl & 0x08) << 9) | (tile << 4) | row;
    uint8_t lo, hi;
    if (slot < 8) {
      a12(false, dot + 8 * slot);
      a12((addr & 0x1000) != 0, dot + 8 * slot + 4);
      std::tie(lo, hi) = fetch_pattern(sprite_banks(), addr);
    } else { // past the hardware cap (debug toggle): no bus side effects
      lo = chr[chr_index(sprite_banks(), addr)];
      hi = chr[chr_index(sprite_banks(), addr + 8)];
    }
    if (!draw || !(lo | hi))
      continue;
    const uint8_t meta = static_cast<uint8_t>(0x10 | ((attr & 3) << 2) |
                                              ((attr & 0x20) ? 0x40 : 0) | (i == 0 ? 0x80 : 0));
    for (int c = 0; c < 8 && x + c < 256; ++c) {
      const int bit = (attr & 0x40) ? c : 7 - c;
      const int value = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
      auto& pixel = r_.sprites[static_cast<size_t>(x + c)];
      if (!value || pixel)
        continue;
      pixel = static_cast<uint8_t>(meta | value);
      r_.any_sprites = true;
      r_.sprite_slots |= 1U << ((x + c) >> 3);
      r_.sprite0_on_line |= i == 0;
    }
  }
  const int dummy = (h == 16 || (ctrl & 0x08)) ? 0x1000 : 0;
  for (int slot = std::min(count, 8); slot < 8; ++slot) {
    a12(false, dot + 8 * slot);
    a12(dummy != 0, dot + 8 * slot + 4);
  }
}

// Sprite evaluation (dots 65-256): select the next line's sprites in
// OAM order and run the overflow search with its timing and hardware
// bug -- each sprite takes 2 dots to reject or 8 to copy, and once 8
// are found the search checks byte m of each sprite as its Y, bumping
// m along with n on a miss (the diagonal scan).
void Ppu::index_sprites() {
  sprites_indexed_ = true;
  line_sprite_count_.fill(0);
  const int h = (ctrl & 0x20) ? 16 : 8;
  for (int i = 0; i < 64; ++i) {
    const int top = oam[static_cast<size_t>(i) * 4];
    for (int line = top; line < top + h && line < 240; ++line) {
      auto& count = line_sprite_count_[static_cast<size_t>(line)];
      line_sprites_[static_cast<size_t>(line)][count++] = static_cast<uint8_t>(i);
    }
  }
}

void Ppu::evaluate_sprites(int64_t dot) {
  if (!sprites_indexed_)
    index_sprites();
  const size_t line = static_cast<size_t>(r_.line);
  if (line_sprite_count_[line] < 8) { // no overflow search: take the index
    r_.selected_count = line_sprite_count_[line];
    std::copy_n(line_sprites_[line].begin(), r_.selected_count, r_.selected.begin());
    return;
  }
  const int h = (ctrl & 0x20) ? 16 : 8;
  auto in_range = [&](int y) {
    return static_cast<unsigned>(r_.line - y) < static_cast<unsigned>(h);
  };
  int64_t at = dot;
  int n = 0, count = 0;
  for (; n < 64 && count < 8; ++n) {
    const bool hit = in_range(oam[static_cast<size_t>(n) * 4]);
    if (hit)
      r_.selected[static_cast<size_t>(count++)] = static_cast<uint8_t>(n);
    at += hit ? 8 : 2;
  }
  r_.selected_count = count;
  if (count < 8)
    return;
  if (!sprite_limit) // debug: render every in-range sprite
    for (int i = n; i < 64; ++i)
      if (in_range(oam[static_cast<size_t>(i) * 4]))
        r_.selected[static_cast<size_t>(r_.selected_count++)] = static_cast<uint8_t>(i);
  if (r_.overflow_dot != NEVER)
    return;
  for (int m = 0; n < 64; ++n, m = (m + 1) & 3, at += 2) {
    if (in_range(oam[static_cast<size_t>(n) * 4 + m])) {
      r_.overflow_dot = at;
      mark_status_change(at);
      return;
    }
  }
}

// ── Events ──

// MMC3 counts A12 rises that follow at least ~3 CPU cycles of A12 low
// (its M2 filter): once per line for the usual bg=$0000/sprites=$1000
// split, never for the 4-dot dips between background fetches.
bool Ppu::a12(bool high, int64_t dot) {
  if (!wants_a12_)
    return false;
  if (high) {
    if (!r_.a12_high) {
      r_.a12_high = true;
      if (dot - r_.a12_low_since > 10) {
        emit(PpuEvent::A12Rise, dot);
        return true;
      }
    }
  } else if (r_.a12_high) {
    r_.a12_high = false;
    r_.a12_low_since = dot;
  }
  return false;
}

void Ppu::emit(PpuEvent::Kind kind, int64_t dot) {
  PpuWatcher* watcher = dry_ ? probe_ : watcher_;
  if (!watcher)
    return;
  watcher->ppu_event({kind, dot, r_.line});
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
