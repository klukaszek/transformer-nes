// Independent conventional PPU oracle, compiled only by test targets.
// Production registers, rendering and sprite computation use transformer tokens.


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

void Ppu::load_sprites(int64_t dot) {
  if (!start_sprite_line())
    return;
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
    array<uint8_t, 8> pixels{};
    for (int c = 0; c < 8; ++c) {
      const int bit = (attr & 0x40) ? c : 7 - c;
      const int value = ((lo >> bit) & 1) | (((hi >> bit) & 1) << 1);
      pixels[static_cast<size_t>(c)] = value ? static_cast<uint8_t>(meta | value) : 0;
    }
    draw_sprite_row(x, pixels);
  }
  const int dummy = (h == 16 || (ctrl & 0x08)) ? 0x1000 : 0;
  for (int slot = std::min(count, 8); slot < 8; ++slot) {
    a12(false, dot + 8 * slot);
    a12(dummy != 0, dot + 8 * slot + 4);
  }
}

// Sprite evaluation (dots 65-256): select the next line's sprites in
// OAM order and run the overflow search with its timing and hardware
// bug: after eight matches, the diagonal scan treats non-Y bytes as Y.
void Ppu::evaluate_sprites(int64_t dot) {
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
  if (r_.overflow_dot == NEVER)
    search_overflow(n, at);
}

// The overflow search once eight sprites are found, from OAM index n at
// `at`: it checks byte m of each sprite as its Y, bumping m along with n
// on a miss (the diagonal scan), 2 dots per sprite.
void Ppu::search_overflow(int n, int64_t at) {
  const int h = (ctrl & 0x20) ? 16 : 8;
  for (int m = 0; n < 64; ++n, m = (m + 1) & 3, at += 2) {
    if (static_cast<unsigned>(r_.line - oam[static_cast<size_t>(n) * 4 + m]) <
        static_cast<unsigned>(h)) {
      r_.overflow_dot = at;
      mark_status_change(at);
      return;
    }
  }
}

bool Ppu::reference_process(const EventAt& e) {
  const int64_t dot = r_.pos;
  switch (e.ev) {
  case Ev::Slot:
    if (r_.line < 240)
      output_pixels(e.slot, dot);
    if (rendering_enabled()) {
      fetch_tile(e.slot + 2, dot);
      increment_x();
      if (e.slot == 31)
        increment_y();
    }
    return true;
  case Ev::Evaluate:
    r_.selected_count = 0;
    if (rendering_enabled())
      evaluate_sprites(dot);
    return true;
  case Ev::Sprites:
    load_sprites(dot);
    return true;
  case Ev::Prefetch:
    if (rendering_enabled()) {
      fetch_tile(0, dot);
      increment_x();
      fetch_tile(1, dot + 8);
      increment_x();
      a12(false, dot + 16);
    }
    return true;
  default:
    return false;
  }
}
