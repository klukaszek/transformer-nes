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

// Conventional bus/view rules are confined to this independent oracle.
static void reference_bus(const int* in, int* out) {
  bool high=in[1];
  int age=in[2],stamp=0,rise=0;
  auto pin=[&](bool level,int at) {
    if(level) {
      if(!high) {high=true;if(age+at>10)rise=1;}
    } else if(high) {high=false;stamp=1;age=-at;}
  };
  if(in[0]==0)pin(in[3],0);
  else {
    pin(false,0);
    if(in[0]!=1 || !in[5])pin(in[0]==1 ? !!(in[4]&16) : !!in[3],4);
  }
  out[0]=high;out[1]=stamp;out[2]=rise;out[3]=in[0]==0?0:4;
}

static void reference_chr_latch(size_t address,bool wide,bool& lo,bool& hi) {
  address&=0x1FFF;
  if(address>=0x1FD8 && address<=0x1FDF)hi=false;
  else if(address>=0x1FE8 && address<=0x1FEF)hi=true;
  else if(wide ? address>=0x0FD8 && address<=0x0FDF : address==0x0FD8)lo=false;
  else if(wide ? address>=0x0FE8 && address<=0x0FEF : address==0x0FE8)lo=true;
}

static void reference_access_bus(const int* in,int* out) {
  const int reg=in[0],writing=in[1],ctrl=in[6],mask=in[7];
  bool lo=in[15],hi=in[16];
  const int address=((in[12]&3)<<12)|(in[11]<<10)|(in[10]<<5)|in[9];
  const bool loaded=reg==6 && writing && in[3] && !(mask&0x18);
  const bool stepped=reg==7 && !in[2];
  if(reg==7 && !writing && in[13] && address<0x2000)
    reference_chr_latch(address,in[14],lo,hi);
  out[0]=loaded||stepped;out[1]=in[8]&1;out[2]=lo;out[3]=hi;
  out[4]=lo!=in[15] || hi!=in[16] || in[4]!=ctrl;
  out[5]=((in[4]^ctrl)&0x38) || ((in[5]^mask)&0x18) || (reg==4 && writing && (ctrl&0x20));
  out[6]=reg==1 && writing;
}

static int reference_write_delay(int reg) { return reg==1 ? 1 : 0; }

static void reference_dma(const int* in, int* out) {
  const int packet[]{in[0] << 8, 513 + in[1], in[2], 256-in[2],
                     256-in[2], in[2], !!(in[3] & 0x20)};
  std::copy(std::begin(packet), std::end(packet), out);
}

static void reference_view(const int* in,int* out) {
  const int values[]{in[0]==1,in[0]<=1,in[1],in[2]*0x55,0x3C0,!!(in[3]&0x20)};
  std::copy(std::begin(values),std::end(values),out);
}

} // namespace

size_t Ppu::palette_index(int addr) {
  size_t offset = (addr - 0x3F00) & 0x1F;
  if (offset == 0x10 || offset == 0x14 || offset == 0x18 || offset == 0x1C)
    offset -= 0x10;
  return offset;
}

void Ppu::reference_oam_dma(span<const uint8_t, 256> page) {
  for (size_t i = 0; i < page.size(); ++i)
    oam[(oam_addr + i) & 0xFF] = page[i];
  oam_changed();
  if (ctrl & 0x20)
    schedule_changed_ = true;
}

bool Ppu::reference_latch_fetch(size_t address) {
  const bool lo=chr_latch.latch_fe[0], hi=chr_latch.latch_fe[1];
  reference_chr_latch(address,chr_latch.wide_low,chr_latch.latch_fe[0],chr_latch.latch_fe[1]);
  return lo!=chr_latch.latch_fe[0] || hi!=chr_latch.latch_fe[1];
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

std::pair<uint8_t, uint8_t> Ppu::fetch_pattern(const size_t* banks, int pattern_addr) {
  if (!chr_latch_active) {
    const uint8_t* bank = chr.data() + banks[(pattern_addr >> 10) & 7] + (pattern_addr & 0x3FF);
    return {bank[0], bank[8]};
  }
  // The FD/FE tile itself still fetches through the old bank: both
  // pattern bytes are read before either fetch flips its latch.
  const uint8_t lo = chr[chr_index(banks, pattern_addr)];
  const uint8_t hi = chr[chr_index(banks, pattern_addr + 8)];
  const bool flipped_lo = reference_latch_fetch(pattern_addr);
  if (reference_latch_fetch(pattern_addr + 8) || flipped_lo)
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
      const int source=(mirroring >> (2 * (a >> 10))) & 3;
      if(source==2)mmc5.write_exram(a&0x3FF,value);
      else if(source<2)nametables[static_cast<size_t>(source*0x400+(a&0x3FF))]=static_cast<uint8_t>(value);
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
// address, and the $2007 read buffer. Memory writes are applied here.
Ppu::RegisterResult Ppu::transition(const RegisterAccess& request) {
  RegisterResult out;
  const int value = request.value;
  switch (request.reg + (request.is_write ? 8 : 0)) {
  case 8: // PPUCTRL
    out.nmi_rise = (value & 0x80) && !(ctrl & 0x80);
    out.nmi_fall = !(value & 0x80) && (ctrl & 0x80);
    ctrl = value;
    t = static_cast<uint16_t>((t & ~0x0C00) | ((value & 3) << 10));
    break;
  case 9: // PPUMASK
    mask = value;
    break;
  case 2: // PPUSTATUS
    out.clear_vblank = true;
    out.read_value = request.status;
    write_latch = false;
    break;
  case 11: // OAMADDR
    oam_addr = static_cast<uint8_t>(value);
    break;
  case 12: // OAMDATA
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
    const size_t base = static_cast<size_t>(((mmc5.chr_upper() << 6) | (mmc5.exram(offset) & 0x3F)) * 0x1000) + static_cast<size_t>(tile_no) * 16 + fine_y;
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

// Independent conventional composition, excluded from production.
void Ppu::draw_sprite_row(int x, span<const uint8_t, 8> row) {
  for (int c = 0; c < 8 && x + c < 256; ++c) {
    auto& pixel = r_.sprites[static_cast<size_t>(x + c)];
    if (!row[c] || pixel)
      continue;
    pixel = row[c];
    r_.sprite_slots[static_cast<size_t>((x + c) >> 3)] = 1;
    r_.sprite0_on_line |= (row[c] & 0x80) != 0;
  }
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
  const bool sprites_here = sprites_on && r_.sprite_slots[static_cast<size_t>(slot)];
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

// Independent conventional event control. Production executes control.py.
bool Ppu::reference_control(ControlEvent event, int64_t dot, const RegisterResult& effects) {
  switch (event) {
  case ControlEvent::Access:
    if (effects.clear_vblank) {
      if (dot == vblank_dot() && r_.vblank_set_dot < 0)
        r_.vblank_suppressed = true;
      if (r_.vblank_set_dot >= 0 && dot - r_.vblank_set_dot <= 2)
        r_.nmi_edge = false;
      r_.vblank = false;
    }
    if (effects.nmi_rise && r_.vblank &&
        dot != r_.frame_start + int64_t{prerender_line} * dots_per_line + 1)
      r_.nmi_delayed = true;
    if (effects.nmi_fall && r_.vblank_set_dot >= 0 && dot - r_.vblank_set_dot <= 2)
      r_.nmi_edge = false;
    break;
  case ControlEvent::VerticalCopy:
    if (rendering_enabled())
      r_.v = static_cast<uint16_t>((r_.v & ~0x7BE0) | (t & 0x7BE0));
    break;
  case ControlEvent::SpriteLine:
    if (rendering_enabled()) {
      r_.v = static_cast<uint16_t>((r_.v & ~0x041F) | (t & 0x041F));
      oam_addr = 0;
    }
    break;
  case ControlEvent::OddSkip:
    if (r_.odd && rendering_enabled())
      r_.line_len = dots_per_line - 1;
    break;
  case ControlEvent::VBlank:
    if (std::exchange(r_.vblank_suppressed, false))
      break;
    r_.vblank = true;
    r_.vblank_set_dot = dot;
    if (ctrl & 0x80)
      r_.nmi_edge = true;
    mark_status_change(dot);
    break;
  case ControlEvent::Clear:
    if (r_.vblank || r_.hit_dot != NEVER || r_.overflow_dot != NEVER)
      mark_status_change(dot);
    r_.vblank = false;
    r_.vblank_set_dot = -1;
    r_.hit_dot = r_.overflow_dot = NEVER;
    break;
  }
  return rendering_enabled();
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
