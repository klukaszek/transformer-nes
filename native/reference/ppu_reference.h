// Private declarations for the test-only conventional PPU oracle.
  static const vector<EventAt>& reference_events_for(int line);
  void reference_advance_line();
  static size_t palette_index(int address);
  bool reference_latch_fetch(size_t address);
  void reference_oam_dma(span<const uint8_t, 256> page);
  RegisterResult transition(const RegisterAccess& request);
  bool reference_process(const EventAt& event);
  bool reference_control(ControlEvent event, int64_t dot, const RegisterResult& effects);
  static size_t nt_offset_for(int address, int mirroring);
  int nt_read(int addr) {
    if (mmc5.active()) {
      const int a = (addr - 0x2000) & 0xFFF;
      const int source=(mirroring >> (2 * (a >> 10))) & 3;
      return source<2 ? nametables[static_cast<size_t>(source*0x400+(a&0x3FF))]
                      : mmc5.page_view(source,a&0x3FF);
    }
    return nametables[nt_offset_for(addr, mirroring)];
  }
  size_t chr_index(const size_t* banks, size_t address) const {
    if(chr_latch_active) {
      const size_t half=(address>>12)&1;
      return chr_latch.banks[half*2+(chr_latch.latch_fe[half]?1:0)] + (address&0xFFF);
    }
    return banks[(address >> 10) & 7] + (address & 0x3FF);
  }
  std::pair<uint8_t, uint8_t> fetch_pattern(const size_t* banks, int pattern_addr);
  int vram_peek(int address);
  void vram_write(int address, int value);
  uint8_t backdrop() const;
  int increment() const { return (ctrl & 0x04) ? 32 : 1; }
  void step_vram_address(bool rendering);
  void increment_x();
  void increment_y();
  void fetch_tile(int index, int64_t dot);
  void output_pixels(int slot, int64_t dot);
  void evaluate_sprites(int64_t dot);
  void load_sprites(int64_t dot);
  void draw_sprite_row(int x, span<const uint8_t, 8> row);
  void search_overflow(int n, int64_t at);
