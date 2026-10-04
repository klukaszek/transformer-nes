// Private declarations for the test-only conventional PPU oracle.
  RegisterResult transition(const RegisterAccess& request);
  bool reference_process(const EventAt& event);
  static size_t nt_offset_for(int address, int mirroring);
  int nt_read(int addr) {
    if (mmc5.active()) {
      const int a = (addr - 0x2000) & 0xFFF;
      return mmc5.read_nametable(mirroring, a >> 10, a & 0x3FF, nametables.data());
    }
    return nametables[nt_offset_for(addr, mirroring)];
  }
  size_t chr_index(const size_t* banks, size_t address) const {
    return chr_latch_active ? chr_latch.map(address)
                           : banks[(address >> 10) & 7] + (address & 0x3FF);
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
  void search_overflow(int n, int64_t at);
