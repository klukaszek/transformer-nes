# PPU test ROMs

`transformer_nes/nes/tests/test_ppu_roms.py` runs blargg's self-reporting
test ROMs with both transformer CPU and PPU, from [christopherpow/nes-test-roms](https://github.com/christopherpow/nes-test-roms).
Fetch the supported suites at the pinned revision and run them with:

```sh
uv run python -m transformer_nes.nes.tests.test_ppu_roms
uv run pytest transformer_nes/nes/tests/test_ppu_roms.py
```

The downloaded ROMs remain ignored. Local tests skip missing fixtures;
CI fetches and requires all 16:

- `mmc3_test_2/rom_singles/1-5*.nes`
- `ppu_vbl_nmi/rom_singles/*.nes`
- `oam_read/oam_read.nes`

The conventional PPU reference was also checked against the screen-reporting
suites in the same repository:
`sprite_hit_tests_2005.10.05` (11/11), `sprite_overflow_tests` (5/5),
`vbl_nmi_timing` (7/7), `mmc3_irq_tests` (all but the MMC3 rev A variant),
`blargg_ppu_tests_2005.09.15b` (all but `power_up_palette`, which checks
the author's console), and `scanline` (no error stars).
