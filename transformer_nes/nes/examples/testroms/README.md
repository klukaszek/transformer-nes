# PPU test ROMs

`transformer_nes/nes/tests/test_ppu_roms.py` runs blargg's self-reporting
test ROMs from [christopherpow/nes-test-roms](https://github.com/christopherpow/nes-test-roms)
when they are present here (ROMs stay local, like the game ROMs):

- `mmc3_test_2/rom_singles/1-5*.nes`
- `ppu_vbl_nmi/rom_singles/*.nes`
- `oam_read/oam_read.nes`

The screen-reporting suites in the same repository also pass:
`sprite_hit_tests_2005.10.05` (11/11), `sprite_overflow_tests` (5/5),
`vbl_nmi_timing` (7/7), `mmc3_irq_tests` (all but the MMC3 rev A variant),
`blargg_ppu_tests_2005.09.15b` (all but `power_up_palette`, which checks
the author's console), and `scanline` (no error stars).
