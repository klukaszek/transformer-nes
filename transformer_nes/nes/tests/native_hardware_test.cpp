#include "nes/apu.h"
#include "nes/bus.h"
#include "nes/mappers/mappers.h"
#include "nes/ppu.h"
#include "nes/tape.h"
#include "ui/frame_clock.h"

#include <cassert>

using namespace transformer_nes::nes;

static vector<uint8_t> banks(size_t count, size_t bank_size) {
  vector<uint8_t> out;
  for (size_t bank = 0; bank < count; ++bank)
    out.insert(out.end(), bank_size, static_cast<uint8_t>(bank));
  return out;
}

// Backs the mapper's host with a plain 64 KiB array: pokes and region
// maps land in `memory`, the machine IRQ line is a no-op.
class ImageMachine final : public ConsoleMachine {
public:
  explicit ImageMachine(array<uint8_t, 0x10000>& memory) : memory_(memory) {}
  void attach(Device&, int, int, bool) override {}
  void set_nmi() override {}
  void set_irq() override {}
  void clear_irq() override {}
  void clear_read_lease() override {}
  MachineRun run_cycles(int64_t, function<bool()>, int64_t) override { return {}; }
  int read_ram(int a) const override { return memory_[static_cast<size_t>(a)]; }
  void poke(int a, int v) override { memory_[static_cast<size_t>(a)] = static_cast<uint8_t>(v); }
  void map_region(int base, span<const uint8_t> data) override {
    std::copy(data.begin(), data.end(), memory_.begin() + base);
  }

private:
  array<uint8_t, 0x10000>& memory_;
};

static void wire(Mapper& mapper, array<uint8_t, 0x10000>& memory, ImageMachine& machine, Ppu& ppu,
                 IrqLine& irq) {
  auto image = mapper.cpu_image();
  std::copy(image.begin(), image.end(), memory.begin());
  irq.machine = &machine;
  mapper.connect({&ppu, &machine, &irq});
}

static void test_mappers() {
  array<uint8_t, 0x10000> memory{};
  ImageMachine machine(memory);
  Ppu ppu(vector<uint8_t>(0x2000), true, false);
  IrqLine irq;

  Cartridge c1{banks(16, BANK16), {}, Mirroring::Horizontal, 1};
  Mmc1 mmc1(c1);
  wire(mmc1, memory, machine, ppu, irq);
  for (int bit = 0; bit < 5; ++bit)
    mmc1.write(0xE000, (5 >> bit) & 1);
  assert((mmc1.prg_banks() == array<size_t, 2>{5, 15}));
  assert(memory[0x8000] == 5 && memory[0xE000] == 15);

  Mmc1 mmc1_rmw(c1);
  wire(mmc1_rmw, memory, machine, ppu, irq);
  // RMW clocks the serial port with the mapped ROM byte (bank 15, bit
  // 1), not the CPU's final value (bit 0).
  for (int bit = 0; bit < 5; ++bit)
    mmc1_rmw.write_rmw(0xE000, 0);
  assert((mmc1_rmw.prg_banks() == array<size_t, 2>{15, 15}));

  Cartridge c3{banks(32, BANK8), vector<uint8_t>(0x20000), Mirroring::Horizontal, 4};
  Mmc3 mmc3(c3);
  wire(mmc3, memory, machine, ppu, irq);
  mmc3.write(0x8000, 6);
  mmc3.write(0x8001, 12);
  assert((mmc3.prg_banks() == array<size_t, 4>{12, 1, 30, 31}));
  assert(memory[0x8000] == 12 && memory[0xE000] == 31);
  mmc3.write(0xA000, 0);
  assert(ppu.mirroring == 2);
  mmc3.write(0xA000, 1);
  assert(ppu.mirroring == 3);

  Cartridge c3_ram{banks(32, BANK8), {}, Mirroring::Horizontal, 4};
  Mmc3 mmc3_ram(c3_ram);
  wire(mmc3_ram, memory, machine, ppu, irq);
  mmc3_ram.write(0x8000, 0);
  mmc3_ram.write(0x8001, 6);
  assert(ppu.chr_bank[0] == 6 * CHR_BANK1 && ppu.chr_bank[1] == 7 * CHR_BANK1);
  auto set_ppu_addr = [&ppu](int address) {
    ppu.write(0x2006, address >> 8, 0);
    ppu.write(0x2006, address & 0xFF, 0);
  };
  ppu.chr[6 * CHR_BANK1] = 0xA6;
  set_ppu_addr(0);
  assert(ppu.read(0x2007, 0) == 0);
  assert(ppu.read(0x2007, 0) == 0xA6);
  ppu.chr[2 * CHR_BANK1] = 0xB2;
  mmc3_ram.write(0x8001, 2);
  set_ppu_addr(0);
  assert(ppu.read(0x2007, 0) == 0);
  assert(ppu.read(0x2007, 0) == 0xB2);
  set_ppu_addr(0);
  ppu.write(0x2007, 0xC2, 0);
  assert(ppu.chr[2 * CHR_BANK1] == 0xC2);

  // The IRQ counter is clocked by the PPU's A12 rises: a clock reloads
  // it from the latch at 0 (or after a $C001 clear), otherwise
  // decrements, and reaching 0 with IRQs enabled asserts the line.
  const PpuEvent rise{PpuEvent::A12Rise, 0, 0};
  mmc3.write(0xC000, 2);
  mmc3.write(0xC001, 0);
  mmc3.write(0xE001, 0);
  auto probe = mmc3.irq_probe();
  assert(probe);
  mmc3.ppu_event(rise); // reload -> 2
  mmc3.ppu_event(rise); // 1
  assert(irq.sources == 0);
  mmc3.ppu_event(rise); // 0: fire
  assert(irq.sources == IRQ_MAPPER);
  // The probe replays the same counter without touching the mapper.
  for (int i = 0; i < 3; ++i)
    probe->ppu_event({PpuEvent::A12Rise, 100 + i, 0});
  assert(probe->fire_dot == 102);
  mmc3.write(0xE000, 0); // $E000 disable also acks the line
  assert(irq.sources == 0 && !mmc3.irq_probe());
  // Latch 0: every clock reloads to 0 and fires (Sharp MMC3).
  mmc3.write(0xE001, 0);
  mmc3.write(0xC000, 0);
  mmc3.ppu_event(rise);
  assert(irq.sources == IRQ_MAPPER);
  mmc3.write(0xE000, 0);

  Cartridge c5{banks(32, BANK8), vector<uint8_t>(0x20000), Mirroring::Horizontal, 5};
  Mmc5 mmc5(c5);
  wire(mmc5, memory, machine, ppu, irq);
  mmc5.write(0x5100, 2);
  mmc5.write(0x5115, 0x98);
  mmc5.write(0x5116, 0x9E);
  assert((mmc5.prg_banks() == array<size_t, 4>{24, 25, 30, 127}));
  mmc5.write(0x5205, 13);
  mmc5.write(0x5206, 17);
  assert(mmc5.read(0x5205) == 221 && mmc5.read(0x5206) == 0);
}

class FakeRuntime final : public SteppedRuntime {
public:
  int pc() const override { return pc_; }
  bool halted() const override { return false; }
  void set_nmi() override {}
  void set_irq() override {}
  void clear_irq() override {}
  void poke(int addr, int value) override { memory[addr] = static_cast<uint8_t>(value); }
  Next peek() const override { return next; }
  CpuStep step() override {
    CpuStep out;
    out.pc = pc_;
    out.writes = next_writes;
    next_writes.clear();
    pc_ += 3;
    return out;
  }
  int pc_ = 0x8000;
  Next next;
  vector<CpuWrite> next_writes;
  array<uint8_t, 0x10000> memory{};
};

class CounterDevice final : public Device {
public:
  int read(int, int64_t) override { return ++value; }
  void write(int, int next, int64_t) override { value = next; }
  void write_rmw(int, int next, int64_t cycle) override {
    rmw_cycle = cycle;
    value = next;
  }
  int value = 0;
  int64_t rmw_cycle = -1;
};

static void test_bus_and_controller() {
  FakeRuntime runtime;
  vector<uint8_t> image(0x10000);
  image[0x8000] = 0xAD;
  image[0x8001] = 0x00;
  image[0x8002] = 0xD0;
  runtime.next = {4, 0xD000}; // LDA $D000
  Bus bus(runtime, image);
  CounterDevice device;
  bus.attach(device, 0xD000, 0xD000);
  bus.step();
  assert(runtime.memory[0xD000] == 1 && bus.read_ram(0xD000) == 1);
  assert(bus.cycles() == 4);

  FakeRuntime rmw_runtime;
  image[0x8000] = 0xEE; // INC $D000
  image[0x8001] = 0x00;
  image[0x8002] = 0xD0;
  image[0xD000] = 0x41;
  rmw_runtime.next = {6, 0xD000};
  Bus rmw_bus(rmw_runtime, image);
  CounterDevice rmw_device;
  rmw_runtime.next_writes.push_back({0xD000, 0x42});
  rmw_bus.attach(rmw_device, 0xD000, 0xD000, true);
  rmw_bus.step();
  assert(rmw_device.rmw_cycle == 6 && rmw_device.value == 0x42);

  FakeRuntime mirror_runtime;
  image[0x8000] = 0xAD; // LDA $1803
  image[0x8001] = 0x03;
  image[0x8002] = 0x18;
  image[0x0003] = 0xA7;
  image[0x1803] = 0;
  mirror_runtime.next = {4, 0x1803};
  Bus mirror_bus(mirror_runtime, image);
  assert(mirror_bus.read_ram(0x0003) == 0xA7);
  assert(mirror_bus.read_ram(0x0803) == 0xA7);
  assert(mirror_bus.read_ram(0x1003) == 0xA7);
  assert(mirror_bus.read_ram(0x1803) == 0xA7);

  Controller controller;
  controller.set_button(0, true);
  controller.write(0x4016, 1, 0);
  controller.write(0x4016, 0, 0);
  assert(controller.read(0x4016, 0) == 0x41);
  assert(controller.read(0x4016, 0) == 0x40);
}

// A PPU showing tile 1 (solid, color 1) everywhere, sprite 0 an
// opaque 8x8 block at OAM (y, x), rendering on from dot 0.
static Ppu solid_screen(int sprite_y, int sprite_x) {
  vector<uint8_t> chr(0x2000);
  for (int i = 0; i < 8; ++i)
    chr[16 + i] = 0xFF;
  Ppu ppu(chr, false, true);
  ppu.write(0x2006, 0x20, 0);
  ppu.write(0x2006, 0x00, 0);
  for (int i = 0; i < 960; ++i)
    ppu.write(0x2007, 1, 0);
  ppu.oam.fill(0xF0);
  ppu.oam[0] = static_cast<uint8_t>(sprite_y);
  ppu.oam[1] = 1;
  ppu.oam[2] = 0;
  ppu.oam[3] = static_cast<uint8_t>(sprite_x);
  ppu.write(0x2006, 0, 0);
  ppu.write(0x2006, 0, 0);
  ppu.write(0x2001, 0x1E, 0);
  return ppu;
}

static void test_ppu() {
  Ppu ppu(vector<uint8_t>(0x2000), true, true);
  ppu.write(0x2006, 0x3F, 0);
  ppu.write(0x2006, 0x00, 0);
  ppu.write(0x2007, 0x21, 0);
  assert(ppu.palette[0] == 0x21);
  array<uint8_t, 256> dma{};
  dma[0] = 0x12;
  ppu.oam_dma(dma, 0);
  assert(ppu.oam[0] == 0x12);
  ppu.write(0x2007, 0x16, 0); // $3F01; v now parks on $3F02
  ppu.catch_up(Ppu::frame_dots);
  assert(ppu.frame_start() == Ppu::frame_dots);
  // Rendering off with v in palette RAM shows that entry (the backdrop
  // override); elsewhere, the backdrop.
  assert(ppu.completed_frame()[0] == 0x00);
  ppu.write(0x2006, 0x3F, Ppu::frame_dots);
  ppu.write(0x2006, 0x01, Ppu::frame_dots);
  ppu.catch_up(2 * Ppu::frame_dots);
  assert(ppu.completed_frame()[0] == 0x16);
  ppu.write(0x2006, 0x20, 2 * Ppu::frame_dots);
  ppu.write(0x2006, 0x00, 2 * Ppu::frame_dots);
  ppu.catch_up(3 * Ppu::frame_dots);
  assert(ppu.completed_frame()[0] == 0x21);

  // Loopy t: $2000 sets the nametable bits, $2005 x/y the scroll bits,
  // and the second $2006 write copies t into v.
  Ppu loopy(vector<uint8_t>(0x2000), false, true);
  loopy.write(0x2000, 0x03, 0);
  loopy.write(0x2005, 0x7D, 0); // coarse x 15, fine x 5
  loopy.write(0x2005, 0x5E, 0); // coarse y 11, fine y 6
  assert(loopy.t == (0x0C00 | (6 << 12) | (11 << 5) | 15) && loopy.fine_x == 5);
  assert(!loopy.write_latch);
  loopy.write(0x2006, 0x04, 0);
  // The high write replaces bits 8-14 (clearing fine y and bit 14).
  assert(loopy.t == ((4 << 8) | (((11 << 5) | 15) & 0xFF)));
  (void)loopy.read(0x2002, 0);                     // resets the shared latch
  assert(!loopy.write_latch);

  // Vblank sets at line 241 dot 1; NMI follows when enabled.
  Ppu status(vector<uint8_t>(0x2000), false, true);
  status.write(0x2000, 0x80, 0);
  status.catch_up(status.vblank_dot() + 1);
  assert(status.take_nmi_edge());
  assert((status.read(0x2002, status.vblank_dot() + 20) & 0x80) != 0);
  // Enabling NMI while the flag is set fires after the next instruction.
  Ppu late(vector<uint8_t>(0x2000), false, true);
  late.catch_up(late.vblank_dot() + 1);
  late.write(0x2000, 0x80, late.vblank_dot() + 10);
  assert(late.take_delayed_nmi() && !late.take_nmi_edge());
  // Reading $2002 one dot before the flag sets reads it clear and
  // cancels the flag and the NMI for that frame.
  Ppu race(vector<uint8_t>(0x2000), false, true);
  race.write(0x2000, 0x80, 0);
  assert((race.read(0x2002, race.vblank_dot()) & 0x80) == 0);
  race.catch_up(race.vblank_dot() + 100);
  assert(!race.take_nmi_edge() && (race.read(0x2002, race.vblank_dot() + 100) & 0x80) == 0);

  // Sprite 0 hits at its first opaque pixel over opaque background:
  // OAM y=100 shows on line 101, x=50 outputs at dot 51.
  Ppu hit = solid_screen(100, 50);
  hit.catch_up(Ppu::frame_dots); // a full frame so every line has fetched
  const int64_t base = hit.frame_start();
  const int64_t at = base + 101 * 341 + 51;
  assert(hit.preview(base + Ppu::frame_dots, nullptr, true) == at);
  assert(hit.position() == base); // the preview left the PPU untouched
  assert((hit.read(0x2002, at - 1) & 0x40) == 0);
  assert((hit.read(0x2002, at) & 0x40) != 0);
  // ... never at x=255.
  Ppu edge = solid_screen(100, 255);
  edge.catch_up(Ppu::frame_dots);
  assert(edge.preview(edge.frame_start() + Ppu::frame_dots, nullptr, true) ==
         edge.frame_start() + 241 * 341 + 1); // first change: vblank

  // bg at $0000, sprites at $1000: one filtered A12 rise per rendered
  // line (240 visible + pre-render), at dot 260.
  struct Rises final : PpuWatcher {
    vector<int64_t> dots;
    void ppu_event(const PpuEvent& e) override {
      if (e.kind == PpuEvent::A12Rise)
        dots.push_back(e.dot);
    }
  } rises;
  Ppu clocked = solid_screen(100, 50);
  clocked.write(0x2000, 0x08, 0);
  clocked.set_watcher(&rises, true);
  clocked.catch_up(Ppu::frame_dots);
  const int64_t frame1 = clocked.frame_start();
  rises.dots.clear();
  clocked.catch_up(2 * Ppu::frame_dots - 1); // odd frame 1 skips a dot
  assert(clocked.frame_start() == 2 * Ppu::frame_dots - 1);
  assert(rises.dots.size() == 241);
  assert(rises.dots[0] == frame1 + 261 && rises.dots[1] == frame1 + 341 + 261);
}

static void test_apu() {
  Apu apu(44'100.0);
  apu.write(0x4015, 0);
  apu.advance_to(29'781);
  auto audio = apu.take_samples();
  assert(audio.size() >= 733 && audio.size() <= 734);
  assert(apu.take_dma_stall() == 0);

  Apu chunked(44'100.0);
  chunked.advance_to(10'000);
  const auto first = chunked.take_samples();
  chunked.advance_to(29'781);
  const auto second = chunked.take_samples();
  assert(first.size() + second.size() >= 733 && first.size() + second.size() <= 734);
}

static void test_tape() {
  auto tape = InputTape::parse("2 08 start\n5 00 (none)\n9 81 a+right\n");
  assert(tape.last_frame() == 9);
  assert(tape.events.size() == 3);
  assert(tape.buttons_at(0) == 0);
  assert(tape.buttons_at(2) == 0x08);
  assert(tape.buttons_at(8) == 0);
  assert(tape.buttons_at(9) == 0x81);
}

static void test_frame_clock() {
  using transformer_nes::ui::FrameClock;
  constexpr double period = 29'781.0 / 1'789'772.727272727;
  FrameClock clock(period);
  // A 6 ms emulation fits a 120 Hz display. Crossing the former half-frame
  // budget must not erase the phase remaining after an emulated frame.
  clock.advance(1.75 * period);
  assert(clock.ready());
  clock.consume();
  clock.discard_backlog();
  clock.advance(0.30 * period);
  assert(clock.ready());
  clock.consume();
  assert(!clock.ready());
  // Large stalls retain at most four complete frames plus fractional phase.
  clock.advance(100.5 * period);
  int due = 0;
  while (clock.ready()) { clock.consume(); ++due; }
  assert(due == 4);
  clock.advance(-1.0);
  clock.advance(0.5 * period);
  assert(clock.ready());
  // An uncapped wall clock keeps NTSC rate across 60/120/144 Hz displays.
  for (double hz : {60.0, 120.0, 144.0}) {
    FrameClock paced(period);
    int frames = 0;
    for (int display = 0; display < int(10 * hz); ++display) {
      paced.advance(1.0 / hz);
      while (paced.ready()) { paced.consume(); ++frames; }
    }
    assert(frames == int(10.0 / period));
  }
}

int main() {
  test_mappers();
  test_bus_and_controller();
  test_ppu();
  test_apu();
  test_tape();
  test_frame_clock();
}
