#include "nes/apu.h"
#include "nes/bus.h"
#include "nes/mappers/mappers.h"
#include "nes/ppu.h"
#include "nes/tape.h"

#include <cassert>

using namespace transformer_nes::nes;

static vector<uint8_t> banks(size_t count, size_t bank_size) {
  vector<uint8_t> out;
  for (size_t bank = 0; bank < count; ++bank)
    out.insert(out.end(), bank_size, static_cast<uint8_t>(bank));
  return out;
}

static void wire(Mapper& mapper, array<uint8_t, 0x10000>& memory) {
  auto image = mapper.cpu_image();
  std::copy(image.begin(), image.end(), memory.begin());
  MapperCallbacks callbacks;
  callbacks.poke = [&](int addr, int value) { memory[addr] = static_cast<uint8_t>(value); };
  callbacks.map_region = [&](int base, span<const uint8_t> data) {
    std::copy(data.begin(), data.end(), memory.begin() + base);
  };
  mapper.connect(std::move(callbacks));
}

static void test_mappers() {
  Cartridge c1{banks(16, BANK16), {}, Mirroring::Horizontal, 1};
  Mmc1 mmc1(c1);
  array<uint8_t, 0x10000> memory{};
  wire(mmc1, memory);
  for (int bit = 0; bit < 5; ++bit)
    mmc1.write(0xE000, (5 >> bit) & 1, -1);
  assert((mmc1.prg_banks() == array<size_t, 2>{5, 15}));
  assert(memory[0x8000] == 5 && memory[0xE000] == 15);

  Mmc1 mmc1_rmw(c1);
  wire(mmc1_rmw, memory);
  for (int bit = 0; bit < 5; ++bit)
    mmc1_rmw.write_rmw(0xE000, (3 >> bit) & 1, ((3 >> bit) & 1) ^ 1);
  assert((mmc1_rmw.prg_banks() == array<size_t, 2>{3, 15}));

  Cartridge c3{banks(32, BANK8), vector<uint8_t>(0x20000), Mirroring::Horizontal, 4};
  Mmc3 mmc3(c3);
  wire(mmc3, memory);
  MapperCallbacks callbacks;
  callbacks.poke = [&](int addr, int value) { memory[addr] = static_cast<uint8_t>(value); };
  callbacks.map_region = [&](int base, span<const uint8_t> data) {
    std::copy(data.begin(), data.end(), memory.begin() + base);
  };
  callbacks.chr_banks = [](const ChrBanks&) {};
  int mirroring = -1;
  int irq_acks = 0;
  callbacks.mirroring = [&](int value) { mirroring = value; };
  callbacks.irq_ack = [&] { ++irq_acks; };
  mmc3.connect(std::move(callbacks));
  mmc3.write(0x8000, 6, -1);
  mmc3.write(0x8001, 12, -1);
  assert((mmc3.prg_banks() == array<size_t, 4>{12, 1, 30, 31}));
  assert(memory[0x8000] == 12 && memory[0xE000] == 31);
  mmc3.write(0xA000, 0, -1);
  assert(mirroring == 2);
  mmc3.write(0xA000, 1, -1);
  assert(mirroring == 3);

  mmc3.write(0xC000, 2, -1);
  mmc3.write(0xC001, 0, -1);
  mmc3.write(0xE001, 0, -1);
  assert(mmc3.irq_scanline(-1) == 2);
  mmc3.notify_scanline_irq();
  // The latch is a delta from the previous expiry: chained IRQs fire
  // latch+1 lines apart, and past the visible frame nothing schedules.
  assert(mmc3.irq_scanline(2) == 5);
  assert(!mmc3.irq_scanline(237));
  mmc3.write(0xE000, 0, -1);
  assert(irq_acks == 1);
  assert(!mmc3.irq_scanline(-1));
  mmc3.write(0xE001, 0, -1);
  assert(mmc3.irq_scanline(-1) == 2);
  // Latch 0 = fire-next-scanline, valid only chained or off an anchor.
  mmc3.write(0xC000, 0, -1);
  assert(!mmc3.irq_scanline(-1));
  assert(mmc3.irq_scanline(100) == 101);
  // A mid-frame $C001 clear anchors the reload at its scanline; the
  // anchor resets at frame start.
  mmc3.write(0xC000, 30, -1);
  mmc3.write(0xC001, 0, 120);
  assert(mmc3.irq_scanline(-1) == 151);
  assert(mmc3.irq_scanline(90) == 151);
  assert(mmc3.irq_scanline(160) == 191);
  mmc3.begin_frame();
  assert(mmc3.irq_scanline(-1) == 30);

  Cartridge c5{banks(32, BANK8), vector<uint8_t>(0x20000), Mirroring::Horizontal, 5};
  Mmc5 mmc5(c5);
  wire(mmc5, memory);
  mmc5.write(0x5100, 2, -1);
  mmc5.write(0x5115, 0x98, -1);
  mmc5.write(0x5116, 0x9E, -1);
  assert((mmc5.prg_banks() == array<size_t, 4>{24, 25, 30, 127}));
  mmc5.write(0x5205, 13, -1);
  mmc5.write(0x5206, 17, -1);
  assert(mmc5.read(0x5205) == 221 && mmc5.read(0x5206) == 0);
}

class FakeRuntime final : public SteppedRuntime {
public:
  int pc() const override { return pc_; }
  int x() const override { return 0; }
  int y() const override { return 0; }
  bool halted() const override { return false; }
  bool nmi_pending() const override { return nmi_; }
  bool irq_ready() const override { return false; }
  void set_nmi() override { nmi_ = true; }
  void set_irq() override {}
  void clear_irq() override {}
  void poke(int addr, int value) override { memory[addr] = static_cast<uint8_t>(value); }
  CpuStep step() override {
    CpuStep out;
    out.pc = pc_;
    out.writes = next_writes;
    next_writes.clear();
    pc_ += 3;
    nmi_ = false;
    return out;
  }
  int pc_ = 0x8000;
  bool nmi_ = false;
  vector<CpuWrite> next_writes;
  array<uint8_t, 0x10000> memory{};
};

class CounterDevice final : public Device {
public:
  int read(int) override { return ++value; }
  void write(int, int next, int64_t) override { value = next; }
  void write_rmw(int, int original, int next) override {
    rmw_original = original;
    value = next;
  }
  int value = 0;
  int rmw_original = -1;
};

static void test_bus_and_controller() {
  FakeRuntime runtime;
  vector<uint8_t> image(0x10000);
  image[0x8000] = 0xAD;
  image[0x8001] = 0x00;
  image[0x8002] = 0xD0;
  OpcodeTable opcodes{};
  opcodes[0xAD] = {4, 3};
  Bus bus(runtime, image, opcodes);
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
  opcodes[0xEE] = {6, 3};
  Bus rmw_bus(rmw_runtime, image, opcodes);
  CounterDevice rmw_device;
  rmw_runtime.next_writes.push_back({0xD000, 0x42});
  rmw_bus.attach(rmw_device, 0xD000, 0xD000, true);
  rmw_bus.step();
  assert(rmw_device.rmw_original == 0x41 && rmw_device.value == 0x42);

  Controller controller;
  controller.set_button(0, true);
  controller.write(0x4016, 1, 0);
  controller.write(0x4016, 0, 0);
  assert(controller.read(0x4016) == 0x41);
  assert(controller.read(0x4016) == 0x40);
}

static void test_ppu_and_apu() {
  Ppu ppu(vector<uint8_t>(0x2000), true, true);
  ppu.write(0x2006, 0x3F);
  ppu.write(0x2006, 0x00);
  ppu.write(0x2007, 0x21);
  assert(ppu.palette[0] == 0x21);
  array<uint8_t, 256> dma{};
  dma[0] = 0x12;
  ppu.oam_dma(dma);
  assert(ppu.oam[0] == 0x12);
  auto frame = ppu.render_frame();
  assert(frame.size() == 240 * 256);

  Apu apu(44'100.0);
  apu.write(0x4015, 0);
  auto audio = apu.end_frame(29'781);
  assert(audio.size() >= 733 && audio.size() <= 734);
  assert(apu.take_dma_stall() == 0);
}

static void test_tape() {
  auto tape = InputTape::parse("2 08 start\n5 00 (none)\n9 81 a+right\n");
  assert(tape.last_frame() == 9);
  assert(tape.events().size() == 3);
  assert(tape.buttons_at(0) == 0);
  assert(tape.buttons_at(2) == 0x08);
  assert(tape.buttons_at(8) == 0);
  assert(tape.buttons_at(9) == 0x81);
}

int main() {
  test_mappers();
  test_bus_and_controller();
  test_ppu_and_apu();
  test_tape();
}
