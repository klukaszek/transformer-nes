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

// Independent conventional oracle for this stand-alone hardware test.
// Production consoles attach the compiled mappers.py weights instead.
static void reference_mapper_irq(void*, int model, const int* in, int* out) {
  const bool mmc5 = model / MapperIrq::event_count == MapperIrq::Mmc5;
  const int event = model % MapperIrq::event_count;
  int counter = in[0], target = in[1], phase = in[2], pending = in[3], enabled = in[4];
  int fire = 0, read = 0;
  switch (event) {
  case MapperIrq::Clock:
    if (mmc5) {
      counter = phase ? counter + 1 : 0;
      phase = 1;
      fire = target && counter == target;
    } else {
      if (!counter || phase) {
        counter = target;
        phase = 0;
      } else {
        --counter;
      }
      fire = enabled && !counter;
    }
    pending |= fire;
    break;
  case MapperIrq::Idle:
    if (mmc5)
      phase = 0;
    break;
  case MapperIrq::Target:
    target = in[6];
    break;
  case MapperIrq::Reload:
    if (!mmc5) {
      counter = 0;
      phase = 1;
    }
    break;
  case MapperIrq::Enable:
    enabled = mmc5 ? !!(in[6] & 0x80) : 1;
    break;
  case MapperIrq::Disable:
    enabled = pending = 0;
    break;
  case MapperIrq::Ack:
    if (mmc5)
      read = 128 * pending + 64 * phase;
    pending = 0;
    break;
  }
  int distance = 0;
  if (mmc5) {
    if (enabled && target && !pending)
      distance = !phase ? target + 1 : std::max(0, target - counter);
  } else if (enabled) {
    distance = !counter || phase ? target + 1 : counter;
  }
  const int predictable = enabled && (!mmc5 || (target && !pending));
  const bool drive = (fire && enabled) || event == MapperIrq::Disable ||
                     (mmc5 && (event == MapperIrq::Enable || event == MapperIrq::Ack));
  const int result[]{counter, target, phase, pending, enabled, enabled && pending,
                     read, predictable, distance, drive};
  std::copy(std::begin(result), std::end(result), out);
}

// A conventional oracle, confined to this standalone test executable.
// Model 0 resets, 1 writes, 2 applies an RMW, and 3 projects the bank map.
static void reference_mapper_banks(void* context, int model, const int* in, int* out) {
  const auto& b = *static_cast<BankControl*>(context);
  const int n = b.state_count;
  if (model == 3) {
    if (b.chip == BankControl::Mmc1) {
      const int mode = (in[0] >> 2) & 3;
      const int bank = (in[1] & 15) % b.n_prg;
      out[0] = mode < 2 ? bank & ~1 : mode == 2 ? 0 : bank;
      out[1] = mode < 2 ? (bank & ~1) + 1 : mode == 2 ? bank : b.n_prg - 1;
      for (int i = 0; i < 8; ++i)
        out[2 + i] = (in[0] & 0x10) ? (in[2 + i / 4] * 4 + i % 4) % b.n_chr
                                             : ((in[2] & 0x1E) * 4 + i) % b.n_chr;
      out[10] = in[0] & 3;
    } else if (b.chip == BankControl::Mmc3) {
      const int a = in[9] % b.n_prg, c = in[10] % b.n_prg;
      out[0] = in[1] ? static_cast<int>(b.n_prg) - 2 : a;
      out[1] = c;
      out[2] = in[1] ? a : static_cast<int>(b.n_prg) - 2;
      out[3] = b.n_prg - 1;
      const int pair[]{(in[3] & 0xFE), (in[3] & 0xFE) + 1, (in[4] & 0xFE), (in[4] & 0xFE) + 1};
      for (int i = 0; i < 4; ++i) {
        out[4 + i] = (in[2] ? in[5 + i] : pair[i]) % b.n_chr;
        out[8 + i] = (in[2] ? pair[i] : in[5 + i]) % b.n_chr;
      }
      out[12] = in[11];
    } else if (b.chip == BankControl::Mmc5) {
      const int src[4][4]{{3,3,3,3},{1,1,3,3},{1,1,2,3},{0,1,2,3}};
      const int spans[4][4]{{4,4,4,4},{2,2,2,2},{2,2,1,1},{1,1,1,1}};
      for (int i = 0; i < 4; ++i) {
        const int j = src[in[0]][i], span = spans[in[0]][i];
        const int bank = ((in[9 + j] & 127) & ~(span - 1)) + i % span;
        out[21 + i] = j != 3 && !(in[9 + j] & 128);
        out[i] = bank % (out[21 + i] ? 8 : b.n_prg);
      }
      for (int bg = 0; bg < 2; ++bg)
        for (int i = 0; i < 8; ++i) {
          const int span = 8 >> in[1], j = (i / span) * span + span - 1;
          const int r = in[13 + (bg ? 8 + j % 4 : j)] | (in[25] << 8);
          out[4 + bg * 8 + i] = (r * span + i % span) % b.n_chr;
        }
      out[20] = in[5]; out[25] = in[8];
      out[26] = in[2] == 2 && in[3] == 1;
      out[27] = in[4]; out[28] = in[6]; out[29] = in[7]; out[30] = in[25];
      out[31] = (in[26] * in[27]) & 255;
      out[32] = (in[26] * in[27]) >> 8;
    } else {
      out[0] = in[0] % b.n_prg;
      for (size_t i = 1; i < b.prg_count; ++i)
        out[i] = b.n_prg - (b.prg_count - i);
      for (int i = 0; i < 4; ++i)
        out[b.prg_count + i] = in[1 + i] % b.n_chr;
      out[b.prg_count + 4] = in[5];
    }
    return;
  }
  const int port = in[n + 1], value = in[n + 2], original = in[n + 3];
  std::copy_n(in, n, out);
  int dirty = 0;
  if (model == 0) {
    std::fill_n(out, n, 0);
    if (b.chip == BankControl::Mmc1) { out[0] = 12 + value; out[4] = 16; }
    else if (b.chip == BankControl::Mmc3) {
      const int initial[]{0, 0, 0, 0, 2, 4, 5, 6, 7, 0, 1, value};
      std::copy_n(initial, n, out);
    } else if (b.chip == BankControl::Mmc5) {
      out[0] = out[1] = 3;
      out[9] = out[10] = out[11] = 128; out[12] = 255;
      for (int i = 0; i < 12; ++i) out[13 + i] = i & 7;
      for (int i = 8; i < 12; ++i) out[13 + i] = i - 8;
      out[26] = out[27] = 255;
    } else out[5] = value;
    dirty = b.chip == BankControl::Mmc5 ? 15 : 7;
  } else if (b.chip == BankControl::Mmc1) {
    auto serial = [&](int data) {
      if (data & 0x80) { out[4] = 16; out[0] |= 12; dirty |= BankControl::Prg; return; }
      const bool complete = out[4] & 1;
      out[4] = (out[4] >> 1) | ((data & 1) << 4);
      if (!complete) return;
      const int reg = (port >> 13) & 3;
      const int dest[]{0, 2, 3, 1};
      out[dest[reg]] = out[4];
      out[4] = 16;
      dirty |= reg == 0 ? 7 : reg == 3 ? BankControl::Prg : BankControl::Chr;
    };
    serial(model == 2 ? original : value);
    if (model == 2 && (value & 0x80)) serial(value);
  } else if (b.chip == BankControl::Mmc3) {
    if (port == 0x8000) {
      const int p = !!(value & 0x40), c = !!(value & 0x80);
      dirty = (p != in[1] ? BankControl::Prg : 0) | (c != in[2] ? BankControl::Chr : 0);
      out[0] = value & 7; out[1] = p; out[2] = c;
    } else if (port == 0x8001) {
      out[3 + in[0]] = value;
      dirty = in[0] >= 6 ? BankControl::Prg : BankControl::Chr;
    } else if (port == 0xA000) { out[11] = 2 + (value & 1); dirty = BankControl::Mirror; }
  } else if (b.chip == BankControl::Mmc5) {
    if (port >= 0x5100 && port <= 0x5107) {
      const int j = port - 0x5100;
      out[j] = value & ((j == 5 || j == 6) ? 255 : 3);
      dirty = j == 1 ? 2 : j >= 4 ? 4 : 1;
    } else if (port == 0x5113) { out[8] = value & 7; dirty = 1; }
    else if (port >= 0x5114 && port <= 0x5117) { out[9 + port - 0x5114] = value; dirty = 1; }
    else if (port >= 0x5120 && port <= 0x512B) { out[13 + port - 0x5120] = value; dirty = 2; }
    else if (port == 0x5130) { out[25] = value & 3; dirty = 6; }
    else { out[26 + port - 0x5205] = value; dirty = 8; }
  } else {
    if (port == 0xA000) { out[0] = value & 15; dirty = BankControl::Prg; }
    else if (port <= 0xE000) { out[1 + ((port - 0xB000) >> 12)] = value & 31; dirty = BankControl::Chr; }
    else { out[5] = 2 + (value & 1); dirty = BankControl::Mirror; }
  }
  out[n] = dirty;
}

// Independent address/port oracle, excluded from the production extension.
static void reference_mapper_access(void* context, int model, const int* in, int* out) {
  const auto& bank = *static_cast<BankControl*>(context);
  const int address = in[0] | (in[1] << 8), writing = in[2];
  int packet[9]{-1, 1, -1, 0, 0, 0, 0, 0, 0};
  int port = address;
  if (bank.chip == BankControl::Mmc1) {
    port &= 0xE000;
    packet[1] = in[3] >= 0 ? 2 : 1;
  } else if (bank.chip == BankControl::Mmc3) port &= 0xE001;
  else if (bank.chip != BankControl::Mmc5) port &= 0xF000;
  if (writing)
    for (size_t i = 0; i < bank.engine.ports.size(); ++i)
      if (bank.engine.ports[i].address == port) packet[0] = static_cast<int>(i);
  if (address >= 0x8000) {
    const int slot = (address - 0x8000) / bank.unit;
    packet[4] = BankControl::Rom;
    packet[5] = in[4 + slot] * bank.unit + (address & (bank.unit - 1));
    packet[8] = writing;
    if (bank.chip == BankControl::Mmc5 && in[8 + slot]) {
      packet[4] = BankControl::Ram;
      packet[7] = writing && in[13] ? BankControl::Ram : 0;
    }
  }
  if (bank.chip == BankControl::Mmc3 && writing) {
    if (port == 0xC000) packet[2] = MapperIrq::Target;
    if (port == 0xC001) packet[2] = MapperIrq::Reload;
    if (port == 0xE000) packet[2] = MapperIrq::Disable;
    if (port == 0xE001) packet[2] = MapperIrq::Enable;
    packet[3] = port >= 0xC000 && port != 0xE000;
  }
  if (bank.chip == BankControl::Mmc5) {
    if (address >= 0x6000 && address < 0x8000) {
      packet[4] = BankControl::Ram;
      packet[5] = in[12] * BANK8 + (address & 0x1FFF);
      packet[7] = writing && in[13] ? BankControl::Ram : 0;
      packet[8] = writing && !in[13];
    } else if (address >= 0x5C00 && address < 0x6000) {
      packet[4] = BankControl::Exram;
      packet[5] = address - 0x5C00;
      packet[7] = writing ? BankControl::Exram : 0;
    } else if (address == 0x5204 && !writing) {
      packet[2] = MapperIrq::Ack; packet[4] = BankControl::Irq;
    } else if (address == 0x5205 || address == 0x5206) {
      packet[6] = in[14 + address - 0x5205];
    }
    if (writing && (address == 0x5203 || address == 0x5204)) {
      packet[2] = address == 0x5203 ? MapperIrq::Target : MapperIrq::Enable;
      packet[3] = 1;
    }
  }
  if (model == 3) std::copy_n(packet + 4, 3, out);
  else std::copy_n(packet, 9, out);
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
  if (auto* bank = mapper.bank_state()) {
    bank->state_count = bank->chip == BankControl::Mmc1 ? 5 : bank->chip == BankControl::Mmc3 ? 12 : bank->chip == BankControl::Mmc5 ? 28 : 6;
    bank->engine.ctx = bank;
    bank->engine.step = reference_mapper_banks;
    bank->engine.reset = 0;
    bank->engine.layout = 3;
    bank->engine.access_ctx = bank;
    bank->engine.access_step = reference_mapper_access;
    bank->engine.address_model = 3;
    if (bank->chip == BankControl::Mmc1)
      for (int p : {0x8000, 0xA000, 0xC000, 0xE000}) bank->engine.ports.push_back({p, 1, 2});
    else if (bank->chip == BankControl::Mmc3)
      for (int p : {0x8000, 0x8001, 0xA000}) bank->engine.ports.push_back({p, 1, 2});
    else if (bank->chip == BankControl::Mmc5) {
      for (int p : {0x5100,0x5101,0x5102,0x5103,0x5104,0x5105,0x5106,0x5107,
                    0x5113,0x5114,0x5115,0x5116,0x5117,0x5120,0x5121,0x5122,0x5123,
                    0x5124,0x5125,0x5126,0x5127,0x5128,0x5129,0x512A,0x512B,0x5130,0x5205,0x5206})
        bank->engine.ports.push_back({p, 1, 2});
    } else
      for (int p : {0xA000, 0xB000, 0xC000, 0xD000, 0xE000, 0xF000}) bank->engine.ports.push_back({p, 1, 2});
    bank->reset();
  }
  auto image = mapper.cpu_image();
  std::copy(image.begin(), image.end(), memory.begin());
  irq.machine = &machine;
  if (auto* state = mapper.irq_state()) {
    state->engine.step = reference_mapper_irq;
    for (int i = 0; i < MapperIrq::event_count; ++i)
      state->engine.models[static_cast<size_t>(i)] =
          static_cast<int>(state->chip) * MapperIrq::event_count + i;
  }
  mapper.connect({&ppu, &machine, &irq});
}

static void test_mappers() {
  array<uint8_t, 0x10000> memory{};
  ImageMachine machine(memory);
  Ppu ppu(vector<uint8_t>(0x2000), true, false);
  IrqLine irq;

  Cartridge c1{banks(16, BANK16), {}, Mirroring::Horizontal, 1};
  BankMapper mmc1(c1, BankControl::Mmc1);
  wire(mmc1, memory, machine, ppu, irq);
  for (int bit = 0; bit < 5; ++bit)
    mmc1.write(0xE000, (5 >> bit) & 1);
  assert(std::ranges::equal(mmc1.prg_banks(), array<int, 2>{5, 15}));
  assert(memory[0x8000] == 5 && memory[0xE000] == 15);

  BankMapper mmc1_rmw(c1, BankControl::Mmc1);
  wire(mmc1_rmw, memory, machine, ppu, irq);
  // RMW clocks the serial port with the mapped ROM byte (bank 15, bit
  // 1), not the CPU's final value (bit 0).
  for (int bit = 0; bit < 5; ++bit)
    mmc1_rmw.write_rmw(0xE000, 0, 1);
  assert(std::ranges::equal(mmc1_rmw.prg_banks(), array<int, 2>{15, 15}));
  // Consume the supplied operand even when it differs from mapped ROM.
  // The result's low bit must not replace the first RMW write.
  mmc1_rmw.write(0xE000, 0x80);
  for (int bit = 0; bit < 5; ++bit)
    mmc1_rmw.write_rmw(0xE000, 1, 0);
  assert(std::ranges::equal(mmc1_rmw.prg_banks(), array<int, 2>{0, 15}));

  Cartridge c3{banks(32, BANK8), vector<uint8_t>(0x20000), Mirroring::Horizontal, 4};
  BankMapper mmc3(c3, BankControl::Mmc3);
  wire(mmc3, memory, machine, ppu, irq);
  mmc3.write(0x8000, 6);
  mmc3.write(0x8001, 12);
  assert(std::ranges::equal(mmc3.prg_banks(), array<int, 4>{12, 1, 30, 31}));
  assert(memory[0x8000] == 12 && memory[0x8001] == 12 && memory[0xE000] == 31);
  mmc3.write(0xA000, 0);
  assert(ppu.mirroring == 2);
  mmc3.write(0xA000, 1);
  assert(ppu.mirroring == 3);

  Cartridge c3_ram{banks(32, BANK8), {}, Mirroring::Horizontal, 4};
  BankMapper mmc3_ram(c3_ram, BankControl::Mmc3);
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
  assert(memory[0x8000] == 24 && memory[0xA000] == 25 &&
         memory[0xC000] == 30 && memory[0xE000] == 31);
  mmc5.write(0x5205, 13);
  mmc5.write(0x5206, 17);
  assert(mmc5.read(0x5205) == 221 && mmc5.read(0x5206) == 0);
}

class FakeRuntime final : public SteppedRuntime {
public:
  bool halted() const override { return false; }
  void set_nmi() override {}
  void set_irq() override {}
  void clear_irq() override {}
  void poke(int addr, int value) override { memory[addr] = static_cast<uint8_t>(value); }
  Next peek() const override { return next; }
  vector<CpuWrite> step() override {
    return std::exchange(next_writes, {});
  }
  Next next;
  vector<CpuWrite> next_writes;
  array<uint8_t, 0x10000> memory{};
};

class CounterDevice final : public Device {
public:
  int read(int, int64_t) override { return ++value; }
  void write(int, int next, int64_t) override { value = next; }
  void write_rmw(int, int next, int original, int64_t cycle) override {
    rmw_cycle = cycle;
    rmw_original = original;
    value = next;
  }
  int value = 0;
  int64_t rmw_cycle = -1;
  int rmw_original = -1;
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
  rmw_runtime.next_writes.push_back({0xD000, 0x42, 0x41});
  rmw_bus.attach(rmw_device, 0xD000, 0xD000, true);
  rmw_bus.step();
  assert(rmw_device.rmw_cycle == 6 && rmw_device.value == 0x42 && rmw_device.rmw_original == 0x41);

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
  controller.engine.step = +[](void*, int event, const int* in, int* out) {
    int strobe = in[0], index = in[1];
    out[2] = 64 | (strobe ? in[2] : index == 8 ? 1 : in[2 + index]);
    out[3] = strobe ? 0 : 8 - index;
    for (int i = 0; i < 8; ++i) out[4 + i] = 64 | (index + i >= 8 ? 1 : in[2 + index + i]);
    if (event == Controller::Write) {
      const int next = in[11] & 1;
      if (strobe || next) index = 0;
      strobe = next;
    } else if (!strobe && event == Controller::Read) index = std::min(8, index + 1);
    else if (!strobe && event == Controller::Consume) index = std::min(8, index + std::max(0, in[12]));
    out[0] = strobe; out[1] = index;
  };
  for (int i = 0; i < Controller::EventCount; ++i) controller.engine.models[i] = i;
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
  // View descriptors materialize fill attributes and select the ExRAM
  // source without duplicating the chip's mode rules in production.
  Ppu viewed(vector<uint8_t>(0x2000), true, false);
  auto read_view = [&viewed](int address) {
    viewed.write(0x2006, address >> 8, 0);
    viewed.write(0x2006, address & 255, 0);
    (void)viewed.read(0x2007, 0);
    return viewed.read(0x2007, 0);
  };
  for (int mode=0;mode<4;++mode) {
    viewed.set_mmc5_state(mode, 0xFF, 0x27, 2, 0);
    assert(read_view(0x2000) == 0x27 && read_view(0x23C0) == 0xAA);
    viewed.set_mmc5_exram(1, 0xA5);
    viewed.set_mmc5_state(mode, 0xAA, 0, 0, 0);
    assert(read_view(0x2001) == (mode<=1 ? 0xA5 : 0));
  }
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

// Independent register/DMC oracles. Production has no register decoder,
// period tables, sample restart, bit-shift or sample-reader state machine.
static void reference_audio_program(const int* in, int* out) {
  std::copy(in, in + 53, out);
  const int port = in[53], value = in[54];
  out[53] = port == 21 ? (in[1] > 0) + 2 * (in[18] > 0) + 4 * (in[35] > 0) +
      8 * (in[42] > 0) + 16 * (in[56] > 0) + 64 * in[58] + 128 * in[57] : 64;
  out[54] = in[55] && port == 23;
  out[55] = !in[55] && port == 21;
  out[56] = in[55] && ((port >= 16 && port <= 19) || port == 21);
  if (!in[55])
    return;
  constexpr int lengths[]{10,254,20,2,40,4,80,6,160,8,60,10,14,12,26,14,
                          12,16,24,18,48,20,96,22,192,24,72,26,16,28,32,30};
  constexpr int noise_period[]{4,8,16,32,64,96,128,160,202,254,380,508,762,1016,2034,4068};
  if (port >= 0 && port < 8) {
    int* p = out + (port / 4) * 17;
    switch (port & 3) {
    case 0: p[10]=value>>6;p[12]=!!(value&32);p[13]=!!(value&16);p[14]=value&15;break;
    case 1: p[3]=!!(value&128);p[4]=(value>>4)&7;p[5]=!!(value&8);p[6]=value&7;p[7]=1;break;
    case 2: p[0]=(p[0]&0x700)|value;break;
    case 3: p[0]=(p[0]&255)|((value&7)<<8);if(p[2])p[1]=lengths[value>>3];p[9]=0;p[11]=1;break;
    }
  } else switch (port) {
  case 8: out[36]=!!(value&128);out[38]=value&127;break;
  case 10:out[34]=(out[34]&0x700)|value;break;
  case 11:out[34]=(out[34]&255)|((value&7)<<8);if(out[37])out[35]=lengths[value>>3];out[39]=1;break;
  case 12:out[46]=!!(value&32);out[47]=!!(value&16);out[48]=value&15;break;
  case 14:out[51]=!!(value&128);out[52]=noise_period[value&15];break;
  case 15:if(out[43])out[42]=lengths[value>>3];out[45]=1;break;
  case 21:
    for(int i=0;i<2;++i){int* p=out+17*i;p[2]=!!(value&(1<<i));if(!p[2])p[1]=0;}
    out[37]=!!(value&4);out[43]=!!(value&8);
    if(!out[37])out[35]=0;
    if(!out[43])out[42]=0;
    break;
  }
}

static void reference_audio_dmc(const int* in, int* out) {
  std::copy(in, in + 14, out);
  const int event=in[14],port=in[15],value=in[16];
  int stall=0;
  if(event==0) switch(port) {
  case 16: {
    constexpr int periods[]{428,380,340,320,286,254,226,214,190,160,142,128,106,85,72,54};
    out[0]=!!(value&128);out[1]=!!(value&64);out[3]=periods[value&15];
    if(!out[0])out[2]=0;
    break;
  }
  case 17:out[4]=value&127;break;
  case 18:out[5]=0xC000|(value<<6);break;
  case 19:out[6]=(value<<4)|1;break;
  case 21:
    if(!(value&16))out[8]=0;
    else if(!out[8]){out[7]=out[5];out[8]=out[6];}
    out[2]=0;break;
  }
  if(event==1 && out[10] && out[8]>0) {
    out[9]=value;out[10]=0;stall=4;
    out[7]=out[7]==0xFFFF?0x8000:out[7]+1;
    if(--out[8]==0) {
      if(out[1]){out[7]=out[5];out[8]=out[6];}
      else if(out[0])out[2]=1;
    }
  }
  if(event==2) {
    if(!out[13]) {
      if(out[11]&1){if(out[4]<=125)out[4]+=2;}
      else if(out[4]>=2)out[4]-=2;
    }
    out[11]>>=1;
    if(--out[12]==0) {
      out[12]=8;
      if(out[10])out[13]=1;
      else{out[13]=0;out[11]=out[9];out[10]=1;}
    }
  }
  out[14]=out[8]>0&&out[10];out[15]=out[8]>0;
  out[16]=2+(out[12]-1)*out[3];
  out[17]=event==2||event==3;out[18]=out[3]-(event==2);out[19]=stall;
}

// Literal clock oracle, excluded from production. Deliberately steps each
// cycle rather than using the constructed quotient and GF(2) powers.
static void reference_audio_oscillators(const int* in, int* out) {
  int tick=in[1], pc[2]{in[3],in[6]}, pp[2]{in[4],in[7]};
  int tc=in[9], tp=in[10], nc=in[14], steps=0;
  for(int cycle=0;cycle<in[0];++cycle) {
    if(tc--==0){tc=in[8];if(in[11]>0 && in[12]>0)tp=(tp+1)&31;}
    tick=!tick;
    if(tick) {
      for(int i=0;i<2;++i)if(pc[i]--==0){pc[i]=in[2+3*i];pp[i]=(pp[i]+1)&7;}
      if(nc--==0) {
        nc=in[13];
        ++steps;
      }
    }
  }
  out[0]=tick;out[1]=pc[0];out[2]=pp[0];out[3]=pc[1];out[4]=pp[1];
  out[5]=tc;out[6]=tp;out[7]=nc;
  out[8]=steps;
}

static void reference_audio_noise(const int* in, int* out) {
  int noise=0;
  for(int i=0;i<15;++i)noise|=in[2+i]<<i;
  for(int i=0;i<in[0];++i) {
    const int feedback=(noise&1)^((noise>>(in[1]?6:1))&1);
    noise=(noise>>1)|(feedback<<14);
  }
  for(int i=0;i<15;++i)out[i]=(noise>>i)&1;
}

// Independent conventional APU modulation/frame/DAC oracle. These rules
// are absent from the production host, which attaches nes/audio.py.
static void reference_audio(void*, int model, const int* in, int* out) {
  if (model == 15) {
    reference_audio_noise(in, out);
    return;
  }
  if (model >= 13) {
    reference_audio_oscillators(in, out);
    return;
  }
  if (model == 8) {
    reference_audio_program(in, out);
    return;
  }
  if (model >= 9) {
    reference_audio_dmc(in, out);
    return;
  }
  if (model >= 4) {
    const int event = model - 4;
    int five = in[0], inhibit = in[1], irq = in[2], step = in[3];
    int reset = 0, quarter = 0, half = 0;
    if (event == 0) {
      quarter = 1;
      half = step == 1 || step == 3;
      if (step == 3) {
        if (!five && !inhibit)
          irq = 1;
        step = 0;
        reset = 1;
      } else {
        ++step;
      }
    } else if (event == 1) {
      five = !!(in[5] & 128);
      inhibit = !!(in[5] & 64);
      if (inhibit)
        irq = 0;
      step = 0;
      reset = 1;
      quarter = half = five;
    } else if (event == 2) {
      irq = 0;
    }
    const int deadlines[]{7457, 14913, 22371, five ? 37281 : 29829};
    const int result[]{five, inhibit, irq, step, reset, quarter, half, deadlines[step]};
    std::copy(std::begin(result), std::end(result), out);
    return;
  }
  auto envelope = [](const int* e, bool clock, int* result) {
    int start = e[0], divider = e[4], decay = e[5];
    if (clock) {
      if (start) {
        start = 0;
        divider = e[3];
        decay = 15;
      } else if (divider-- == 0) {
        divider = e[3];
        if (decay > 0)
          --decay;
        else if (e[1])
          decay = 15;
      }
    }
    result[0] = start; result[1] = divider; result[2] = decay;
  };
  const bool quarter = model == 0 || model == 2, half = model == 1 || model == 2;
  constexpr int duty[4][8]{{0,1,0,0,0,0,0,0}, {0,1,1,0,0,0,0,0},
                          {0,1,1,1,1,0,0,0}, {1,0,0,1,1,1,1,1}};
  for (int i = 0; i < 2; ++i) {
    const int* p = in + 1 + i * 17;
    int timer = p[0], divider = p[8];
    const int change = timer >> p[6];
    const int target = p[5] ? timer - change - (i == 0) : timer + change;
    const bool muted = timer < 8 || target > 2047;
    if (model == 3) {
      out[i] = p[2] && p[1] && !muted && duty[p[10]][p[9]] ? (p[13] ? p[14] : p[16]) : 0;
      continue;
    }
    int* result = out + i * 7;
    envelope(p + 11, quarter, result);
    result[3] = p[1] - (half && p[1] > 0 && !p[12]);
    if (half && divider == 0 && p[3] && p[6] > 0 && !muted && target >= 0 && target <= 2047)
      timer = target;
    if (half)
      divider = divider == 0 || p[7] ? p[4] : divider - 1;
    result[4] = timer;
    result[5] = half ? 0 : p[7];
    result[6] = divider;
  }
  const int* t = in + 35;
  const int* n = in + 43;
  if (model == 3) {
    const int pos = t[7];
    out[2] = t[3] && t[1] && t[6] && t[0] >= 2 ? (pos < 16 ? 15 - pos : pos - 16) : 0;
    out[3] = n[1] && n[0] && !(n[2] & 1) ? (n[5] ? n[6] : n[8]) : 0;
    return;
  }
  out[14] = t[1] - (half && t[1] > 0 && !t[2]);
  out[15] = quarter && !t[2] ? 0 : t[5];
  out[16] = quarter ? (t[5] ? t[4] : std::max(0, t[6] - 1)) : t[6];
  envelope(n + 3, quarter, out + 17);
  out[20] = n[0] - (half && n[0] > 0 && !n[4]);
}

static void reference_sample(void* context, int, const float* in, float* out) {
  int channels[52], levels[4];
  std::copy_n(in, 52, channels);
  reference_audio(context, 3, channels, levels);
  const int p = levels[0] + levels[1];
  const double pulse = p ? 95.88 / (8128.0 / p + 100.0) : 0.0;
  const double tnd = levels[2] / 8227.0 + levels[3] / 12241.0 + in[52] / 22638.0;
  out[0] = pulse + (tnd > 0.0 ? 159.79 / (1.0 / tnd + 100.0) : 0.0);
  out[1] = out[0] - in[53] + 0.9995 * in[54];
  out[2] = std::clamp(2.0 * out[1], -1.0, 1.0);
}

static void attach_audio(Apu& apu) {
  Apu::Engine engine;
  engine.ctx = &apu; engine.step = reference_audio;
  engine.control = {0, 1, 2}; engine.frame = {4, 5, 6, 7};
  engine.sample_ctx = &apu; engine.sample_step = reference_sample;
  engine.program = 8;
  engine.program_writes.fill(8);
  engine.dmc = {9, 10, 11, 12};
  engine.dmc_writes.fill(9);
  engine.oscillators = {13, 14};
  engine.noise = {{{15, 15}, {15, 15}}};
  apu.set_engine(engine);
}

static void test_apu() {
  Apu apu(44'100.0);
  attach_audio(apu);
  assert(apu.write(0x4015, 0));
  assert(!apu.write(0x4000, 0));
  apu.advance_to(29'781);
  vector<float> audio;
  apu.take_samples(audio);
  assert(audio.size() >= 733 && audio.size() <= 734);
  assert(apu.take_dma_stall() == 0);

  Apu chunked(44'100.0);
  attach_audio(chunked);
  chunked.advance_to(10'000);
  vector<float> piece;
  chunked.take_samples(piece);
  const auto first = piece;
  chunked.advance_to(29'781);
  chunked.take_samples(piece);
  const auto& second = piece;
  assert(first.size() + second.size() >= 733 && first.size() + second.size() <= 734);
  assert(std::equal(first.begin(), first.end(), audio.begin()));
  assert(std::equal(second.begin(), second.end(), audio.begin() + first.size()));
  chunked.take_samples(piece);
  assert(piece.empty()); // draining again must not return the recycled frame


  // A 65-byte DMC sample crosses $FFFF. DMA effects consume exactly
  // four cycles per byte and only the reader wraps to $8000.
  Apu streamed;
  attach_audio(streamed);
  vector<int> addresses;
  streamed.memory = [&addresses](int address) {
    addresses.push_back(address);
    return address & 255;
  };
  streamed.write(0x4010, 0x8F);
  streamed.write(0x4012, 0xFF);
  streamed.write(0x4013, 4);
  streamed.write(0x4015, 16);
  streamed.advance_to(29'000);
  assert(addresses.size() == 65 && streamed.take_dma_stall() == 4 * 65);
  for (size_t i = 0; i < addresses.size(); ++i)
    assert(addresses[i] == (i < 64 ? 0xFFC0 + int(i) : 0x8000));
  assert(streamed.read(0x4015) == 0x80);
  streamed.write(0x4015, 0);
  assert(!streamed.irq_pending());

  // Batch size and sample rate do not change oscillator state or samples.
  for (double rate : {1.0, 8000.0, 44100.0, 48000.0, 96000.0, 2000000.0}) {
    Apu whole(rate), singles(rate);
    for (Apu* a : {&whole, &singles}) {
      attach_audio(*a);
      a->write(0x4015, 15);
      for (auto [port, value] : {std::pair{0x4000, 0xBF}, {0x4002, 8}, {0x4003, 0x18},
                                 {0x4004, 0x5C}, {0x4006, 31}, {0x4007, 0x18},
                                 {0x4008, 0xFF}, {0x400A, 9}, {0x400B, 0x18},
                                 {0x400C, 0x1A}, {0x400E, 0x80}, {0x400F, 0x18},
                                 {0x4017, 0x80}})
        a->write(port, value);
    }
    whole.advance_to(100000);
    for (int cycle=1;cycle<=100000;++cycle)
      singles.advance_to(cycle);
    vector<float> a, b;
    whole.take_samples(a);
    singles.take_samples(b);
    assert(a == b);
    assert(whole.sample_accum == singles.sample_accum);
    assert(whole.apu_tick == singles.apu_tick);
    for (int i=0;i<2;++i) {
      assert(whole.pulse[i].timer_counter == singles.pulse[i].timer_counter);
      assert(whole.pulse[i].seq_pos == singles.pulse[i].seq_pos);
    }
    assert(whole.triangle.timer_counter == singles.triangle.timer_counter);
    assert(whole.triangle.seq_pos == singles.triangle.seq_pos);
    assert(whole.noise.timer_counter == singles.noise.timer_counter);
    assert(whole.noise.lfsr == singles.noise.lfsr);
  }
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
