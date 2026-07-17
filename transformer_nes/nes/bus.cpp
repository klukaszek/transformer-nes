#include "nes/bus.h"

#include <algorithm>
#include <stdexcept>

namespace transformer_nes::nes {

namespace {

constexpr bool is_memory_rmw(uint8_t opcode) {
  const bool alu_rmw_family = (opcode & 0x80) == 0 || (opcode & 0x40) != 0;
  return alu_rmw_family && (opcode & 3) >= 2;
}

} // namespace

void DeviceMap::attach(Device& device, int first, int last, bool writes_only) {
  if (first < 0 || last > 0xFFFF || first > last)
    throw std::invalid_argument("invalid device address range");
  mappings_.push_back({first, last, &device, writes_only});
}

Device* DeviceMap::find(int address, bool for_read) const {
  for (const auto& mapping : mappings_)
    if (address >= mapping.first && address <= mapping.last && !(for_read && mapping.writes_only))
      return mapping.device;
  return nullptr;
}

Bus::Bus(SteppedRuntime& runtime, span<const uint8_t> image, const OpcodeTable& opcodes)
    : runtime_(runtime), opcodes_(opcodes) {
  if (image.size() != ram_.size())
    throw std::invalid_argument("memory image must be 64 KiB");
  std::copy(image.begin(), image.end(), ram_.begin());
}
void Bus::attach(Device& device, int lo, int hi, bool writes_only) {
  devices_.attach(device, lo, hi, writes_only);
}
optional<int> Bus::read_target(int pc, int x, int y) const {
  const auto pc_index = address_index(pc);
  int mode = opcodes_[ram_[pc_index]].read_mode;
  if (mode < 0)
    return std::nullopt;
  int lo = ram_[address_index(pc + 1)];
  if (mode == 0)
    return lo;
  if (mode == 1)
    return (lo + x) & 0xFF;
  if (mode == 2)
    return (lo + y) & 0xFF;
  int hi = ram_[address_index(pc + 2)];
  if (mode == 3)
    return lo | (hi << 8);
  if (mode == 4)
    return ((lo | (hi << 8)) + x) & 0xFFFF;
  if (mode == 5)
    return ((lo | (hi << 8)) + y) & 0xFFFF;
  if (mode == 6) {
    int zp = (lo + x) & 0xFF;
    return ram_[address_index(zp)] | (ram_[address_index((zp + 1) & 0xFF)] << 8);
  }
  if (mode == 7) {
    int base = ram_[address_index(lo)] | (ram_[address_index((lo + 1) & 0xFF)] << 8);
    return (base + y) & 0xFFFF;
  }
  return std::nullopt;
}
CpuStep Bus::step() {
  const uint8_t opcode = ram_[address_index(runtime_.pc())];
  if (runtime_.nmi_pending() || runtime_.irq_ready())
    cycles_ += 7;
  else {
    int pc = runtime_.pc();
    cycles_ += opcodes_[ram_[address_index(pc)]].cycles;
    auto target = read_target(pc, runtime_.x(), runtime_.y());
    if (target)
      if (auto* d = devices_.find(*target, true)) {
        int value = d->read(*target) & 0xFF;
        runtime_.poke(*target, value);
        ram_[address_index(*target)] = static_cast<uint8_t>(value);
        cycles_ += d->take_stall();
      }
  }
  CpuStep row = runtime_.step();
  for (const auto& write : row.writes) {
    const int original = ram_[address_index(write.addr)];
    ram_[address_index(write.addr)] = static_cast<uint8_t>(write.value);
    if (auto* d = devices_.find(write.addr)) {
      if (is_memory_rmw(opcode))
        d->write_rmw(write.addr, original, write.value);
      else
        d->write_at(write.addr, write.value, stamp_base_ + cycles_);
      cycles_ += d->take_stall();
    }
  }
  return row;
}
vector<CpuStep> Bus::run(int max_steps, const vector<int>& nmi_at, const vector<int>& irq_at) {
  vector<CpuStep> out;
  for (int i = 0; i < max_steps && !halted(); ++i) {
    if (std::find(nmi_at.begin(), nmi_at.end(), i) != nmi_at.end())
      set_nmi();
    if (std::find(irq_at.begin(), irq_at.end(), i) != irq_at.end())
      set_irq();
    out.push_back(step());
  }
  return out;
}
MachineRun Bus::run_cycles(int64_t budget, function<bool()> pause, int64_t cycle_base) {
  if (!pause)
    pause = [] { return false; };
  const int64_t start = cycles_;
  stamp_base_ = cycle_base - start;
  while (!halted() && cycles_ - start < budget) {
    step();
    if (pause())
      break;
  }
  return {cycles_ - start, halted()};
}
void Bus::map_region(int base, span<const uint8_t> data) {
  for (size_t offset = 0; offset < data.size(); ++offset)
    poke(base + static_cast<int>(offset), data[offset]);
}
void Controller::set_button(size_t index, bool pressed) {
  if (index >= buttons_.size())
    throw std::out_of_range("controller button");
  buttons_[index] = pressed;
}
int Controller::read(int addr) {
  if (addr != 0x4016)
    return 0x40;
  if (strobe_)
    return 0x40 | buttons_[0];
  if (index_ >= 8)
    return 0x41;
  return 0x40 | buttons_[index_++];
}
void Controller::write(int addr, int value) {
  if (addr != 0x4016)
    return;
  bool next = (value & 1) != 0;
  if ((strobe_ && !next) || next)
    index_ = 0;
  strobe_ = next;
}
ReadLease Controller::read_lease(int addr) {
  if (addr != 0x4016)
    return {};
  if (strobe_)
    return 0x40 | buttons_[0];
  if (index_ >= 8)
    return 0x41;
  vector<uint8_t> out;
  for (size_t i = index_; i < buttons_.size(); ++i)
    out.push_back(static_cast<uint8_t>(0x40 | buttons_[i]));
  return out;
}
void Controller::lease_consumed(int addr, int count) {
  if (addr == 0x4016 && !strobe_ && count > 0)
    index_ = std::min(buttons_.size(), index_ + static_cast<size_t>(count));
}
void OamDma::write(int, int value) {
  array<uint8_t, 256> page{};
  int base = (value & 0xFF) << 8;
  for (size_t i = 0; i < page.size(); ++i)
    page[i] = static_cast<uint8_t>(read_cpu_(base + static_cast<int>(i)));
  write_oam_(page);
  stall_ += 513;
}

} // namespace transformer_nes::nes
