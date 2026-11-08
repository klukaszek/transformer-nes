#include "nes/bus.h"

#include <algorithm>
#include <stdexcept>

namespace transformer_nes::nes {

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

Bus::Bus(SteppedRuntime& runtime, span<const uint8_t> image) : runtime_(runtime) {
  if (image.size() != ram_.size())
    throw std::invalid_argument("memory image must be 64 KiB");
  std::copy(image.begin(), image.end(), ram_.begin());
}
void Bus::attach(Device& device, int lo, int hi, bool writes_only) {
  devices_.attach(device, lo, hi, writes_only);
}
void Bus::step() {
  const auto next = runtime_.peek();
  cycles_ += next.cycles;
  if (next.read >= 0)
    if (auto* d = devices_.find(next.read, true)) {
      const int value = d->read(next.read, stamp_base_ + cycles_) & 0xFF;
      runtime_.poke(next.read, value);
      ram_[cpu_ram_index(next.read)] = static_cast<uint8_t>(value);
      cycles_ += d->take_stall();
    }
  for (const auto& write : runtime_.step()) {
    ram_[cpu_ram_index(write.addr)] = static_cast<uint8_t>(write.value);
    if (auto* d = devices_.find(write.addr)) {
      if (write.original >= 0)
        d->write_rmw(write.addr, write.value, write.original, stamp_base_ + cycles_);
      else
        d->write(write.addr, write.value, stamp_base_ + cycles_);
      cycles_ += d->take_stall();
    }
  }
}
MachineRun Bus::run_cycles(int64_t budget, function<bool()> pause, int64_t cycle_base) {
  const int64_t start = cycles_;
  stamp_base_ = cycle_base - start;
  while (!halted() && cycles_ - start < budget) {
    step();
    if (pause && pause())
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
array<int, 12> Controller::apply(Event event, int value, int count) {
  if (!engine.step)
    throw std::runtime_error("controller transformer is required");
  array<int, 13> in{};
  std::copy(state_.begin(), state_.end(), in.begin());
  std::copy(buttons_.begin(), buttons_.end(), in.begin() + 2);
  in[10] = event;
  in[11] = value;
  in[12] = count;
  array<int, 12> out;
  engine.step(engine.ctx, engine.model, in.data(), out.data());
  std::copy_n(out.begin(), state_.size(), state_.begin());
  return out;
}
int Controller::read(int addr, int64_t) {
  return addr == 0x4016 ? apply(Read)[2] : 0x40;
}
void Controller::write(int addr, int value, int64_t) {
  if (addr == 0x4016)
    apply(Write, value & 0xFF);
}
ReadLease Controller::read_lease(int addr) {
  if (addr != 0x4016)
    return {};
  const auto out = apply(Lease);
  if (out[3] < 0 || out[3] > 8)
    throw std::runtime_error("invalid controller reply packet length");
  if (out[3] == 0)
    return out[2];
  ReadReply reply;
  reply.size = out[3];
  std::copy_n(out.begin() + 4, reply.size, reply.bytes.begin());
  return reply;
}
void Controller::lease_consumed(int addr, int count) {
  if (addr == 0x4016)
    apply(Consume, 0, count);
}
} // namespace transformer_nes::nes
