#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace transformer_nes::nes {

using std::function;
using std::unique_ptr;
using std::vector;

class Apu final {
public:
  using MemoryReader = function<int(int)>;

  explicit Apu(double sample_rate = 44'100.0);
  ~Apu();

  Apu(const Apu&) = delete;
  Apu& operator=(const Apu&) = delete;
  Apu(Apu&&) noexcept;
  Apu& operator=(Apu&&) noexcept;

  [[nodiscard]] int read(int address);
  void write(int address, int value);
  [[nodiscard]] vector<float> end_frame(int cpu_cycles, MemoryReader reader = {});
  [[nodiscard]] int take_dma_stall();
  [[nodiscard]] bool irq_pending() const;

private:
  struct Impl;
  unique_ptr<Impl> impl_;
};

} // namespace transformer_nes::nes
