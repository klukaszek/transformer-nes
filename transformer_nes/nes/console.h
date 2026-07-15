#pragma once

#include "apu.h"
#include "bus.h"
#include "cartridge.h"
#include "mappers.h"
#include "ppu.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace transformer_nes::nes {

using std::int64_t;
using std::size_t;
using std::uint8_t;
using std::unique_ptr;
using std::vector;

struct JournalEntry {
  int frame = 0;
  int phase = 0;
  int kind = 0;
  int line = 0;
  int value = 0;
  int pc = -1;
  int opcode = -1;
};

class Console {
public:
  Console(unique_ptr<Mapper>, unique_ptr<Ppu>, unique_ptr<Apu>, ConsoleMachine&);
  ~Console();
  Ppu::Frame run_frame();
  Ppu::Frame run(int frames);
  void set_button(size_t index, bool pressed) { controller_.set_button(index, pressed); }
  int read_cpu(int addr) const { return machine_.read_ram(addr); }
  const vector<float>& last_audio() const { return last_audio_; }
  const vector<JournalEntry>& journal() const { return journal_; }
  const Ppu& ppu() const { return *ppu_; }
  Ppu& ppu() { return *ppu_; }
  const Mapper& mapper() const { return *mapper_; }
  Mapper& mapper() { return *mapper_; }
  long long cycles() const { return cycles_; }
  int frame_index() const { return frame_index_; }
  bool halted() const { return halted_; }
  int sprite0_scanline() const { return sprite0_scanline_; }

private:
  class PpuDevice;
  class ApuDevice;
  class MapperDevice;
  class FrameCounter;
  enum class IrqSource : uint8_t { mapper = 1, apu = 2 };

  void attach_devices();
  void run_until(int64_t);
  void set_irq(IrqSource source, bool active);
  void sync_irq();
  void check_interrupts();
  void add_event(int phase, int line, int value);
  unique_ptr<Mapper> mapper_;
  unique_ptr<Ppu> ppu_;
  unique_ptr<Apu> apu_;
  ConsoleMachine& machine_;
  Controller controller_;
  unique_ptr<OamDma> dma_;
  unique_ptr<PpuDevice> ppu_device_;
  unique_ptr<ApuDevice> apu_device_;
  unique_ptr<MapperDevice> mapper_device_;
  unique_ptr<FrameCounter> frame_counter_;
  int64_t cycles_ = 0, frame_base_ = 0;
  int frame_index_ = 0, sprite0_scanline_ = -1;
  bool halted_ = false;
  uint8_t irq_sources_ = 0;
  vector<float> last_audio_;
  vector<JournalEntry> journal_;
};

} // namespace transformer_nes::nes
