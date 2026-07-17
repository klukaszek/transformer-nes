#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nes/apu.h"
#include "nes/bus.h"
#include "nes/cartridge.h"
#include "nes/console.h"
#include "nes/mappers/mappers.h"
#include "nes/palette.h"
#include "nes/ppu.h"
#include "nes/tape.h"

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;
using namespace transformer_nes::nes;

namespace {

using std::function;
using std::int64_t;
using std::pair;
using std::size_t;
using std::span;
using std::string;
using std::uint8_t;
using std::unique_ptr;
using std::vector;

py::bytes to_bytes(span<const uint8_t> data) {
  return py::bytes(reinterpret_cast<const char*>(data.data()),
                   static_cast<py::ssize_t>(data.size()));
}

class PythonSteppedRuntime final : public SteppedRuntime {
public:
  explicit PythonSteppedRuntime(py::object core) : core_(std::move(core)) {}
  int pc() const override { return py::cast<int>(core_.attr("pc")); }
  int x() const override { return py::cast<int>(core_.attr("x")); }
  int y() const override { return py::cast<int>(core_.attr("y")); }
  bool halted() const override { return py::cast<bool>(core_.attr("halted")); }
  bool nmi_pending() const override { return py::cast<bool>(core_.attr("nmi_pending")); }
  bool irq_ready() const override { return py::cast<bool>(core_.attr("irq_ready")); }
  void set_nmi() override { core_.attr("set_nmi")(); }
  void set_irq() override { core_.attr("set_irq")(); }
  void clear_irq() override { core_.attr("clear_irq")(); }
  void poke(int a, int v) override { core_.attr("poke")(a, v); }
  CpuStep step() override {
    py::object row = core_.attr("step")();
    py::object state = row.attr("state");
    CpuStep out;
    out.pc = py::cast<int>(state.attr("pc"));
    out.a = py::cast<int>(state.attr("a"));
    out.x = py::cast<int>(state.attr("x"));
    out.y = py::cast<int>(state.attr("y"));
    out.s = py::cast<int>(state.attr("s"));
    out.p = py::cast<int>(state.attr("p"));
    out.halted = py::cast<bool>(state.attr("halted"));
    for (py::handle w : row.attr("writes"))
      out.writes.push_back({py::cast<int>(w.attr("addr")), py::cast<int>(w.attr("value"))});
    return out;
  }

private:
  py::object core_;
};

class PythonEngineMachine final : public ConsoleMachine {
public:
  PythonEngineMachine(py::object runtime, const vector<uint8_t>& image, int start)
      : rt_(std::move(runtime)) {
    rt_.attr("load_image")(to_bytes(image), py::arg("start_pc") = start);
  }
  void attach(Device& d, int lo, int hi, bool writes_only = false) override {
    devices_.attach(d, lo, hi, writes_only);
    if (!writes_only)
      rt_.attr("watch_read")(lo, hi);
    rt_.attr("watch_write")(lo, hi);
  }
  void set_nmi() override { rt_.attr("set_nmi")(); }
  void set_irq() override { rt_.attr("set_irq")(); }
  void clear_irq() override { rt_.attr("clear_irq")(); }
  void clear_read_lease() override {
    flush_queues();
    lease_owners_.clear();
    lease_queues_.clear();
    rt_.attr("clear_read_lease")();
  }
  int read_ram(int a) const override { return py::cast<int>(rt_.attr("read_ram")(a)); }
  void poke(int a, int v) override { rt_.attr("inject_write")(a, v); }
  void map_region(int a, span<const uint8_t> d) override { rt_.attr("map_region")(a, to_bytes(d)); }
  MachineRun run_cycles(int64_t budget, function<bool()> pause = {},
                        int64_t cycle_base = 0) override {
    int64_t cycles = 0;
    bool halted = false;
    while (cycles < budget && !halted) {
      py::dict out = rt_.attr("run")(INT64_MAX, py::arg("max_cycles") = budget - cycles,
                                     py::arg("trace") = false)
                         .cast<py::dict>();
      cycles += py::cast<int64_t>(out["cycles"]);
      // The native write journal / PRG banking are not wired to this
      // console; fail loudly rather than drop mapper side effects.
      if (py::len(out["wj_addr"]) != 0)
        throw std::runtime_error("engine returned journaled writes but no drain is wired");
      drain_leases(out);
      // A watched write pauses the engine, so the chunk's writes
      // executed at its end: stamp them with the cycle there.
      cycles += service_writes(out, cycle_base + cycles);
      string reason = py::cast<string>(out["reason"]);
      if (reason == "watch_read")
        cycles += service_read(py::cast<int>(out["read_addr"]));
      halted = reason == "halt" || reason == "trap";
      if (pause && pause())
        break;
    }
    return {cycles, halted};
  }

private:
  Device* device_at(int a) const { return devices_.find(a); }
  int service_writes(const py::dict& out, int64_t cycle) {
    int stalls = 0;
    py::sequence aa = out["hit_addr"], vv = out["hit_value"];
    if (py::len(aa) != py::len(vv))
      throw std::runtime_error("CPU engine returned mismatched write arrays");
    if (py::len(aa) == 0)
      return 0;
    // Watched writes pause the step that made them, so pause_ir is that
    // step's opcode: RMW writes route through write_rmw (MMC1's quirk).
    const bool rmw = is_memory_rmw(static_cast<uint8_t>(py::cast<int>(out["pause_ir"])));
    for (py::size_t i = 0; i < py::len(aa); ++i) {
      int a = py::cast<int>(aa[i]);
      if (auto* d = device_at(a)) {
        drop_device_leases(d);
        if (rmw)
          d->write_rmw(a, py::cast<int>(vv[i]), cycle);
        else
          d->write(a, py::cast<int>(vv[i]), cycle);
        stalls += d->take_stall();
      }
    }
    return stalls;
  }
  int service_read(int a) {
    flush_queues();
    auto* d = device_at(a);
    int value = d ? d->read(a) & 0xFF : read_ram(a);
    int stalls = d ? d->take_stall() : 0;
    if (d) {
      auto lease = d->read_lease(a);
      int evicted = -1;
      if (auto* v = std::get_if<int>(&lease)) {
        evicted = py::cast<int>(rt_.attr("set_read_lease")(a, *v, 0));
        lease_queues_.erase(a);
      } else if (auto* q = std::get_if<vector<uint8_t>>(&lease)) {
        evicted = py::cast<int>(rt_.attr("set_read_lease_seq")(a, to_bytes(*q), 0));
        lease_queues_.insert(a);
      }
      if (!std::holds_alternative<std::monostate>(lease))
        lease_owners_[a] = d;
      if (evicted >= 0 && evicted != a) {
        lease_owners_.erase(evicted);
        lease_queues_.erase(evicted);
      }
    }
    poke(a, value);
    return stalls;
  }
  void flush_queues() {
    for (int addr : lease_queues_) {
      int count = py::cast<int>(rt_.attr("take_lease_served")(addr));
      if (auto owner = lease_owners_.find(addr); count && owner != lease_owners_.end())
        owner->second->lease_consumed(addr, count);
    }
  }
  void drain_leases(const py::dict& out) {
    py::sequence addresses = out["ls_addr"], served = out["ls_served"];
    if (py::len(addresses) != py::len(served))
      throw std::runtime_error("CPU engine returned mismatched lease arrays");
    for (py::size_t i = 0; i < py::len(addresses); ++i) {
      int addr = py::cast<int>(addresses[i]);
      auto owner = lease_owners_.find(addr);
      if (lease_queues_.erase(addr) && owner != lease_owners_.end())
        owner->second->lease_consumed(addr, py::cast<int>(served[i]));
      lease_owners_.erase(addr);
    }
  }
  void drop_device_leases(Device* device) {
    vector<int> drop;
    for (auto [addr, owner] : lease_owners_)
      if (owner == device)
        drop.push_back(addr);
    for (int addr : drop) {
      if (lease_queues_.erase(addr)) {
        int count = py::cast<int>(rt_.attr("take_lease_served")(addr));
        if (count)
          device->lease_consumed(addr, count);
      }
      rt_.attr("clear_read_lease_addr")(addr);
      lease_owners_.erase(addr);
    }
  }
  py::object rt_;
  DeviceMap devices_;
  std::unordered_map<int, Device*> lease_owners_;
  std::unordered_set<int> lease_queues_;
};

class ConsoleHost {
public:
  ConsoleHost(Cartridge cart, py::object factory, const vector<pair<int, int>>& opcode_values,
              double sample_rate) {
    if (opcode_values.size() != OpcodeTable{}.size())
      throw std::invalid_argument("opcode table must contain exactly 256 entries");
    OpcodeTable opcodes{};
    for (size_t i = 0; i < opcode_values.size(); ++i)
      opcodes[i] = {opcode_values[i].first, opcode_values[i].second};
    Ppu ppu(cart.chr.empty() ? vector<uint8_t>(0x2000) : cart.chr, cart.chr.empty(),
            cart.mirroring == Mirroring::Vertical);
    auto mapper = make_mapper(std::move(cart));
    auto image = mapper->cpu_image();
    int start = image[0xFFFC] | (image[0xFFFD] << 8);
    py::tuple backend = factory(to_bytes(image), start).cast<py::tuple>();
    if (backend.size() != 2)
      throw std::invalid_argument("backend factory must return (runtime, is_engine)");
    py::object runtime = backend[0];
    bool engine = py::cast<bool>(backend[1]);
    if (engine)
      machine_ = std::make_unique<PythonEngineMachine>(runtime, image, start);
    else {
      stepped_ = std::make_unique<PythonSteppedRuntime>(runtime);
      machine_ = std::make_unique<Bus>(*stepped_, image, opcodes);
    }
    console_ =
        std::make_unique<Console>(std::move(mapper), std::move(ppu), Apu(sample_rate), *machine_);
  }
  py::array_t<uint8_t> run_frame() {
    auto frame = console_->run_frame();
    py::array_t<uint8_t> out({Ppu::frame_height, Ppu::frame_width});
    std::copy(frame.begin(), frame.end(), out.mutable_data());
    return out;
  }
  void set_buttons(int mask) {
    for (int i = 0; i < 8; ++i)
      console_->set_button(i, ((mask >> i) & 1) != 0);
  }
  py::bytes read_ram(int start, int count) {
    vector<uint8_t> out(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i)
      out[static_cast<size_t>(i)] = static_cast<uint8_t>(machine_->read_ram((start + i) & 0xFFFF));
    return to_bytes(out);
  }
  // Battery-backed PRG RAM ($6000-$7FFF): one native region map in, one
  // bulk read out -- no per-byte pokes, no engine WRITE tokens.
  py::bytes prg_ram() { return read_ram(0x6000, 0x2000); }
  void load_prg_ram(const py::bytes& data) {
    string s = data;
    if (s.size() != 0x2000)
      throw std::invalid_argument("battery image must be 8192 bytes");
    machine_->map_region(0x6000, span(reinterpret_cast<const uint8_t*>(s.data()), s.size()));
  }
  void set_sprite_limit(bool enabled) { console_->ppu.sprite_limit = enabled; }
  py::dict run_tape(const string& path, int tail_frames, int max_frames) {
    const auto result = replay_tape(*console_, InputTape::load(path), tail_frames, max_frames);
    py::dict out;
    out["frames"] = result.frames;
    out["seconds"] = result.seconds;
    out["fps"] = result.seconds > 0.0 ? result.frames / result.seconds : 0.0;
    out["halted"] = result.halted;
    out["cycles"] = console_->cycles;
    return out;
  }
  Console& console() { return *console_; }

private:
  unique_ptr<PythonSteppedRuntime> stepped_;
  unique_ptr<ConsoleMachine> machine_;
  unique_ptr<Console> console_;
};

} // namespace

PYBIND11_MODULE(nes_hardware, m) {
  // The NTSC palette (include/nes/palette.h) as a (64, 3) uint8 array,
  // for Python-side frame rendering (tape analysis, debug dumps).
  py::array_t<uint8_t> pal({64, 3});
  std::copy(NES_PALETTE.begin(), NES_PALETTE.end(), pal.mutable_data());
  m.attr("NES_PALETTE") = pal;
  py::class_<Cartridge>(m, "Cartridge")
      .def_static("load", &Cartridge::load)
      .def_static("parse",
                  [](py::bytes b) {
                    string s = b;
                    return Cartridge::parse(
                        span(reinterpret_cast<const uint8_t*>(s.data()), s.size()));
                  })
      .def_property_readonly("prg",
                             [](const Cartridge& c) {
                               return py::bytes(reinterpret_cast<const char*>(c.prg.data()),
                                                c.prg.size());
                             })
      .def_property_readonly("chr",
                             [](const Cartridge& c) {
                               return py::bytes(reinterpret_cast<const char*>(c.chr.data()),
                                                c.chr.size());
                             })
      .def_readonly("mapper", &Cartridge::mapper)
      .def_readonly("battery", &Cartridge::battery);
  py::class_<ConsoleHost>(m, "Console")
      .def(py::init<Cartridge, py::object, const vector<pair<int, int>>&, double>(),
           py::arg("cartridge"), py::arg("backend_factory"), py::arg("opcodes"),
           py::arg("sample_rate") = 44100.0)
      .def("run_frame", &ConsoleHost::run_frame)
      .def("set_buttons", &ConsoleHost::set_buttons)
      .def("run_tape", &ConsoleHost::run_tape, py::arg("path"), py::arg("tail_frames") = 120,
           py::arg("max_frames") = 0)
      .def("read_ram", &ConsoleHost::read_ram, py::arg("start"), py::arg("count"))
      .def("prg_ram", &ConsoleHost::prg_ram)
      .def("load_prg_ram", &ConsoleHost::load_prg_ram, py::arg("data"))
      .def("set_sprite_limit", &ConsoleHost::set_sprite_limit, py::arg("enabled"))
      .def_property_readonly("frame_index",
                             [](ConsoleHost& h) { return h.console().frame_index; })
      .def_property_readonly("cycles", [](ConsoleHost& h) { return h.console().cycles; })
      .def_property_readonly("halted", [](ConsoleHost& h) { return h.console().halted; })
      .def_property_readonly("last_audio", [](ConsoleHost& h) {
        const auto& a = h.console().last_audio;
        py::array_t<float> out(a.size());
        std::copy(a.begin(), a.end(), out.mutable_data());
        return out;
      });
}
