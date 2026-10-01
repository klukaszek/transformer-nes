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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;
using namespace transformer_nes::nes;

namespace {

using std::function;
using std::int64_t;
using std::size_t;
using std::span;
using std::string;
using std::string_view;
using std::tuple;
using std::uint8_t;
using std::unique_ptr;
using std::vector;

py::bytes to_bytes(span<const uint8_t> data) {
  return py::bytes(reinterpret_cast<const char*>(data.data()),
                   static_cast<py::ssize_t>(data.size()));
}

// A graph engine's token fields in order, which the PPU fills and reads
// by position: "name" entries, or {"stem", n} for stem0..stem(n-1).
struct Fields {
  string name;
  int count = 0;
  Fields(const char* n) : name(n) {}
  Fields(const char* stem, int n) : name(stem), count(n) {}
};
vector<string> fields(std::initializer_list<Fields> spec) {
  vector<string> out;
  for (const Fields& f : spec)
    if (f.count == 0)
      out.push_back(f.name);
    else
      for (int i = 0; i < f.count; ++i)
        out.push_back(f.name + std::to_string(i));
  return out;
}
// ppu2c02/render.py INPUTS, OUTPUTS, ppu2c02/sprites.py's two tokens and
// ppu2c02/registers.py's.
const vector<string> render_inputs = fields(
    {"is_slot", "slot", "hit_set", "fine_x", "ctrl", "mask", {"nt_page", 4}, "latch_lo",
     "latch_hi", "wide_low", "ext_mode", "chr_upper", "v_cx", "v_cy", "v_nt", "v_fy", {"bg", 16},
     {"sprite", 8}});
const vector<string> render_outputs =
    fields({{"pixel", 8}, {"tile", 8}, "new_v_cx", "new_v_cy", "new_v_nt", "new_v_fy", "fetched",
            {"hit", 8}, "new_latch_lo", "new_latch_hi"});
const vector<string> evaluate_inputs = fields({"is_evaluate", "line", "ctrl"});
const vector<string> evaluate_outputs = fields({"count", {"index", 8}});
const vector<string> fetch_inputs =
    fields({"is_fetch", "line", "ctrl", "count", {"index", 8}, "latch_lo", "latch_hi", "wide_low",
            {"bank", 8}});
const vector<string> fetch_outputs = [] {
  vector<string> out;
  for (int k = 0; k < 8; ++k)
    for (int c = 0; c < 8; ++c)
      out.push_back("sprite" + std::to_string(k) + "_pixel" + std::to_string(c));
  for (const char* field : {"_x", "_fetch", "_a12"})
    for (int k = 0; k < 8; ++k)
      out.push_back("sprite" + std::to_string(k) + field);
  out.insert(out.end(), {"new_latch_lo", "new_latch_hi"});
  return out;
}();
const vector<string> register_inputs =
    fields({"is_access", "reg", "value", "is_write", "status", "rendering", {"nt_page", 4},
            "chr_writable", "latch_lo", "latch_hi", {"bank", 8}, "t_cx", "t_cy", "t_nt", "t_fy", "v_cx", "v_cy", "v_nt", "v_fy",
            "fine_x", "w", "ctrl", "mask", "oam_addr", "read_buffer"});
const vector<string> register_outputs =
    fields({"new_t_cx", "new_t_cy", "new_t_nt", "new_t_fy", "new_v_cx", "new_v_cy", "new_v_nt",
            "new_v_fy", "new_fine_x", "new_w", "new_ctrl", "new_mask", "new_oam_addr",
            "new_read_buffer", "read_value", "mem_write", "mem_write_addr", "mem_write_value",
            "clear_vblank", "nmi_rise"});

class PythonSteppedRuntime final : public SteppedRuntime {
public:
  explicit PythonSteppedRuntime(py::object core) : core_(std::move(core)) {}
  int pc() const override { return py::cast<int>(core_.attr("pc")); }
  bool halted() const override { return py::cast<bool>(core_.attr("halted")); }
  void set_nmi() override { core_.attr("set_nmi")(); }
  void set_irq() override { core_.attr("set_irq")(); }
  void clear_irq() override { core_.attr("clear_irq")(); }
  void poke(int a, int v) override { core_.attr("poke")(a, v); }
  Next peek() const override {
    const auto [cycles, read] = core_.attr("peek")().cast<std::pair<int, int>>();
    return {cycles, read};
  }
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

// The generated CPU engine's native machine ABI
// (cpu6502/backends/weights/run_protocol.py: MachineRunOut, MachineAbi).
namespace cpu_engine {
constexpr int abi_version = 1;
struct RunOut {
  const char* reason;
  long cycles;
  int read_addr;
  long read_cycles;
  int pause_ir;
  const int* hit_addr;
  const int* hit_value;
  long n_hits;
};
struct Abi {
  void (*run)(void*, long max_cycles, RunOut*);
  int (*read_ram)(void*, int);
  void (*inject_write)(void*, int, int);
  void (*map_region)(void*, int, const unsigned char*, long);
  void (*add_window)(void*, int, int);
  int (*set_read_lease)(void*, int, int);
  int (*set_read_lease_seq)(void*, int, const unsigned char*, long);
  long (*take_lease_served)(void*, int);
  void (*clear_read_lease)(void*);
  void (*clear_read_lease_addr)(void*, int);
  void (*set_nmi)(void*);
  void (*set_irq)(void*);
  void (*clear_irq)(void*);
};
} // namespace cpu_engine

// The transformer CPU's run loop driven natively: Python sets it up
// (image, watches) and names the shape windows of newly mapped code;
// every run, lease, write and interrupt goes straight through the ABI.
class EngineMachine final : public ConsoleMachine {
public:
  EngineMachine(py::object runtime, const vector<uint8_t>& image, int start)
      : rt_(std::move(runtime)) {
    rt_.attr("load_image")(to_bytes(image), py::arg("start_pc") = start);
    const auto abi = rt_.attr("machine_abi")().cast<py::tuple>();
    if (abi[0].cast<int>() != cpu_engine::abi_version)
      throw std::runtime_error("CPU engine machine ABI version mismatch");
    ctx_ = reinterpret_cast<void*>(abi[1].cast<std::uintptr_t>());
    abi_ = reinterpret_cast<const cpu_engine::Abi*>(abi[2].cast<std::uintptr_t>());
  }
  void attach(Device& d, int lo, int hi, bool writes_only = false) override {
    devices_.attach(d, lo, hi, writes_only);
    if (!writes_only)
      rt_.attr("watch_read")(lo, hi);
    rt_.attr("watch_write")(lo, hi);
  }
  void set_nmi() override { abi_->set_nmi(ctx_); }
  void set_irq() override { abi_->set_irq(ctx_); }
  void clear_irq() override { abi_->clear_irq(ctx_); }
  void clear_read_lease() override {
    flush_queues();
    lease_owners_.clear();
    lease_queues_.clear();
    lease_deadline_ = NEVER;
    abi_->clear_read_lease(ctx_);
  }
  int read_ram(int a) const override { return abi_->read_ram(ctx_, a); }
  void poke(int a, int v) override { abi_->inject_write(ctx_, a, v); }
  void map_region(int a, span<const uint8_t> d) override {
    abi_->map_region(ctx_, a, d.data(), static_cast<long>(d.size()));
    for (auto [shape, head] : region_windows(a, d))
      abi_->add_window(ctx_, shape, head);
  }
  void drop_leases(Device& d) override { drop_device_leases(&d); }
  MachineRun run_cycles(int64_t budget, function<bool()> pause = {},
                        int64_t cycle_base = 0) override {
    int64_t cycles = 0;
    bool halted = false;
    while (cycles < budget && !halted) {
      // A time-limited lease (the PPU's $2002) expires mid-run: run up
      // to its deadline, then drop the leases so the device answers.
      int64_t chunk = budget - cycles;
      if (lease_deadline_ != NEVER) {
        const int64_t left = lease_deadline_ - (cycle_base + cycles);
        if (left <= 0) {
          clear_read_lease();
          continue;
        }
        chunk = std::min(chunk, left);
      }
      cpu_engine::RunOut out;
      abi_->run(ctx_, static_cast<long>(chunk), &out);
      cycles += out.cycles;
      // Paused writes are stamped at the end of the returned chunk.
      cycles += service_writes(out, cycle_base + cycles);
      const string_view reason = out.reason;
      halted = reason == "halt" || reason == "trap";
      // The paused instruction has not run yet; its read lands on its
      // last cycle.  Interrupts the read raised (the PPU catching up
      // past vblank) must wait until that instruction completes, so the
      // pause check is skipped until the next chunk runs it.
      if (reason == "watch_read") {
        cycles += service_read(out.read_addr, cycle_base + cycles + out.read_cycles);
        continue;
      }
      if (pause && pause())
        break;
    }
    return {cycles, halted};
  }

private:
  Device* device_at(int a) const { return devices_.find(a); }
  int service_writes(const cpu_engine::RunOut& out, int64_t cycle) {
    int stalls = 0;
    // Watched writes pause the step that made them, so pause_ir is that
    // step's opcode: RMW writes route through write_rmw (MMC1's quirk).
    const bool rmw = out.n_hits && is_memory_rmw(static_cast<uint8_t>(out.pause_ir));
    // The hit arrays live until the next run; servicing never runs.
    for (long i = 0; i < out.n_hits; ++i) {
      const int a = out.hit_addr[i], v = out.hit_value[i];
      if (auto* d = device_at(a)) {
        drop_device_leases(d);
        if (rmw)
          d->write_rmw(a, v, cycle);
        else
          d->write(a, v, cycle);
        stalls += d->take_stall();
      }
    }
    return stalls;
  }
  int service_read(int a, int64_t cycle) {
    flush_queues();
    auto* d = device_at(a);
    int value = d ? d->read(a, cycle) & 0xFF : read_ram(a);
    int stalls = d ? d->take_stall() : 0;
    if (d) {
      auto lease = d->read_lease(a);
      int evicted = -1;
      if (auto* v = std::get_if<int>(&lease)) {
        evicted = abi_->set_read_lease(ctx_, a, *v);
        lease_queues_.erase(a);
      } else if (auto* q = std::get_if<vector<uint8_t>>(&lease)) {
        evicted = abi_->set_read_lease_seq(ctx_, a, q->data(), static_cast<long>(q->size()));
        lease_queues_.insert(a);
      }
      if (!std::holds_alternative<std::monostate>(lease)) {
        lease_owners_[a] = d;
        lease_deadline_ = std::min(lease_deadline_, d->lease_expiry());
      }
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
      const long count = abi_->take_lease_served(ctx_, addr);
      if (auto owner = lease_owners_.find(addr); count && owner != lease_owners_.end())
        owner->second->lease_consumed(addr, static_cast<int>(count));
    }
  }
  void drop_device_leases(Device* device) {
    vector<int> drop;
    for (auto [addr, owner] : lease_owners_)
      if (owner == device)
        drop.push_back(addr);
    for (int addr : drop) {
      if (lease_queues_.erase(addr)) {
        const long count = abi_->take_lease_served(ctx_, addr);
        if (count)
          device->lease_consumed(addr, static_cast<int>(count));
      }
      abi_->clear_read_lease_addr(ctx_, addr);
      lease_owners_.erase(addr);
    }
  }
  // Shape windows of a mapped region, named by Python once per distinct
  // (base, content) and replayed natively on every later bank switch.
  struct Region {
    int base;
    vector<uint8_t> data;
    vector<std::pair<int, int>> windows;
  };
  const vector<std::pair<int, int>>& region_windows(int base, span<const uint8_t> d) {
    uint64_t h = 1469598103934665603ULL ^ static_cast<uint64_t>(base);
    for (size_t i = 0; i < d.size(); i += 8) {
      uint64_t word = 0;
      std::memcpy(&word, d.data() + i, std::min<size_t>(8, d.size() - i));
      h = (h ^ word) * 1099511628211ULL;
    }
    auto& bucket = regions_[h];
    for (const Region& r : bucket)
      if (r.base == base && std::ranges::equal(r.data, d))
        return r.windows;
    Region r{base, vector<uint8_t>(d.begin(), d.end()), {}};
    for (py::handle w : rt_.attr("region_windows")(base, to_bytes(d)))
      r.windows.emplace_back(py::cast<int>(w[py::int_(0)]), py::cast<int>(w[py::int_(1)]));
    return bucket.emplace_back(std::move(r)).windows;
  }
  py::object rt_;
  void* ctx_ = nullptr;
  const cpu_engine::Abi* abi_ = nullptr;
  DeviceMap devices_;
  int64_t lease_deadline_ = NEVER;
  std::unordered_map<int, Device*> lease_owners_;
  std::unordered_set<int> lease_queues_;
  std::unordered_map<uint64_t, vector<Region>> regions_;
};

class ConsoleHost {
public:
  ConsoleHost(Cartridge cart, py::object factory, double sample_rate) {
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
      machine_ = std::make_unique<EngineMachine>(runtime, image, start);
    else {
      stepped_ = std::make_unique<PythonSteppedRuntime>(runtime);
      machine_ = std::make_unique<Bus>(*stepped_, image);
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
  // The overlay's X-ray: which render model drew each slot of the last
  // frame (Ppu::completed_expert_map).
  py::bytes ppu_expert_map() const {
    const auto& map = console_->ppu.completed_expert_map();
    return py::bytes(reinterpret_cast<const char*>(map.data()), map.size());
  }
  // Run the PPU as the compiled transformer built by
  // ppu2c02.machine.ppu_engine (register accesses, tile slots, the sprite
  // unit), kept alive here; None restores the native PPU.  Its token
  // graphs must be the PPU's, checked field by field.
  void set_ppu_engine(py::object engine) {
    if (engine.is_none()) {
      console_->ppu.set_engine({});
      ppu_engine_ = py::none();
      return;
    }
    const py::dict graphs = engine.attr("graphs");
    auto token_is = [&](const char* name, const vector<string>& in, const vector<string>& out) {
      if (!graphs.contains(name))
        return false;
      const py::object g = graphs[name];
      return py::cast<vector<string>>(g.attr("inputs")) == in &&
             py::cast<vector<string>>(g.attr("outputs")) == out;
    };
    if (!token_is("render", render_inputs, render_outputs) ||
        !token_is("evaluate", evaluate_inputs, evaluate_outputs) ||
        !token_is("fetch", fetch_inputs, fetch_outputs) ||
        !token_is("access", register_inputs, register_outputs))
      throw std::invalid_argument("engine is not the PPU graphs");
    const auto abi = engine.attr("abi")().cast<py::tuple>();
    Ppu::PpuEngine api;
    api.ctx = reinterpret_cast<void*>(abi[0].cast<std::uintptr_t>());
    api.write = reinterpret_cast<void (*)(void*, int, int)>(abi[1].cast<std::uintptr_t>());
    api.step =
        reinterpret_cast<void (*)(void*, int, const int*, int*)>(abi[2].cast<std::uintptr_t>());
    api.trace =
        reinterpret_cast<void (*)(void*, const int*, int*)>(abi[3].cast<std::uintptr_t>());
    api.trace_rows = abi[4].cast<int>();
    api.trace_width = abi[5].cast<int>();
    const auto dispatch = engine.attr("dispatch").cast<py::list>();
    for (size_t latched = 0; latched < 2; ++latched) {
      const py::dict d = dispatch[latched];
      auto& out = api.dispatch[latched];
      const auto inner = d["inner"].cast<vector<vector<int>>>();
      for (size_t t = 0; t < 2; ++t)
        std::ranges::copy(inner[t], out.inner[t].begin());
      out.sprites = static_cast<uint8_t>(d["sprites"].cast<int>());
      std::ranges::copy(d["first"].cast<vector<int>>(), out.first.begin());
      std::ranges::copy(d["last"].cast<vector<int>>(), out.last.begin());
    }
    const py::dict sprites = engine.attr("sprite_dispatch");
    std::ranges::copy(sprites["evaluate"].cast<vector<int>>(), api.sprites.evaluate.begin());
    std::ranges::copy(sprites["fetch"].cast<vector<int>>(), api.sprites.fetch.begin());
    const auto registers = engine.attr("register_dispatch").cast<vector<vector<int>>>();
    for (size_t w = 0; w < 2; ++w)
      std::ranges::copy(registers[w], api.registers[w].begin());
    ppu_engine_ = engine;
    console_->ppu.set_engine(api);
  }
  // The X-ray's pixel dissection (Ppu::request_slot_trace): the model
  // that drew the slot and the full model's residual stream after the
  // embedding and each layer; None until the slot is drawn.
  void request_ppu_slot_trace(int line, int slot) { console_->ppu.request_slot_trace(line, slot); }
  py::object take_ppu_slot_trace() {
    auto trace = console_->ppu.take_slot_trace();
    if (!trace)
      return py::none();
    py::dict out;
    out["line"] = trace->line;
    out["slot"] = trace->slot;
    out["model"] = trace->model;
    out["residual"] = py::array_t<int>(static_cast<py::ssize_t>(trace->residual.size()),
                                       trace->residual.data());
    return out;
  }
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
  py::object ppu_engine_ = py::none();
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
      .def(py::init<Cartridge, py::object, double>(), py::arg("cartridge"),
           py::arg("backend_factory"), py::arg("sample_rate") = 44100.0)
      .def("run_frame", &ConsoleHost::run_frame)
      .def("set_buttons", &ConsoleHost::set_buttons)
      .def("run_tape", &ConsoleHost::run_tape, py::arg("path"), py::arg("tail_frames") = 120,
           py::arg("max_frames") = 0)
      .def("read_ram", &ConsoleHost::read_ram, py::arg("start"), py::arg("count"))
      .def("prg_ram", &ConsoleHost::prg_ram)
      .def("load_prg_ram", &ConsoleHost::load_prg_ram, py::arg("data"))
      .def("set_sprite_limit", &ConsoleHost::set_sprite_limit, py::arg("enabled"))
      .def("ppu_expert_map", &ConsoleHost::ppu_expert_map,
           "Which render model drew each 8-pixel slot of the last frame "
           "(240x32 bytes: 0 full graph, k expert k, 255 native)")
      .def("set_ppu_engine", &ConsoleHost::set_ppu_engine, py::arg("engine"))
      .def("request_ppu_slot_trace", &ConsoleHost::request_ppu_slot_trace, py::arg("line"),
           py::arg("slot"))
      .def("take_ppu_slot_trace", &ConsoleHost::take_ppu_slot_trace)
      .def_property(
          "ppu_force_full_render",
          [](ConsoleHost& h) { return h.console().ppu.force_full_render; },
          [](ConsoleHost& h, bool on) { h.console().ppu.force_full_render = on; },
          "render every graph slot with the full model (no folded experts)")
      .def_property_readonly("ppu_graph_chr_full",
                             [](ConsoleHost& h) { return h.console().ppu.graph_chr_full(); })
      .def_property_readonly("frame_index",
                             [](ConsoleHost& h) { return h.console().frame_index; })
      .def_property_readonly("cycles", [](ConsoleHost& h) { return h.console().cycles; })
      .def_property_readonly("controller_polls",
                             [](ConsoleHost& h) { return h.console().controller.polls; })
      .def_property_readonly("halted", [](ConsoleHost& h) { return h.console().halted; })
      .def_property_readonly("last_audio", [](ConsoleHost& h) {
        const auto& a = h.console().last_audio;
        py::array_t<float> out(a.size());
        std::copy(a.begin(), a.end(), out.mutable_data());
        return out;
      });
}
