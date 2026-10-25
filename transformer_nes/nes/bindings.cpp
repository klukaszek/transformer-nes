#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../cpu6502/backends/weights/native/machine_abi.h"
#include "nes/apu.h"
#include "nes/bus.h"
#include "nes/cartridge.h"
#include "nes/console.h"
#include "nes/mappers/mappers.h"
#include "nes/ppu.h"
#include "nes/tape.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string_view>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

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
// ppu2c02/render.py INPUTS, OUTPUTS, ppu2c02/sprites.py's tokens and
// ppu2c02/registers.py's.
const vector<string> render_inputs = fields(
    {"is_slot", "slot", "hit_set", "fine_x", "ctrl", "mask", {"nt_page", 4}, "latch_lo",
     "latch_hi", "wide_low", "ext_mode", "chr_upper", "v_cx", "v_cy", "v_nt", "v_fy", {"bg", 16},
     {"sprite", 8}});
const vector<string> render_outputs =
    fields({{"pixel", 8}, {"tile", 8}, "new_v_cx", "new_v_cy", "new_v_nt", "new_v_fy", "fetched",
            {"hit", 8}, "new_latch_lo", "new_latch_hi"});
const vector<string> evaluate_inputs = fields({"is_evaluate", "line", "ctrl", "start"});
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
            "clear_vblank", "nmi_rise", "nmi_fall"});

class PythonSteppedRuntime final : public SteppedRuntime {
public:
  explicit PythonSteppedRuntime(py::object core) : core_(std::move(core)) {}
  bool halted() const override { return py::cast<bool>(core_.attr("halted")); }
  void set_nmi() override { core_.attr("set_nmi")(); }
  void set_irq() override { core_.attr("set_irq")(); }
  void clear_irq() override { core_.attr("clear_irq")(); }
  void poke(int a, int v) override { core_.attr("poke")(a, v); }
  Next peek() const override {
    const auto [cycles, read] = core_.attr("peek")().cast<std::pair<int, int>>();
    return {cycles, read};
  }
  vector<CpuWrite> step() override {
    py::object row = core_.attr("step")();
    vector<CpuWrite> out;
    for (py::handle w : row.attr("writes"))
      out.push_back({py::cast<int>(w.attr("addr")), py::cast<int>(w.attr("value")),
                     py::cast<int>(w.attr("original"))});
    return out;
  }

private:
  py::object core_;
};

// The transformer CPU's run loop driven natively: Python sets it up
// (image, watches) and names the shape windows of newly mapped code;
// every run, lease, write and interrupt goes straight through the ABI.
class EngineMachine final : public ConsoleMachine {
public:
  EngineMachine(py::object runtime, const vector<uint8_t>& image, int start,
                span<const uint8_t> prg_rom)
      : rt_(std::move(runtime)), prg_rom_(prg_rom) {
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
    leases_.fill({});
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
    // The hit arrays live until the next run; servicing never runs.
    for (long i = 0; i < out.n_hits; ++i) {
      const int a = out.hit_addr[i], v = out.hit_value[i];
      if (auto* d = device_at(a)) {
        drop_device_leases(d);
        if (out.hit_original[i] >= 0)
          d->write_rmw(a, v, out.hit_original[i], cycle);
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
      const bool queued = std::holds_alternative<ReadReply>(lease);
      if (auto* v = std::get_if<int>(&lease)) {
        evicted = abi_->set_read_lease(ctx_, a, *v);
      } else if (auto* q = std::get_if<ReadReply>(&lease)) {
        evicted = abi_->set_read_lease_seq(ctx_, a, q->bytes.data(), q->size);
      }
      if (evicted >= 0 && evicted != a)
        if (auto* old = lease_at(evicted))
          *old = {};
      if (!std::holds_alternative<std::monostate>(lease)) {
        auto* slot = lease_at(a);
        if (!slot)
          slot = lease_at(-1);
        if (!slot)
          throw std::logic_error("CPU lease capacity disagrees with device tracking");
        *slot = {d, a, queued};
        lease_deadline_ = std::min(lease_deadline_, d->lease_expiry());
      }
    }
    poke(a, value);
    return stalls;
  }
  struct Lease {
    Device* owner = nullptr;
    int address = -1;
    bool queued = false;
  };
  Lease* lease_at(int address) {
    for (auto& lease : leases_)
      if (lease.address == address)
        return &lease;
    return nullptr;
  }
  void flush_queue(const Lease& lease) {
    if (lease.queued) {
      const long count = abi_->take_lease_served(ctx_, lease.address);
      if (count)
        lease.owner->lease_consumed(lease.address, static_cast<int>(count));
    }
  }
  void flush_queues() {
    for (const auto& lease : leases_)
      flush_queue(lease);
  }
  void release_lease(Lease& lease) {
    flush_queue(lease);
    abi_->clear_read_lease_addr(ctx_, lease.address);
    lease = {};
  }
  void drop_device_leases(Device* device) {
    for (auto& lease : leases_)
      if (lease.owner == device)
        release_lease(lease);
  }
  using Windows = vector<std::pair<int, int>>;
  Windows match_region(int base, span<const uint8_t> data) {
    Windows windows;
    for (py::handle w : rt_.attr("region_windows")(base, to_bytes(data)))
      windows.emplace_back(py::cast<int>(w[py::int_(0)]), py::cast<int>(w[py::int_(1)]));
    return windows;
  }
  // ROM windows key on immutable slices; RAM windows require content validation.
  struct Region {
    int base;
    vector<uint8_t> data;
    Windows windows;
  };
  const Windows& region_windows(int base, span<const uint8_t> d) {
    const auto offset = reinterpret_cast<std::uintptr_t>(d.data()) -
                        reinterpret_cast<std::uintptr_t>(prg_rom_.data());
    if (offset <= prg_rom_.size() && d.size() <= prg_rom_.size() - offset) {
      const auto key = std::tuple{base, offset, d.size()};
      if (const auto found = rom_regions_.find(key); found != rom_regions_.end())
        return found->second;
      return rom_regions_.emplace(key, match_region(base, d)).first->second;
    }
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
    Region r{base, vector<uint8_t>(d.begin(), d.end()), match_region(base, d)};
    return bucket.emplace_back(std::move(r)).windows;
  }
  py::object rt_;
  span<const uint8_t> prg_rom_;
  std::map<std::tuple<int, size_t, size_t>, Windows> rom_regions_;
  void* ctx_ = nullptr;
  const cpu_engine::Abi* abi_ = nullptr;
  DeviceMap devices_;
  int64_t lease_deadline_ = NEVER;
  array<Lease, cpu_engine::read_lease_capacity> leases_;
  std::unordered_map<uint64_t, vector<Region>> regions_;
};

class ConsoleHost {
public:
  ConsoleHost(Cartridge cart, py::object factory, double sample_rate) {
    const bool chr_writable = cart.chr.empty();
    const bool vertical = cart.mirroring == Mirroring::Vertical;
    auto mapper = make_mapper(std::move(cart));
    // Mapper constructors retain geometry; the PPU owns the CHR payload.
    Ppu ppu(mapper->take_chr(), chr_writable, vertical);
    if (auto* bank = mapper->bank_state()) {
      const auto module = py::module_::import("transformer_nes.nes.mappers");
      mapper_bank_engine_ = module.attr("mapper_bank_engine")(bank->chip, bank->n_prg, bank->n_chr);
      const py::dict graphs = mapper_bank_engine_.attr("graphs");
      const py::object transition = graphs["transition"], layout = graphs["layout"];
      const auto inputs = transition.attr("inputs").cast<vector<string>>();
      if (inputs.size() < 5 || inputs.size() > bank->states.front().size())
        throw std::invalid_argument("invalid mapper register count");
      bank->state_count = static_cast<int>(inputs.size()) - 4;
      vector<string> state(inputs.begin(), inputs.begin() + bank->state_count), outputs;
      for (const auto& name : state)
        outputs.push_back("new_" + name);
      outputs.push_back("dirty");
      auto maps = fields({{"map_prg", static_cast<int>(bank->prg_count)},
                          {"map_chr", static_cast<int>(bank->chr_count)}, "map_mirror"});
      if (bank->chip == BankControl::Mmc5) {
        const auto extra = fields({{"map_ram", 4}, "map_ram_bank", "map_writable",
                                   "map_exram_mode", "map_fill_tile", "map_fill_color", "map_upper",
                                   "map_mul_lo", "map_mul_hi"});
        maps.insert(maps.end(), extra.begin(), extra.end());
      }
      if (vector<string>(inputs.end() - 4, inputs.end()) != fields({"event", "port", "value", "original"}) ||
          transition.attr("outputs").cast<vector<string>>() != outputs ||
          layout.attr("inputs").cast<vector<string>>() != state ||
          layout.attr("outputs").cast<vector<string>>() != maps)
        throw std::invalid_argument("engine is not the mapper bank graphs");
      const auto abi = mapper_bank_engine_.attr("abi")().cast<py::tuple>();
      bank->engine.ctx = reinterpret_cast<void*>(abi[0].cast<std::uintptr_t>());
      bank->engine.step = reinterpret_cast<void (*)(void*, int, const int*, int*)>(
          abi[2].cast<std::uintptr_t>());
      bank->engine.reset = mapper_bank_engine_.attr("model_id")("transition", "reset").cast<int>();
      bank->engine.layout = mapper_bank_engine_.attr("model_id")("layout").cast<int>();
      mapper_control_engine_ = module.attr("mapper_control_engine")();
      const string access_name = "access_" + std::to_string(bank->chip);
      const string address_name = "address_" + std::to_string(bank->chip);
      const py::dict control_graphs = mapper_control_engine_.attr("graphs");
      const auto access_inputs = fields({"addr_lo", "addr_hi", "is_write", "original",
                                         {"bank", 4}, {"ram", 4}, "ram_bank", "writable", "mul_lo", "mul_hi"});
      for (const auto& name : {access_name, address_name}) {
        const py::object token = control_graphs[py::str(name)];
        const auto outputs = name == access_name ?
            fields({"port_index", "bank_event", "irq_event", "schedule_changed", "read_source",
                    "read_index", "read_value", "write_source", "repair"}) :
            fields({"read_source", "read_index", "read_value"});
        if (token.attr("inputs").cast<vector<string>>() != access_inputs ||
            token.attr("outputs").cast<vector<string>>() != outputs)
          throw std::invalid_argument("engine is not the mapper access graphs");
      }
      const auto control_abi = mapper_control_engine_.attr("abi")().cast<py::tuple>();
      bank->engine.access_ctx = reinterpret_cast<void*>(control_abi[0].cast<std::uintptr_t>());
      bank->engine.access_step = reinterpret_cast<void (*)(void*, int, const int*, int*)>(
          control_abi[2].cast<std::uintptr_t>());
      bank->engine.access_models = {
          mapper_control_engine_.attr("model_id")(access_name, "read").cast<int>(),
          mapper_control_engine_.attr("model_id")(access_name, "write").cast<int>()};
      bank->engine.address_model = mapper_control_engine_.attr("model_id")(address_name).cast<int>();
      for (int port : module.attr("BANK_PORTS")[py::int_(bank->chip)].cast<vector<int>>()) {
        char name[32];
        std::snprintf(name, sizeof name, "write_%X", port);
        const int write = mapper_bank_engine_.attr("model_id")("transition", name).cast<int>();
        int rmw = write;
        if (bank->chip == BankControl::Mmc1) {
          std::snprintf(name, sizeof name, "rmw_%X", port);
          rmw = mapper_bank_engine_.attr("model_id")("transition", name).cast<int>();
        }
        bank->engine.ports.push_back({port, write, rmw});
      }
      bank->reset();
    }
    if (auto* irq = mapper->irq_state()) {
      mapper_control_engine_ = py::module_::import("transformer_nes.nes.mappers")
                               .attr("mapper_control_engine")();
      const char* chip = irq->chip == MapperIrq::Mmc3 ? "mmc3" : "mmc5";
      const py::object graph = mapper_control_engine_.attr("graphs")[py::str(chip)];
      if (graph.attr("inputs").cast<vector<string>>() !=
              fields({"counter", "target", "phase", "pending", "enabled", "event", "value"}) ||
          graph.attr("outputs").cast<vector<string>>() !=
              fields({"new_counter", "new_target", "new_phase", "new_pending", "new_enabled",
                      "irq", "read_value", "predictable", "next_edge", "irq_write"}))
        throw std::invalid_argument("engine is not the mapper IRQ graphs");
      const auto abi = mapper_control_engine_.attr("abi")().cast<py::tuple>();
      irq->engine.ctx = reinterpret_cast<void*>(abi[0].cast<std::uintptr_t>());
      irq->engine.step = reinterpret_cast<void (*)(void*, int, const int*, int*)>(
          abi[2].cast<std::uintptr_t>());
      const array<const char*, MapperIrq::event_count> events{
          "clock", "idle", "target", "reload", "enable", "disable", "ack"};
      for (size_t i = 0; i < events.size(); ++i)
        irq->engine.models[i] = mapper_control_engine_.attr("model_id")(chip, events[i]).cast<int>();
    }
    auto image = mapper->cpu_image();
    int start = image[0xFFFC] | (image[0xFFFD] << 8);
    py::tuple backend = factory(to_bytes(image), start).cast<py::tuple>();
    if (backend.size() != 2)
      throw std::invalid_argument("backend factory must return (runtime, is_engine)");
    py::object runtime = backend[0];
    bool engine = py::cast<bool>(backend[1]);
    if (engine)
      machine_ = std::make_unique<EngineMachine>(runtime, image, start, mapper->prg_rom());
    else {
      stepped_ = std::make_unique<PythonSteppedRuntime>(runtime);
      machine_ = std::make_unique<Bus>(*stepped_, image);
    }
    console_ =
        std::make_unique<Console>(std::move(mapper), std::move(ppu), Apu(sample_rate), *machine_);
    const py::object audio = py::module_::import("transformer_nes.nes.audio");
    const py::object oscillators = py::module_::import("transformer_nes.nes.oscillators");
    const int maximum = static_cast<int>(std::clamp(
        std::ceil(console_->apu.cycles_per_sample), 1.0,
        static_cast<double>(oscillators.attr("MAX_CYCLES").cast<int>())));
    audio_engine_ = audio.attr("audio_engine")(maximum);
    const py::dict audio_graphs = audio_engine_.attr("graphs");
    for (const auto& [name, input, output] :
         {std::tuple{"control", "INPUTS", "OUTPUTS"},
          {"frame", "FRAME_INPUTS", "FRAME_OUTPUTS"}, {"program", "PROGRAM_INPUTS", "PROGRAM_OUTPUTS"},
          {"dmc", "DMC_INPUTS", "DMC_OUTPUTS"}}) {
      const py::object graph = audio_graphs[name];
      if (graph.attr("inputs").cast<vector<string>>() != audio.attr(input).cast<vector<string>>() ||
          graph.attr("outputs").cast<vector<string>>() != audio.attr(output).cast<vector<string>>())
        throw std::invalid_argument("engine is not the APU graphs");
    }
    if (py::len(audio.attr("INPUTS")) != Apu::channel_input_count ||
        py::len(audio.attr("OUTPUTS")) != Apu::channel_output_count ||
        py::len(audio.attr("PROGRAM_INPUTS")) != 59 || py::len(audio.attr("PROGRAM_OUTPUTS")) != 57 ||
        py::len(audio.attr("DMC_INPUTS")) != 17 || py::len(audio.attr("DMC_OUTPUTS")) != 20)
      throw std::invalid_argument("invalid APU token size");
    for (const auto& [name, inputs, outputs, ni, no] :
         {std::tuple{"oscillate", "CLOCK_INPUTS", "CLOCK_OUTPUTS", size_t{15}, size_t{9}},
          {"tick", "CLOCK_INPUTS", "CLOCK_OUTPUTS", size_t{15}, size_t{9}},
          {"noise", "NOISE_INPUTS", "NOISE_OUTPUTS", size_t{17}, size_t{15}}}) {
      const py::object graph = audio_graphs[name];
      if (graph.attr("inputs").cast<vector<string>>() != oscillators.attr(inputs).cast<vector<string>>() ||
          graph.attr("outputs").cast<vector<string>>() != oscillators.attr(outputs).cast<vector<string>>() ||
          py::len(graph.attr("inputs")) != ni || py::len(graph.attr("outputs")) != no)
        throw std::invalid_argument("engine is not the oscillator graphs");
    }
    const auto audio_abi = audio_engine_.attr("abi")().cast<py::tuple>();
    Apu::Engine audio_api;
    audio_api.ctx = reinterpret_cast<void*>(audio_abi[0].cast<std::uintptr_t>());
    audio_api.step = reinterpret_cast<void (*)(void*, int, const int*, int*)>(
        audio_abi[2].cast<std::uintptr_t>());
    const array<const char*, 3> channel_events{"quarter", "half", "both"};
    const array<const char*, 4> frame_events{"clock", "write", "read", "observe"};
    for (size_t i = 0; i < channel_events.size(); ++i)
      audio_api.control[i] = audio_engine_.attr("model_id")("control", channel_events[i]).cast<int>();
    for (size_t i = 0; i < frame_events.size(); ++i)
      audio_api.frame[i] = audio_engine_.attr("model_id")("frame", frame_events[i]).cast<int>();
    audio_api.program = audio_engine_.attr("model_id")("program").cast<int>();
    audio_api.program_writes.fill(audio_api.program);
    for (const int port : audio.attr("WRITE_PORTS").cast<vector<int>>())
      audio_api.program_writes[static_cast<size_t>(port)] =
          audio_engine_.attr("model_id")("program", "write_" + std::to_string(port)).cast<int>();
    audio_api.dmc[0] = audio_engine_.attr("model_id")("dmc").cast<int>();
    const array<const char*, 3> dmc_events{"fetch", "clock", "observe"};
    for (size_t i = 0; i < dmc_events.size(); ++i)
      audio_api.dmc[i + 1] = audio_engine_.attr("model_id")("dmc", dmc_events[i]).cast<int>();
    audio_api.dmc_writes.fill(audio_api.dmc[0]);
    for (const int port : audio.attr("DMC_WRITE_PORTS").cast<vector<int>>())
      audio_api.dmc_writes[static_cast<size_t>(port)] =
          audio_engine_.attr("model_id")("dmc", "write_" + std::to_string(port)).cast<int>();
    for (size_t tick = 0; tick < 2; ++tick) {
      audio_api.oscillators[tick] = audio_engine_.attr("model_id")(
          tick ? "tick" : "oscillate").cast<int>();
      for (size_t mode = 0; mode < 2; ++mode)
        audio_api.noise[tick][mode] = audio_engine_.attr("model_id")(
            "noise", string(tick ? "tick_mode" : "mode") + std::to_string(mode)).cast<int>();
    }
    sample_engine_ = audio.attr("sample_engine")();
    const py::object sample_token = sample_engine_.attr("graphs")["sample"];
    if (sample_token.attr("inputs").cast<vector<string>>() != audio.attr("SAMPLE_INPUTS").cast<vector<string>>() ||
        sample_token.attr("outputs").cast<vector<string>>() != audio.attr("SAMPLE_OUTPUTS").cast<vector<string>>() ||
        py::len(sample_token.attr("inputs")) != 55 || py::len(sample_token.attr("outputs")) != 3 ||
        sample_engine_.attr("dtype").attr("name").cast<string>() != "float32")
      throw std::invalid_argument("engine is not the float32 APU sample graph");
    const auto sample_abi = sample_engine_.attr("abi")().cast<py::tuple>();
    audio_api.sample_ctx = reinterpret_cast<void*>(sample_abi[0].cast<std::uintptr_t>());
    audio_api.sample_step = reinterpret_cast<void (*)(void*, int, const float*, float*)>(
        sample_abi[2].cast<std::uintptr_t>());
    audio_api.sample_model = sample_engine_.attr("model_id")("sample").cast<int>();
    console_->apu.set_engine(audio_api);
    controller_engine_ = py::module_::import("transformer_nes.nes.controller")
                             .attr("controller_engine")();
    const py::object token = controller_engine_.attr("graphs")["controller"];
    if (token.attr("inputs").cast<vector<string>>() !=
            fields({"strobe", "index", {"button", 8}, "event", "value", "count"}) ||
        token.attr("outputs").cast<vector<string>>() !=
            fields({"new_strobe", "new_index", "reply", "length", {"byte", 8}}))
      throw std::invalid_argument("engine is not the controller graph");
    const auto controller_abi = controller_engine_.attr("abi")().cast<py::tuple>();
    auto& control = console_->controller.engine;
    control.ctx = reinterpret_cast<void*>(controller_abi[0].cast<std::uintptr_t>());
    control.step = reinterpret_cast<void (*)(void*, int, const int*, int*)>(
        controller_abi[2].cast<std::uintptr_t>());
    const array<const char*, Controller::EventCount> events{"write", "read", "lease", "consume"};
    for (size_t i = 0; i < events.size(); ++i)
      control.models[i] = controller_engine_.attr("model_id")("controller", events[i]).cast<int>();
#ifndef TRANSFORMER_NES_PPU_REFERENCE
    set_ppu_engine(py::module_::import("transformer_nes.ppu2c02.machine")
                       .attr("ppu_engine")(console_->ppu.graph_chr_full()));
#endif
  }
  py::array_t<uint8_t> run_frame() {
    const auto& frame = console_->run_frame();
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
  // unit), kept alive here. Its token graphs must be the PPU's,
  // checked field by field.
  void set_ppu_engine(py::object engine) {
    if (engine.is_none()) {
#ifdef TRANSFORMER_NES_PPU_REFERENCE
      console_->ppu.set_engine({});
      ppu_engine_ = py::none();
      ppu_control_engine_ = py::none();
      return;
#else
      throw std::invalid_argument("PPU transformer engine is required");
#endif
    }
    const py::dict graphs = engine.attr("graphs");
    auto token_is = [&](const char* name, const vector<string>& in, const vector<string>& out) {
      if (!graphs.contains(name))
        return false;
      const py::object g = graphs[name];
      return py::cast<vector<string>>(g.attr("inputs")) == in &&
             py::cast<vector<string>>(g.attr("outputs")) == out;
    };
    vector<string> fetch_status_outputs(fetch_outputs.begin(), fetch_outputs.begin() + 8);
    fetch_status_outputs.push_back(fetch_outputs[64]);
    fetch_status_outputs.insert(fetch_status_outputs.end(), fetch_outputs.begin() + 72,
                                fetch_outputs.end());
    if (!token_is("render", render_inputs, render_outputs) ||
        !token_is("render_status", render_inputs,
                  vector<string>(render_outputs.begin() + 8, render_outputs.end())) ||
        !token_is("fetch_status", fetch_inputs, fetch_status_outputs) ||
        !token_is("evaluate", evaluate_inputs, evaluate_outputs) ||
        !token_is("fetch", fetch_inputs, fetch_outputs) ||
        !token_is("fetch_bus", fetch_inputs,
                  vector<string>(fetch_outputs.begin() + 72, fetch_outputs.end())) ||
        !token_is("overflow", {"is_overflow", "line", "ctrl", "start"}, {"index"}) ||
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
      out.blank = static_cast<uint8_t>(d["blank"].cast<int>());
      std::ranges::copy(d["first"].cast<vector<int>>(), out.first.begin());
      std::ranges::copy(d["last"].cast<vector<int>>(), out.last.begin());
    }
    const py::dict sprites = engine.attr("sprite_dispatch");
    std::ranges::copy(sprites["evaluate"].cast<vector<int>>(), api.sprites.evaluate.begin());
    std::ranges::copy(sprites["fetch"].cast<vector<int>>(), api.sprites.fetch.begin());
    std::ranges::copy(sprites["fetch_bus"].cast<vector<int>>(), api.sprites.fetch_bus.begin());
    std::ranges::copy(sprites["fetch_status"].cast<vector<int>>(), api.sprites.fetch_status.begin());
    api.render_status = engine.attr("status_render_dispatch").cast<vector<int>>();
    std::ranges::copy(sprites["overflow"].cast<vector<int>>(), api.sprites.overflow.begin());
    std::ranges::copy(sprites["debug_fetch"].cast<vector<int>>(), api.sprites.debug_fetch.begin());
    const auto registers = engine.attr("register_dispatch").cast<vector<vector<int>>>();
    for (size_t w = 0; w < 2; ++w)
      std::ranges::copy(registers[w], api.registers[w].begin());
    py::object control = py::module_::import("transformer_nes.ppu2c02.control")
                             .attr("control_engine")();
    const py::object token = control.attr("graphs")["control"];
    if (token.attr("inputs").cast<vector<string>>() !=
            fields({"event", "mask", "ctrl", "odd", "phase", "flag_age", "vblank", "suppressed",
                    "nmi_edge", "nmi_delayed", "t_cx", "t_cy", "t_nt", "t_fy",
                    "v_cx", "v_cy", "v_nt", "v_fy", "oam_addr", "clear_vblank",
                    "nmi_rise", "nmi_fall", "status_pending"}) ||
        token.attr("outputs").cast<vector<string>>() !=
            fields({"new_v_cx", "new_v_cy", "new_v_nt", "new_v_fy", "new_oam_addr",
                    "new_vblank", "new_suppressed", "new_nmi_edge", "new_nmi_delayed",
                    "set_stamp", "clear_stamp", "clear_status", "status_changed", "skip_dot",
                    "rendering"}))
      throw std::invalid_argument("engine is not the PPU control graph");
    const auto control_abi = control.attr("abi")().cast<py::tuple>();
    api.control_ctx = reinterpret_cast<void*>(control_abi[0].cast<std::uintptr_t>());
    api.control_step = reinterpret_cast<void (*)(void*, int, const int*, int*)>(
        control_abi[2].cast<std::uintptr_t>());
    const array<const char*, 6> events{"access", "vertical_copy", "sprite_line", "odd_skip",
                                      "vblank", "clear"};
    for (size_t i = 0; i < events.size(); ++i)
      api.control_models[i] = control.attr("model_id")("control", events[i]).cast<int>();
    const py::object module = py::module_::import("transformer_nes.ppu2c02.control");
    const py::dict graphs_control = control.attr("graphs");
    for (const auto& [name, input, output, ni, no] :
         {std::tuple{"bus", "BUS_INPUTS", "BUS_OUTPUTS", 6, 4},
          {"background_span", "SPAN_INPUTS", "SPAN_OUTPUTS", 5, 6},
          {"access_bus", "ACCESS_INPUTS", "ACCESS_OUTPUTS", 17, 7},
          {"access_delay", "ACCESS_INPUTS", "DELAY_OUTPUTS", 17, 1},
          {"view", "VIEW_INPUTS", "VIEW_OUTPUTS", 4, 6},
          {"compose", "COMPOSE_INPUTS", "COMPOSE_OUTPUTS", 19, 10},
          {"dma", "DMA_INPUTS", "DMA_OUTPUTS", 4, 7},
          {"dma_request", "DMA_INPUTS", "DMA_REQUEST", 4, 2},
          {"dma_transfer", "DMA_INPUTS", "DMA_TRANSFER", 4, 5}}) {
      const py::object graph = graphs_control[name];
      if (graph.attr("inputs").cast<vector<string>>() != module.attr(input).cast<vector<string>>() ||
          graph.attr("outputs").cast<vector<string>>() != module.attr(output).cast<vector<string>>() ||
          py::len(graph.attr("inputs")) != static_cast<size_t>(ni) ||
          py::len(graph.attr("outputs")) != static_cast<size_t>(no))
        throw std::invalid_argument("engine is not the PPU bus graphs");
    }
    api.background_span_model = control.attr("model_id")("background_span").cast<int>();
    const array<const char*, 3> bus_events{"pin", "background", "sprite"};
    for (size_t i = 0; i < bus_events.size(); ++i)
      api.bus_models[i] = control.attr("model_id")("bus", bus_events[i]).cast<int>();
    for (int w = 0; w < 2; ++w)
      for (int reg = 0; reg < 8; ++reg)
        api.access_bus_models[w][reg] = control.attr("model_id")(
            "access_bus", string(w ? "write" : "read") + std::to_string(reg)).cast<int>();
    api.view_model = control.attr("model_id")("view").cast<int>();
    api.access_delay_model = control.attr("model_id")("access_delay").cast<int>();
    api.compose_model = control.attr("model_id")("compose").cast<int>();
    api.dma_request_model = control.attr("model_id")("dma_request").cast<int>();
    api.dma_transfer_model = control.attr("model_id")("dma_transfer").cast<int>();
    ppu_control_engine_ = control;
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
  py::object ppu_engine() const { return ppu_engine_; }

private:
  py::object mapper_bank_engine_ = py::none();
  py::object controller_engine_ = py::none();
  py::object audio_engine_ = py::none();
  py::object sample_engine_ = py::none();
  py::object mapper_control_engine_ = py::none();
  py::object ppu_engine_ = py::none();
  py::object ppu_control_engine_ = py::none();
  unique_ptr<PythonSteppedRuntime> stepped_;
  unique_ptr<ConsoleMachine> machine_;
  unique_ptr<Console> console_;
};

} // namespace

#ifdef TRANSFORMER_NES_PPU_REFERENCE
PYBIND11_MODULE(nes_hardware_reference, m) {
#else
PYBIND11_MODULE(nes_hardware, m) {
#endif
  py::class_<Cartridge>(m, "Cartridge", py::module_local())
      .def_static("load", &Cartridge::load)
      .def_static("parse",
                  [](py::bytes b) {
                    string s = b;
                    return Cartridge::parse(
                        span(reinterpret_cast<const uint8_t*>(s.data()), s.size()));
                  })
      .def_readonly("mapper", &Cartridge::mapper)
      .def_readonly("battery", &Cartridge::battery);
  py::class_<ConsoleHost>(m, "Console", py::module_local())
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
      .def_property_readonly("ppu_engine", &ConsoleHost::ppu_engine)
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
      .def_property_readonly("halted", [](ConsoleHost& h) { return h.console().halted; })
      .def_property_readonly("last_audio", [](ConsoleHost& h) {
        const auto& a = h.console().last_audio;
        py::array_t<float> out(a.size());
        std::copy(a.begin(), a.end(), out.mutable_data());
        return out;
      });
}
