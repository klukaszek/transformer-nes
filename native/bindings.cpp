#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "machine_abi.h"
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
#include <string>
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
// A mapper bank layout reads its register state, then the board's ROM
// sizes (nes/mappers.py BOARD; BankControl::board).
vector<string> with_board(vector<string> state) {
  state.insert(state.end(), {"prg_16k", "chr_4k"});
  return state;
}
// ppu2c02/render.py INPUTS, OUTPUTS, ppu2c02/sprites.py's tokens and
// ppu2c02/registers.py's.
const vector<string> render_inputs = fields(
    {"is_slot", "slot", "hit_set", "fine_x", "ctrl", "mask", {"nt_page", 4}, "latch_lo",
     "latch_hi", "wide_low", "ext_mode", "chr_upper", "v_cx", "v_cy", "v_nt", "v_fy", {"bg", 16},
     {"sprite", 8}, "chr_4k"});
const vector<string> render_outputs =
    fields({{"pixel", 8}, {"tile", 8}, "new_v_cx", "new_v_cy", "new_v_nt", "new_v_fy", "fetched",
            {"hit", 8}, "new_latch_lo", "new_latch_hi"});
const vector<string> evaluate_inputs = fields({"is_evaluate", "line", "ctrl", "start"});
const vector<string> evaluate_outputs = fields({"count", {"index", 8}});
const vector<string> sprite_inputs =
    fields({"is_fetch", "line", "ctrl", "present", "index", "latch_lo", "latch_hi", "wide_low",
            {"bank", 8}, "whole_chr"});
const vector<string> sprite_outputs =
    fields({{"pixel", 8}, "x", "fetch", "a12", "new_latch_lo", "new_latch_hi"});
const vector<string> register_inputs =
    fields({"is_access", "reg", "value", "is_write", "status", "rendering", {"nt_page", 4},
            "chr_writable", "latch_lo", "latch_hi", {"bank", 8}, "t_cx", "t_cy", "t_nt", "t_fy", "v_cx", "v_cy", "v_nt", "v_fy",
            "fine_x", "w", "ctrl", "mask", "oam_addr", "read_buffer", "whole_chr"});
const vector<string> register_outputs =
    fields({"new_t_cx", "new_t_cy", "new_t_nt", "new_t_fy", "new_v_cx", "new_v_cy", "new_v_nt",
            "new_v_fy", "new_fine_x", "new_w", "new_ctrl", "new_mask", "new_oam_addr",
            "new_read_buffer", "read_value", "mem_write", "mem_write_addr", "mem_write_value",
            "clear_vblank", "nmi_rise", "nmi_fall"});

// A compiled chip's (model/backend.py:CompiledChip) engine and raw step.
std::pair<void*, void (*)(void*, int, const int*, int*)> engine_of(const py::object& chip) {
  const auto abi = chip.attr("abi")().cast<py::tuple>();
  return {reinterpret_cast<void*>(abi[0].cast<std::uintptr_t>()),
          reinterpret_cast<void (*)(void*, int, const int*, int*)>(abi[2].cast<std::uintptr_t>())};
}
// Whether a chip has token type `name` with exactly these fields.
bool token_is(const py::object& chip, const char* name, const vector<string>& in,
              const vector<string>& out) {
  const py::dict graphs = chip.attr("graphs");
  return graphs.contains(name) && graphs[name].attr("inputs").cast<vector<string>>() == in &&
         graphs[name].attr("outputs").cast<vector<string>>() == out;
}
int model_id(const py::object& chip, const char* token) {
  return chip.attr("model_id")(token).cast<int>();
}
int model_id(const py::object& chip, const char* token, const string& specialization) {
  return chip.attr("model_id")(token, specialization).cast<int>();
}

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
  // `nes`: the compiled NES (nes/model.py:compiled()) whose chips this
  // console runs; None for the process's own.
  ConsoleHost(Cartridge cart, py::object factory, double sample_rate, py::object nes) {
    if (nes.is_none())
      nes = py::module_::import("transformer_nes.nes.model").attr("compiled")();
    nes_ = nes;
    const bool chr_writable = cart.chr.empty();
    const bool vertical = cart.mirroring == Mirroring::Vertical;
    auto mapper = make_mapper(std::move(cart));
    // Mapper constructors retain geometry; the PPU owns the CHR payload.
    Ppu ppu(mapper->take_chr(), chr_writable, vertical);
    // The mapper chip's circuits (nes/mappers.py:chip): register transitions,
    // bank layout, bus access and address projection, and its IRQ counter.
    if (mapper->bank_state() || mapper->irq_state()) {
      const int number = mapper->bank_state() ? mapper->bank_state()->chip
                         : mapper->irq_state()->chip == MapperIrq::Mmc3 ? BankControl::Mmc3
                                                                        : BankControl::Mmc5;
      mapper_engine_ = nes.attr("mapper")(number);
    }
    if (auto* bank = mapper->bank_state()) {
      const py::object transition = mapper_engine_.attr("graphs")["transition"];
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
      const auto access_inputs = fields({"addr_lo", "addr_hi", "is_write", "original",
                                         {"bank", 4}, {"ram", 4}, "ram_bank", "writable", "mul_lo", "mul_hi"});
      if (vector<string>(inputs.end() - 4, inputs.end()) != fields({"event", "port", "value", "original"}) ||
          !token_is(mapper_engine_, "transition", inputs, outputs) ||
          !token_is(mapper_engine_, "layout", with_board(state), maps) ||
          !token_is(mapper_engine_, "access", access_inputs,
                    fields({"port", "bank_event", "irq_event", "schedule_changed", "read_source",
                            "read_index", "read_value", "write_source", "repair"})) ||
          !token_is(mapper_engine_, "address", access_inputs,
                    fields({"read_source", "read_index", "read_value"})))
        throw std::invalid_argument("engine is not the mapper chip");
      std::tie(bank->engine.ctx, bank->engine.step) = engine_of(mapper_engine_);
      std::tie(bank->engine.access_ctx, bank->engine.access_step) = engine_of(mapper_engine_);
      const auto model_id = [&](const char* token, py::object spec) {
        return mapper_engine_.attr("model_id")(token, spec).cast<int>();
      };
      bank->engine.transition = model_id("transition", py::none());
      bank->engine.layout = model_id("layout", py::none());
      bank->engine.access_models = {model_id("access", py::str("read")), model_id("access", py::str("write"))};
      bank->engine.address_model = model_id("address", py::none());
      bank->reset();
    }
    if (auto* irq = mapper->irq_state()) {
      if (!token_is(mapper_engine_, "irq",
                    fields({"counter", "target", "phase", "pending", "enabled", "event", "value"}),
                    fields({"new_counter", "new_target", "new_phase", "new_pending", "new_enabled",
                            "irq", "read_value", "predictable", "next_edge", "irq_write"})))
        throw std::invalid_argument("engine is not the mapper IRQ graphs");
      std::tie(irq->engine.ctx, irq->engine.step) = engine_of(mapper_engine_);
      irq->engine.model = mapper_engine_.attr("model_id")("irq").cast<int>();
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
    // The APU's chips (apu2a03/audio.py): the digital chip, then the mixer.
    audio_engine_ = nes.attr("chip")("apu");
    sample_engine_ = nes.attr("chip")("apu_mixer");
    const py::dict audio_graphs = audio_engine_.attr("graphs");
    const auto size_is = [&](const char* name, size_t ni, size_t no) {
      return py::len(audio_graphs[name].attr("inputs")) == ni && py::len(audio_graphs[name].attr("outputs")) == no;
    };
    if (!size_is("control", Apu::channel_input_count, Apu::channel_output_count) ||
        !size_is("frame", 6, 8) || !size_is("program", 59, 57) || !size_is("dmc", 17, 20) ||
        !size_is("oscillate", 15, 9) || !size_is("tick", 15, 9) || !size_is("noise", 17, 15))
      throw std::invalid_argument("engine is not the APU chip");
    const py::dict mixer_graphs = sample_engine_.attr("graphs");
    if (py::len(mixer_graphs["sample"].attr("inputs")) != Apu::sample_input_count ||
        py::len(mixer_graphs["sample"].attr("outputs")) != 3 ||
        py::len(mixer_graphs["gains"].attr("inputs")) != Apu::channel_input_count - 1 ||
        py::len(mixer_graphs["gains"].attr("outputs")) != Apu::sample_parameter_count ||
        sample_engine_.attr("chip").attr("f_bits").cast<int>() != 0)
      throw std::invalid_argument("engine is not the APU mixer chip");
    Apu::Engine audio_api;
    std::tie(audio_api.ctx, audio_api.step) = engine_of(audio_engine_);
    const auto audio_model = [&](const char* token, py::object spec) {
      return audio_engine_.attr("model_id")(token, spec).cast<int>();
    };
    audio_api.control = audio_model("control", py::none());
    audio_api.frame = audio_model("frame", py::none());
    audio_api.program = audio_model("program", py::none());
    audio_api.dmc = {audio_model("dmc", py::none()), audio_model("dmc", py::str("fetch")),
                     audio_model("dmc", py::str("clock"))};
    audio_api.oscillators = {audio_model("oscillate", py::none()), audio_model("tick", py::none())};
    for (size_t tick = 0; tick < 2; ++tick)
      for (size_t mode = 0; mode < 2; ++mode)
        audio_api.noise[tick][mode] =
            audio_model("noise", py::str(string(tick ? "tick_mode" : "mode") + std::to_string(mode)));
    audio_api.span = audio_engine_.attr("chip").attr("constants")["span"].cast<int>();
    const auto sample_abi = sample_engine_.attr("abi")().cast<py::tuple>();
    audio_api.sample_ctx = reinterpret_cast<void*>(sample_abi[0].cast<std::uintptr_t>());
    audio_api.sample_step = reinterpret_cast<void (*)(void*, int, const float*, float*)>(
        sample_abi[2].cast<std::uintptr_t>());
    audio_api.sample_model = sample_engine_.attr("model_id")("sample").cast<int>();
    audio_api.parameter_model = sample_engine_.attr("model_id")("gains").cast<int>();
    console_->apu.set_engine(audio_api);
    controller_engine_ = nes.attr("chip")("controller");
    if (!token_is(controller_engine_, "controller",
                  fields({"strobe", "index", {"button", 8}, "event", "value", "count"}),
                  fields({"new_strobe", "new_index", "reply", "length", {"byte", 8}})))
      throw std::invalid_argument("engine is not the controller chip");
    auto& control = console_->controller.engine;
    std::tie(control.ctx, control.step) = engine_of(controller_engine_);
    control.model = controller_engine_.attr("model_id")("controller").cast<int>();
#ifndef TRANSFORMER_NES_PPU_REFERENCE
    set_ppu_engine(py::module_::import("transformer_nes.ppu2c02.machine")
                       .attr("PpuEngine")(nes, console_->ppu.whole_chr()));
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
  // Run the PPU as the model's compiled ppu and ppu_control chips
  // (ppu2c02/machine.py:PpuEngine), kept alive here. Their token types must
  // be the PPU's, checked field by field.
  void set_ppu_engine(py::object engine) {
    if (engine.is_none()) {
#ifdef TRANSFORMER_NES_PPU_REFERENCE
      console_->ppu.set_engine({});
      ppu_engine_ = py::none();
      return;
#else
      throw std::invalid_argument("PPU transformer engine is required");
#endif
    }
    const py::object ppu = engine.attr("ppu"), control = engine.attr("control");
    if (!token_is(ppu, "render", render_inputs, render_outputs) ||
        !token_is(ppu, "render_status", render_inputs,
                  vector<string>(render_outputs.begin() + 8, render_outputs.end())) ||
        !token_is(ppu, "evaluate", evaluate_inputs, evaluate_outputs) ||
        !token_is(ppu, "sprite", sprite_inputs, sprite_outputs) ||
        !token_is(ppu, "sprite_bus", sprite_inputs,
                  vector<string>(sprite_outputs.begin() + 9, sprite_outputs.end())) ||
        !token_is(ppu, "overflow", {"is_overflow", "line", "ctrl", "start"}, {"index"}) ||
        !token_is(ppu, "access", register_inputs, register_outputs))
      throw std::invalid_argument("engine is not the PPU chip");
    const auto abi = ppu.attr("abi")().cast<py::tuple>();
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
      out.general = static_cast<uint8_t>(d["general"].cast<int>());
    }
    const py::dict sprites = engine.attr("sprite_dispatch");
    api.sprites.evaluate = static_cast<uint8_t>(sprites["evaluate"].cast<int>());
    api.sprites.overflow = static_cast<uint8_t>(sprites["overflow"].cast<int>());
    for (const auto& [name, table] : {std::pair{"fetch", &api.sprites.fetch},
                                      std::pair{"fetch_bus", &api.sprites.fetch_bus}}) {
      const auto ids = sprites[name].cast<vector<vector<int>>>();
      for (size_t latched = 0; latched < 2; ++latched)
        std::ranges::copy(ids[latched], (*table)[latched].begin());
    }
    api.sprites.absent = static_cast<uint8_t>(sprites["absent"].cast<int>());
    api.render_status = engine.attr("status_render_dispatch").cast<vector<int>>();
    const auto registers = engine.attr("register_dispatch").cast<vector<vector<int>>>();
    for (size_t w = 0; w < 2; ++w)
      std::ranges::copy(registers[w], api.registers[w].begin());

    const py::object module = py::module_::import("transformer_nes.ppu2c02.control");
    const auto io = [&](const char* name) { return module.attr(name).cast<vector<string>>(); };
    if (!token_is(control, "control",
                  fields({"event", "mask", "ctrl", "odd", "phase", "flag_age", "vblank",
                          "suppressed", "nmi_edge", "nmi_delayed", "t_cx", "t_cy", "t_nt", "t_fy",
                          "v_cx", "v_cy", "v_nt", "v_fy", "oam_addr", "clear_vblank",
                          "nmi_rise", "nmi_fall", "status_pending"}),
                  fields({"new_v_cx", "new_v_cy", "new_v_nt", "new_v_fy", "new_oam_addr",
                          "new_vblank", "new_suppressed", "new_nmi_edge", "new_nmi_delayed",
                          "set_stamp", "clear_stamp", "clear_status", "status_changed",
                          "skip_dot", "rendering"})) ||
        !token_is(control, "clock", fields({"line", "index"}),
                  fields({"next_line", "wrap", "table", "count", "dot", "event", "slot", "run"})))
      throw std::invalid_argument("engine is not the PPU control chip");
    for (const auto& [name, input, output, ni, no] :
         {std::tuple{"bus", "BUS_INPUTS", "BUS_OUTPUTS", 6, 4},
          {"background_span", "SPAN_INPUTS", "SPAN_OUTPUTS", 5, 6},
          {"access_bus", "ACCESS_INPUTS", "ACCESS_OUTPUTS", 17, 7},
          {"access_delay", "ACCESS_INPUTS", "DELAY_OUTPUTS", 17, 1},
          {"view", "VIEW_INPUTS", "VIEW_OUTPUTS", 4, 6},
          {"compose", "COMPOSE_INPUTS", "COMPOSE_OUTPUTS", 19, 10},
          {"dma_request", "DMA_INPUTS", "DMA_REQUEST", 4, 2},
          {"dma_transfer", "DMA_INPUTS", "DMA_TRANSFER", 4, 5}})
      if (!token_is(control, name, io(input), io(output)) ||
          io(input).size() != static_cast<size_t>(ni) || io(output).size() != static_cast<size_t>(no))
        throw std::invalid_argument("engine is not the PPU control chip");
    std::tie(api.control_ctx, api.control_step) = engine_of(control);
    api.control_model = model_id(control, "control");
    const array<const char*, 3> bus_events{"pin", "background", "sprite"};
    for (size_t i = 0; i < bus_events.size(); ++i)
      api.bus_models[i] = model_id(control, "bus", bus_events[i]);
    api.background_span_model = model_id(control, "background_span");
    api.clock_model = model_id(control, "clock");
    api.access_bus_model = model_id(control, "access_bus");
    api.access_delay_model = model_id(control, "access_delay");
    api.view_model = model_id(control, "view");
    api.compose_model = model_id(control, "compose");
    api.dma_request_model = model_id(control, "dma_request");
    api.dma_transfer_model = model_id(control, "dma_transfer");
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
    out["frame_seconds"] = result.frame_seconds;
    out["frame_cpu_seconds"] = result.frame_cpu_seconds;
    return out;
  }
  Console& console() { return *console_; }
  py::object ppu_engine() const { return ppu_engine_; }

private:
  py::object nes_ = py::none();
  py::object mapper_engine_ = py::none();
  py::object controller_engine_ = py::none();
  py::object audio_engine_ = py::none();
  py::object sample_engine_ = py::none();
  py::object ppu_engine_ = py::none();
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
      .def(py::init<Cartridge, py::object, double, py::object>(), py::arg("cartridge"),
           py::arg("backend_factory"), py::arg("sample_rate") = 44100.0, py::arg("nes") = py::none())
      .def("run_frame", &ConsoleHost::run_frame)
      .def("audio", [](ConsoleHost& h) {
             const auto& samples = h.console().last_audio;
             py::array_t<float> out(samples.size());
             std::copy(samples.begin(), samples.end(), out.mutable_data());
             return out;
           }, "The samples the last frame produced (float32, at the console's sample rate)")
      .def("set_buttons", &ConsoleHost::set_buttons)
      .def("run_tape", &ConsoleHost::run_tape, py::arg("path"), py::arg("tail_frames") = 120,
           py::arg("max_frames") = 0)
      .def("read_ram", &ConsoleHost::read_ram, py::arg("start"), py::arg("count"))
      .def("prg_ram", &ConsoleHost::prg_ram)
      .def("load_prg_ram", &ConsoleHost::load_prg_ram, py::arg("data"))
      .def("set_sprite_limit", &ConsoleHost::set_sprite_limit, py::arg("enabled"))
      .def("ppu_expert_map", &ConsoleHost::ppu_expert_map,
           "Which render model drew each 8-pixel slot of the last frame "
           "(240x32 bytes: 0 full graph, k expert k, 255 no output yet)")
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
      .def_property_readonly("ppu_whole_chr",
                             [](ConsoleHost& h) { return h.console().ppu.whole_chr(); })
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
