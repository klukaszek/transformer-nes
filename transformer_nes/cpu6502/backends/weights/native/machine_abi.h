#pragma once

// Shared by the generated CPU engine and native NES console. The function
// table and run receipt cross a module boundary; change both with this version.
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

