// The contract between a compiled model's chip kernels and the runtime.
//
// The backend (model/backend.py) compiles one library per model: the fixed
// runtime (runtime.cpp) and, per chip, its scalar kernels and one Bank
// descriptor. A generated table, nes_chips, names them. This header is shared
// by both sides and must stay self-contained.
#pragma once
#include <cstdint>

#define TRANSFORMER_NES_BANK_ABI 3

namespace transformer_nes_bank {

enum ScalarKind : int32_t { Int32 = 0, Float32 = 1 };

// One compiled circuit.  `run` is a
//   void(const Scalar* x, const uint8_t* memory, Scalar* out, int32_t* reads)
// for the chip's scalar type.
struct Circuit {
    const void* run;
    int32_t width;  // outputs written
    int32_t aux;    // memory-read addresses reported (X-ray)
    int32_t f_bits; // fixed-point fraction bits of an integer circuit's readouts
};

// A token's caller-facing input and output counts, one per circuit.
struct Token {
    int32_t n_in, n_out;
};

// A chip: its circuits in model-id order and the engine parameters.
struct Bank {
    int32_t abi;          // TRANSFORMER_NES_BANK_ABI
    int32_t scalar;       // ScalarKind
    int32_t n_circuits;
    const Circuit* circuits;
    const Token* tokens;  // n_circuits entries
    int32_t memory;       // bytes of chip memory the kernels address
    int32_t trace_rows;   // residual rows of model 0's trace (layers + 1)
    int32_t primary_width;  // model 0's residual width
};

struct ChipEntry {
    const char* name;
    const Bank* bank;
};

}  // namespace transformer_nes_bank

// Defined by the backend's generated chip table.
extern const transformer_nes_bank::ChipEntry nes_chips[];
extern const int nes_chip_count;
