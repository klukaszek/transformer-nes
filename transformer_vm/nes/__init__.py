"""NES system layer: everything console-shaped lives here.

The pure 6502 implementation (CPU cores, graph, kernel, conformance
suites) stays in transformer_vm.cpu6502 and knows nothing about the NES.
This package wires devices to the CPU's machine interfaces over the MMIO
bus; the PPU, cartridge/mappers, and the frame loop land here.
"""
