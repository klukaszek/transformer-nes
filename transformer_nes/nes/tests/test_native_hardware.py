"""The native console's C++ unit tests (tests/native_hardware_test.cpp),
built and run through the Makefile."""

from transformer_nes.build import make


def test_native_hardware():
    make("test-native")  # raises on a failing assertion's nonzero exit
