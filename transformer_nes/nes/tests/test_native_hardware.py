"""The native console's C++ unit tests (tests/native_hardware_test.cpp),
built and run through the Makefile."""

from transformer_nes.nes.build import _make


def test_native_hardware():
    _make("test-native")  # raises on a failing assertion's nonzero exit
