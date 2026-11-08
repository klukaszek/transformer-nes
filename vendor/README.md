# Native dependencies

Run `./vendor/install_deps.sh` from the repository root to fetch the pinned
revisions listed in that script:

- `sokol/`: window, graphics, input, audio, and ImGui integration headers.
- `imgui/`: debug overlay headers and its four compiled implementation files.
- `pybind11/`: headers and license used by the native NES extensions.

The root Makefile builds the console and frontend into `bin/`. Sokol's
implementation lives in `nes/sokol_impl.m`, compiled as Objective-C on macOS
and C elsewhere. Generated transformer extensions use PyTorch's bundled
pybind11 headers and the separate cache in `.cache/engine/`.
