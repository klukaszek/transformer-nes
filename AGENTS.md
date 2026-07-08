# Agent Notes

This repo keeps several CPU and transformer backends on purpose.  Do not
delete an older path just because the generated weights engine is faster:
the slower paths are the verification ladder and diagnostics.

## Current Production Path

- NES gameplay: `Console(..., backend="weights")` builds
  `Cpu6502TransformerMachine` with `matmul="codegen"`, exact `int32`
  arithmetic, opcode-routed experts, and the native C++ run loop.  When the
  run loop is available, `KernelBus` drives the weights engine directly and
  Python only services MMIO pauses.
- CPU weights CLI: `6502-nestest --runner weights` and
  `6502-klaus --runner weights` default to `--matmul codegen`.  The direct
  `6502-transformer` smoke test should also default to `codegen`.
- The current single-game latency target is CPU throughput.  SMB gameplay is
  around 305K instr/s / 28 fps after routed experts; real-time needs roughly
  592K instr/s.

## Verification Ladder

- `reference.py`: oracle, fastest plain emulator, not graph semantics.
- `evaluator.py`: exact Python graph evaluator, slow but debuggable.
- `kernel.py`: generated C++ graph kernel, production oracle for NES speed.
- `weights_runner.py`: real analytic transformer weights.  Its default
  `codegen` path is the flagship runtime.
- `model/engine.py` and `model/weights_engine.cpp`: generic CSR C++ weights
  engine.  Keep it as a cross-check for generated code.
- `matmul="sparse"` and `matmul="dense"` are diagnostics; `sparse` is still
  needed for the true-softmax `cache="standard"` experiment.

## Common Pitfalls

- The roadmap is a chronological lab notebook; later entries supersede earlier
  performance claims.  Check the bottom before trusting a number near the top.
- `plan.yaml` and `allocation.yaml` are generated in the working directory and
  gitignored.  Avoid treating them as source.
- The generated engine cache is model-digest keyed.  If benchmarking variants,
  make sure the generated C++ struct/module names differ so two extensions do
  not cross-resolve symbols.
- `checkpoint_tokens` remains supported, but the default generated engine's
  state no longer grows with the token log.
- Do not route NES/device logic into `cpu6502/`; dependency direction is
  `nes/` -> `cpu6502/` -> `graph/` + `attention/`.

## Useful Commands

```bash
uv run pytest -m "not slow"
uv run 6502-nestest --runner weights
uv run 6502-klaus --runner weights --graph-steps 200000
uv run 6502-transformer --steps 30
uv run 6502-transformer --steps 30 --cache standard --matmul sparse
```
