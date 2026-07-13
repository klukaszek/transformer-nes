"""The 6502 machine backends, one semantics each way.

reference is the oracle; evaluator is the exact graph (and the runtime the
generated backend compiles from); weights/ is
the machine as actual transformer weights.  run_protocol is the native
run-loop protocol shared by the two generated backends.
"""
