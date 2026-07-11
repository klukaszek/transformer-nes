"""The 6502 machine backends, one semantics each way.

reference is the oracle; evaluator is the exact graph (and the runtime the
generated backends compile from); kernel is the graph as C++; weights_runner
is the machine as actual transformer weights.  run_protocol, experts, and
windows serve the two generated backends.
"""
