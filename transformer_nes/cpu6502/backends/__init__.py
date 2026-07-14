"""The 6502 machine backends, one semantics each way.

reference is the oracle; graph is the exact evaluator (and the runtime the
generated engine compiles from); weights/ is the machine as actual
transformer weights, with its run-loop protocol in weights/run_protocol.py.
"""
