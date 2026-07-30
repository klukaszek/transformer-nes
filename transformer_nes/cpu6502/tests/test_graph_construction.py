"""Contract for the 6502 graph construction boundary."""


def test_graph_builder_exposes_codegen_inputs():
    from transformer_nes.cpu6502.construction import Cpu6502Graph

    graph = Cpu6502Graph()

    assert graph.input_tokens
    assert graph.output_tokens
    assert graph.start_pc_lo.name == "start_pc_lo"
    assert graph.start_pc_hi.name == "start_pc_hi"
