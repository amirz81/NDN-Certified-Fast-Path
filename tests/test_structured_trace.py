from ndnhw.structured_trace import StructuredTraceConfig, generate_structured_trace


def test_structured_trace_generation():
    trace, trie = generate_structured_trace(StructuredTraceConfig(n_interests=200, n_prefixes=32, seed=1))
    assert len(trace) == 200
    assert sum(1 for node in trie.nodes if node.is_fib_prefix) == 32
    assert len(trie) > 32
