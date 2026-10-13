from experiments.run_v04_bpc_lite import build_certificate_stress_trace
from ndnhw.cert_bank_portfolio import cert_bank_portfolio


def test_cert_bank_portfolio_feasible():
    trace, trie, action_ids = build_certificate_stress_trace(seed=1)
    total_load = sum(n.access_load for n in trie.nodes)
    res = cert_bank_portfolio(
        trie, trace, total_budget=24, D=8, B=2, C=total_load * 0.35,
        action_ids=action_ids, cert_budget_options=[0, 4, 8],
    )
    assert res.feasible
    assert 0.0 <= res.certified_coverage <= 1.0
    assert len(res.selected) <= res.prefix_budget
