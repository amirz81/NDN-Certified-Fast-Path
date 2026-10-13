from ndnhw.adversarial import deep_prefix_cost_case, unsafe_shortcut_case
from ndnhw.pct_dp import pct_dp
from ndnhw.metrics import evaluate_coverage


def test_pct_dp_deep_cost_prefers_shallow_when_budget_small():
    trace, trie = deep_prefix_cost_case()
    res = pct_dp(trie, M=3, D=10)
    names = {str(trie.get(v).name) for v in res.selected}
    assert "/x" in names or "/y" in names or "/z" in names
    assert "/a/b/c/d/e/f" not in names


def test_unsafe_shortcut_has_naive_but_not_certified_coverage():
    trace, trie, selected = unsafe_shortcut_case()
    report = evaluate_coverage(trie, trace, selected, D=10)
    assert report.naive_coverage > report.certified_coverage
    assert report.unsafe_rate > 0
