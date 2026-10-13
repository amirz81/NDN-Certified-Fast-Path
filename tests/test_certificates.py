from ndnhw.adversarial import unsafe_shortcut_case
from ndnhw.certificates import deterministic_action_ids, evaluate_certificate_aware_coverage


def test_certificate_budget_never_reduces_coverage():
    trace, trie, selected = unsafe_shortcut_case()
    action_ids = deterministic_action_ids(trie, mode="top_component")
    r0 = evaluate_certificate_aware_coverage(trie, trace, selected, D=10, action_ids=action_ids, max_certificates=0)
    r1 = evaluate_certificate_aware_coverage(trie, trace, selected, D=10, action_ids=action_ids, max_certificates=10)
    assert r1.certified_coverage >= r0.certified_coverage
