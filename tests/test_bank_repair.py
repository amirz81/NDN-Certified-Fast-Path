from ndnhw.adversarial import bank_fits_memory_fails_case
from ndnhw.bank_repair import repair_bank_greedy


def test_bank_repair_feasible():
    trace, trie = bank_fits_memory_fails_case()
    res = repair_bank_greedy(trie, M=6, D=10, B=2, C=150)
    assert res.feasible
    assert res.value > 0
