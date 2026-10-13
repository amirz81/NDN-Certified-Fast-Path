from ndnhw.adversarial import bank_fits_memory_fails_case
from ndnhw.ilp_oracle import exact_bank_oracle
from ndnhw.bank_greedy import bank_greedy, bank_greedy_ls
from ndnhw.banking import evaluate_bank_assignment


def test_exact_bank_oracle_feasible():
    trace, trie = bank_fits_memory_fails_case()
    oracle = exact_bank_oracle(trie, M=6, D=10, B=2, C=150, max_nodes=12)
    report = evaluate_bank_assignment(trie, oracle.assignment, B=2, C=150)
    assert report.feasible
    assert oracle.value > 0


def test_bank_greedy_ls_is_feasible():
    trace, trie = bank_fits_memory_fails_case()
    res = bank_greedy_ls(trie, M=6, D=10, B=2, C=150)
    assert res.feasible
    assert res.value >= bank_greedy(trie, M=6, D=10, B=2, C=150).value
