from __future__ import annotations

from ndnhw.adversarial import bank_fits_memory_fails_case
from ndnhw.ilp_oracle import exact_bank_oracle
from ndnhw.bank_greedy import bank_greedy, bank_greedy_ls


def main() -> None:
    trace, trie = bank_fits_memory_fails_case()

    M, D, B, C = 6, 10, 2, 150

    oracle = exact_bank_oracle(trie, M=M, D=D, B=B, C=C, max_nodes=12)
    greedy = bank_greedy(trie, M=M, D=D, B=B, C=C)
    greedy_ls = bank_greedy_ls(trie, M=M, D=D, B=B, C=C, max_rounds=3)

    def names(selected):
        return [str(trie.get(v).name) for v in sorted(selected)]

    print("Oracle:", oracle.value, names(oracle.selected), oracle.assignment, "searched:", oracle.searched_subsets)
    print("BankGreedy:", greedy.value, names(greedy.selected), greedy.assignment)
    print("BankGreedy-LS:", greedy_ls.value, names(greedy_ls.selected), greedy_ls.assignment)


if __name__ == "__main__":
    main()
