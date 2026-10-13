from __future__ import annotations

from ndnhw.adversarial import unsafe_shortcut_case, deep_prefix_cost_case, bank_fits_memory_fails_case
from ndnhw.metrics import evaluate_coverage
from ndnhw.pct_dp import pct_dp
from ndnhw.baselines import topk_closure
from ndnhw.banking import hashbank, greedy_load_balance
from ndnhw.ilp_oracle import exact_bank_oracle


def main() -> None:
    print("\n=== Unsafe shortcut case ===")
    trace, trie, selected = unsafe_shortcut_case()
    report = evaluate_coverage(trie, trace, selected, D=10)
    print(report.as_dict())

    print("\n=== Deep prefix hidden ancestor cost ===")
    trace, trie = deep_prefix_cost_case()
    for M in [3, 4, 5, 6]:
        dp = pct_dp(trie, M=M, D=10)
        topk = topk_closure(trie, M=M, D=10)
        print({
            "M": M,
            "PCT-DP": {"value": dp.value, "selected": [str(trie.get(v).name) for v in sorted(dp.selected)]},
            "TopK": {"value": topk.value, "selected": [str(trie.get(v).name) for v in sorted(topk.selected)]},
        })

    print("\n=== Memory fits, banks can fail ===")
    trace, trie = bank_fits_memory_fails_case()
    dp = pct_dp(trie, M=6, D=10)
    hash_assignment, hash_report = hashbank(trie, dp.selected, B=2, C=150)
    lb_assignment, lb_report = greedy_load_balance(trie, dp.selected, B=2, C=150)
    oracle = exact_bank_oracle(trie, M=6, D=10, B=2, C=150, max_nodes=12)
    print("PCT-DP selected:", [str(trie.get(v).name) for v in sorted(dp.selected)], "value:", dp.value)
    print("HashBank:", hash_report.as_dict())
    print("GreedyLoadBalance:", lb_report.as_dict())
    print("Oracle selected:", [str(trie.get(v).name) for v in sorted(oracle.selected)], "value:", oracle.value)


if __name__ == "__main__":
    main()
