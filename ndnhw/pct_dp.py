from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .trie import PrefixTrie


@dataclass(frozen=True)
class DPResult:
    selected: set[int]
    value: float
    budget_used: int


def pct_dp(trie: PrefixTrie, M: int, D: int, include_zero_weight_ancestors: bool = True) -> DPResult:
    """Exact DP for memory-only prefix-closed trie placement.

    Maximizes sum of selected FIB weights under:
    - prefix closure,
    - at most M selected nodes,
    - selected node depth <= D.

    Root is treated as free and not counted in M.

    Returns the best selected set excluding root.
    """

    if M <= 0 or D <= 0:
        return DPResult(selected=set(), value=0.0, budget_used=0)

    # dp[node_id] maps k -> (value, selected_set_excluding_root) for subtree when node is selected.
    memo: dict[int, dict[int, tuple[float, set[int]]]] = {}

    def solve(v: int) -> dict[int, tuple[float, set[int]]]:
        node = trie.get(v)

        if node.depth > D:
            return {}

        if v == trie.root_id:
            base_cost = 0
            base_value = 0.0
            base_selected: set[int] = set()
        else:
            base_cost = 1
            base_value = node.fib_weight if node.is_fib_prefix else 0.0
            base_selected = {v}

        if base_cost > M:
            return {}

        table: dict[int, tuple[float, set[int]]] = {base_cost: (base_value, set(base_selected))}

        for child_id in node.children.values():
            child_table = solve(child_id)
            if not child_table:
                continue

            new_table = dict(table)

            for k1, (val1, set1) in table.items():
                for k2, (val2, set2) in child_table.items():
                    kk = k1 + k2
                    if kk > M:
                        continue
                    vv = val1 + val2
                    if kk not in new_table or vv > new_table[kk][0]:
                        new_table[kk] = (vv, set1 | set2)

            table = new_table

        memo[v] = table
        return table

    root_table = solve(trie.root_id)
    if not root_table:
        return DPResult(selected=set(), value=0.0, budget_used=0)

    best_k, (best_val, best_set) = max(root_table.items(), key=lambda item: (item[1][0], -item[0]))
    return DPResult(selected=best_set, value=best_val, budget_used=best_k)
