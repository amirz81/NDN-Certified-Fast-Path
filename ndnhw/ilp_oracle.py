from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .trie import PrefixTrie


@dataclass(frozen=True)
class OracleResult:
    selected: set[int]
    assignment: dict[int, int]
    value: float
    feasible: bool
    searched_subsets: int


def _bank_feasible_assignment(
    trie: PrefixTrie,
    selected: set[int],
    B: int,
    C: float,
) -> dict[int, int] | None:
    """Exact backtracking bank assignment for a fixed selected set.

    This is intended for small instances. Nodes are placed in descending load order.
    """
    nodes = sorted(selected, key=lambda v: trie.get(v).access_load, reverse=True)
    loads = [0.0 for _ in range(B)]
    assignment: dict[int, int] = {}

    def backtrack(i: int) -> bool:
        if i == len(nodes):
            return True

        v = nodes[i]
        load = trie.get(v).access_load

        tried_loads: set[float] = set()
        for b in sorted(range(B), key=lambda x: loads[x]):
            if loads[b] in tried_loads:
                continue
            tried_loads.add(loads[b])

            if loads[b] + load <= C:
                loads[b] += load
                assignment[v] = b
                if backtrack(i + 1):
                    return True
                assignment.pop(v, None)
                loads[b] -= load

        return False

    if backtrack(0):
        return dict(assignment)
    return None


def exact_bank_oracle(
    trie: PrefixTrie,
    M: int,
    D: int,
    B: int,
    C: float,
    max_nodes: int = 22,
) -> OracleResult:
    """Exact small-instance oracle for bank-aware prefix placement.

    It enumerates all subsets of eligible trie nodes, keeps only prefix-closed subsets,
    then checks exact bank feasibility by backtracking.

    This is not a scalable solver. It is deliberately an oracle for adversarial cases,
    toy experiments, and validating heuristics.
    """
    eligible = [
        node.node_id for node in trie.iter_nodes()
        if node.depth <= D
    ]

    if len(eligible) > max_nodes:
        raise ValueError(
            f"exact_bank_oracle is exponential: {len(eligible)} eligible nodes > max_nodes={max_nodes}. "
            "Use a smaller trace or increase max_nodes knowingly."
        )

    best_selected: set[int] = set()
    best_assignment: dict[int, int] = {}
    best_value = 0.0
    searched = 0

    for r in range(0, min(M, len(eligible)) + 1):
        for combo in combinations(eligible, r):
            searched += 1
            selected = set(combo)
            if not trie.is_prefix_closed(selected):
                continue

            value = sum(trie.get(v).fib_weight for v in selected if trie.get(v).is_fib_prefix)
            if value < best_value:
                continue

            assignment = _bank_feasible_assignment(trie, selected, B, C)
            if assignment is None:
                continue

            if value > best_value or (value == best_value and len(selected) < len(best_selected)):
                best_value = value
                best_selected = selected
                best_assignment = assignment

    return OracleResult(
        selected=best_selected,
        assignment=best_assignment,
        value=best_value,
        feasible=True,
        searched_subsets=searched,
    )
