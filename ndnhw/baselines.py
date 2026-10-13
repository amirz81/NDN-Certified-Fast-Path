from __future__ import annotations

from dataclasses import dataclass

from .trie import PrefixTrie
from .pct_dp import pct_dp
from .banking import greedy_load_balance, BankReport, stable_hash_bank


@dataclass(frozen=True)
class PlacementResult:
    selected: set[int]
    value: float
    budget_used: int


@dataclass(frozen=True)
class BankedPlacementResult:
    selected: set[int]
    assignment: dict[int, int]
    value: float
    bank_report: BankReport


def topk_closure(trie: PrefixTrie, M: int, D: int) -> PlacementResult:
    """Naive baseline: choose popular FIB prefixes, then take prefix closure."""
    selected: set[int] = set()
    candidates = [
        node for node in trie.iter_nodes()
        if node.is_fib_prefix and node.depth <= D and node.fib_weight > 0
    ]
    candidates.sort(key=lambda n: n.fib_weight, reverse=True)

    for node in candidates:
        closure = trie.closure({node.node_id}, include_root=False)
        closure = {v for v in closure if trie.get(v).depth <= D}
        new_selected = selected | closure
        if len(new_selected) <= M:
            selected = new_selected

    value = sum(trie.get(v).fib_weight for v in selected if trie.get(v).is_fib_prefix)
    return PlacementResult(selected=selected, value=value, budget_used=len(selected))


def hash_bank_assignment(trie: PrefixTrie, selected: set[int], B: int) -> dict[int, int]:
    return {v: stable_hash_bank(str(trie.get(v).name), B) for v in selected}


def coverage_first(trie: PrefixTrie, M: int, D: int, B: int, C: float) -> BankedPlacementResult:
    """Strong baseline: optimize coverage by PCT-DP first, assign banks second."""
    dp = pct_dp(trie, M=M, D=D)
    assignment, report = greedy_load_balance(trie, dp.selected, B=B, C=C)
    return BankedPlacementResult(
        selected=dp.selected,
        assignment=assignment,
        value=dp.value,
        bank_report=report,
    )
