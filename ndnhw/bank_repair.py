from __future__ import annotations

from dataclasses import dataclass

from .trie import PrefixTrie
from .banking import greedy_load_balance, evaluate_bank_assignment
from .bank_greedy import bank_greedy


@dataclass(frozen=True)
class RepairResult:
    selected: set[int]
    assignment: dict[int, int]
    value: float
    feasible: bool
    bank_loads: dict[int, float]
    rounds: int


def _value(trie: PrefixTrie, selected: set[int]) -> float:
    return sum(trie.get(v).fib_weight for v in selected if trie.get(v).is_fib_prefix)


def _leaf_nodes(trie: PrefixTrie, selected: set[int]) -> list[int]:
    leaves = []
    for v in selected:
        if not (set(trie.get(v).children.values()) & selected):
            leaves.append(v)
    return leaves


def _density(trie: PrefixTrie, node_id: int) -> float:
    node = trie.get(node_id)
    return (node.fib_weight if node.is_fib_prefix else 0.0) / max(node.access_load, 1.0)


def repair_bank_greedy(
    trie: PrefixTrie,
    M: int,
    D: int,
    B: int,
    C: float,
    max_rounds: int = 15,
) -> RepairResult:
    """A stronger feasible local-search heuristic.

    Starts from BankGreedy. Then repeatedly tries to add a candidate closure; if memory
    or bank constraints fail, it removes low-density selected leaves until feasible.
    """
    base = bank_greedy(trie, M=M, D=D, B=B, C=C)
    selected = set(base.selected)

    candidates = [
        node.node_id for node in trie.iter_nodes()
        if node.is_fib_prefix and node.depth <= D and node.fib_weight > 0
    ]

    rounds = 0
    for _ in range(max_rounds):
        rounds += 1
        current_value = _value(trie, selected)
        best_trial = selected
        best_value = current_value

        for cand in candidates:
            closure = trie.closure({cand}, include_root=False)
            closure = {u for u in closure if trie.get(u).depth <= D}
            trial = selected | closure

            # Memory repair: drop low-density leaves until the set fits.
            while len(trial) > M:
                leaves = _leaf_nodes(trie, trial)
                removable = [v for v in leaves if v not in closure]
                if not removable:
                    break
                drop = min(removable, key=lambda v: (_density(trie, v), trie.get(v).fib_weight))
                trial.remove(drop)

            if len(trial) > M or not trie.is_prefix_closed(trial):
                continue

            # Bank repair: if overloaded, drop low-density leaves not in new closure.
            assignment, report = greedy_load_balance(trie, trial, B=B, C=C)
            guard = 0
            while not report.feasible and guard < M:
                guard += 1
                leaves = _leaf_nodes(trie, trial)
                removable = [v for v in leaves if v not in closure]
                if not removable:
                    break
                drop = min(removable, key=lambda v: (_density(trie, v), trie.get(v).fib_weight))
                trial.remove(drop)
                if not trie.is_prefix_closed(trial):
                    break
                assignment, report = greedy_load_balance(trie, trial, B=B, C=C)

            if not report.feasible or not trie.is_prefix_closed(trial):
                continue

            val = _value(trie, trial)
            if val > best_value:
                best_value = val
                best_trial = set(trial)

        if best_value <= current_value:
            break
        selected = best_trial

    assignment, report = greedy_load_balance(trie, selected, B=B, C=C)
    return RepairResult(
        selected=selected,
        assignment=assignment,
        value=_value(trie, selected),
        feasible=report.feasible,
        bank_loads=report.bank_loads,
        rounds=rounds,
    )
