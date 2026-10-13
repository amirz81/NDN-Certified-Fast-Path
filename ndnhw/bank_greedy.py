from __future__ import annotations

from dataclasses import dataclass

from .trie import PrefixTrie
from .banking import evaluate_bank_assignment, greedy_load_balance


@dataclass(frozen=True)
class BankGreedyResult:
    selected: set[int]
    assignment: dict[int, int]
    value: float
    bank_loads: dict[int, float]
    feasible: bool


def _value(trie: PrefixTrie, selected: set[int]) -> float:
    return sum(trie.get(v).fib_weight for v in selected if trie.get(v).is_fib_prefix)


def _least_loaded_feasible_assignment(
    trie: PrefixTrie,
    selected: set[int],
    B: int,
    C: float,
) -> tuple[dict[int, int], dict[int, float]] | None:
    assignment: dict[int, int] = {}
    loads = {b: 0.0 for b in range(B)}

    for node_id in sorted(selected, key=lambda v: trie.get(v).access_load, reverse=True):
        load = trie.get(node_id).access_load
        placed = False
        for b in sorted(range(B), key=lambda bb: loads[bb]):
            if loads[b] + load <= C:
                assignment[node_id] = b
                loads[b] += load
                placed = True
                break
        if not placed:
            return None

    return assignment, loads


def bank_greedy(
    trie: PrefixTrie,
    M: int,
    D: int,
    B: int,
    C: float,
    lambda_bank: float = 1.0,
) -> BankGreedyResult:
    """Bank-aware greedy prefix placement.

    Candidate score:
        marginal FIB coverage / (missing ancestor cost + lambda * bank penalty)
    """
    selected: set[int] = set()
    assignment: dict[int, int] = {}

    candidates = {
        node.node_id for node in trie.iter_nodes()
        if node.is_fib_prefix and node.depth <= D and node.fib_weight > 0
    }

    while candidates:
        best = None

        for v in list(candidates):
            closure = trie.closure({v}, include_root=False)
            closure = {u for u in closure if trie.get(u).depth <= D}
            missing = closure - selected

            if not missing:
                candidates.discard(v)
                continue

            if len(selected) + len(missing) > M:
                candidates.discard(v)
                continue

            trial = selected | missing
            assigned = _least_loaded_feasible_assignment(trie, trial, B, C)
            if assigned is None:
                candidates.discard(v)
                continue

            gain = _value(trie, trial) - _value(trie, selected)
            if gain <= 0:
                candidates.discard(v)
                continue

            _, trial_loads = assigned
            bank_penalty = max(trial_loads.values()) - (sum(trial_loads.values()) / B)
            score = gain / (len(missing) + lambda_bank * bank_penalty / max(C, 1e-9) + 1e-9)

            if best is None or score > best[0]:
                best = (score, v, trial, assigned)

        if best is None:
            break

        old_selected = set(selected)
        _, chosen, selected, assigned = best
        assignment, _ = assigned
        candidates.discard(chosen)

        if selected == old_selected:
            break

    report = evaluate_bank_assignment(trie, assignment, B, C)
    return BankGreedyResult(
        selected=selected,
        assignment=assignment,
        value=_value(trie, selected),
        bank_loads=report.bank_loads,
        feasible=report.feasible,
    )


def _selected_leaf_nodes(trie: PrefixTrie, selected: set[int]) -> list[int]:
    leaves = []
    for v in selected:
        child_ids = set(trie.get(v).children.values())
        if not (child_ids & selected):
            leaves.append(v)
    return leaves


def bank_greedy_ls(
    trie: PrefixTrie,
    M: int,
    D: int,
    B: int,
    C: float,
    max_rounds: int = 10,
) -> BankGreedyResult:
    """BankGreedy followed by conservative 1-for-closure local search."""
    current = bank_greedy(trie, M=M, D=D, B=B, C=C)
    selected = set(current.selected)

    candidates = [
        node.node_id for node in trie.iter_nodes()
        if node.is_fib_prefix and node.depth <= D and node.fib_weight > 0
    ]

    for _ in range(max_rounds):
        improved = False
        best_selected = set(selected)
        best_value = _value(trie, selected)

        removable = _selected_leaf_nodes(trie, selected)
        for r in removable:
            base = set(selected)
            base.remove(r)

            if not trie.is_prefix_closed(base):
                continue

            for cand in candidates:
                closure = trie.closure({cand}, include_root=False)
                closure = {u for u in closure if trie.get(u).depth <= D}
                trial = base | closure

                if len(trial) > M or not trie.is_prefix_closed(trial):
                    continue

                assigned = _least_loaded_feasible_assignment(trie, trial, B, C)
                if assigned is None:
                    continue

                val = _value(trie, trial)
                if val > best_value:
                    best_value = val
                    best_selected = trial
                    improved = True

        selected = best_selected
        if not improved:
            break

    assignment, report = greedy_load_balance(trie, selected, B=B, C=C)
    return BankGreedyResult(
        selected=selected,
        assignment=assignment,
        value=_value(trie, selected),
        bank_loads=report.bank_loads,
        feasible=report.feasible,
    )
