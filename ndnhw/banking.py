from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class BankReport:
    feasible: bool
    bank_loads: dict[int, float]
    overloaded_banks: list[int]
    max_load: float
    total_overload: float

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "bank_loads": self.bank_loads,
            "overloaded_banks": self.overloaded_banks,
            "max_load": self.max_load,
            "total_overload": self.total_overload,
        }


def stable_hash_bank(name: str, B: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return int(digest, 16) % B


def evaluate_bank_assignment(
    trie,
    assignment: dict[int, int],
    B: int,
    C: float,
) -> BankReport:
    loads = {b: 0.0 for b in range(B)}
    for node_id, bank in assignment.items():
        loads[bank] += trie.get(node_id).access_load

    overloaded = [b for b, load in loads.items() if load > C]
    total_overload = sum(max(0.0, load - C) for load in loads.values())
    return BankReport(
        feasible=not overloaded,
        bank_loads=loads,
        overloaded_banks=overloaded,
        max_load=max(loads.values()) if loads else 0.0,
        total_overload=total_overload,
    )


def hashbank(trie, selected: set[int], B: int, C: float) -> tuple[dict[int, int], BankReport]:
    assignment = {v: stable_hash_bank(str(trie.get(v).name), B) for v in selected}
    return assignment, evaluate_bank_assignment(trie, assignment, B, C)


def greedy_load_balance(trie, selected: set[int], B: int, C: float = float("inf")) -> tuple[dict[int, int], BankReport]:
    """Assign heavier selected nodes first to the least-loaded bank."""
    assignment: dict[int, int] = {}
    loads = {b: 0.0 for b in range(B)}

    for node_id in sorted(selected, key=lambda v: trie.get(v).access_load, reverse=True):
        bank = min(loads, key=lambda b: loads[b])
        assignment[node_id] = bank
        loads[bank] += trie.get(node_id).access_load

    return assignment, evaluate_bank_assignment(trie, assignment, B, C)
