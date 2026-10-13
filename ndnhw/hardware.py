from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareBudget:
    """Hardware budget for realizability experiments.

    D: parser depth, maximum parsed name components.
    M: number of hardware-resident trie nodes/prefixes.
    B: number of memory banks.
    C: per-bank load capacity.
    U: update bandwidth per time window.
    """

    D: int
    M: int
    B: int = 1
    C: float = float("inf")
    U: float = float("inf")

    def validate(self) -> None:
        if self.D < 0:
            raise ValueError("parser depth D must be non-negative")
        if self.M < 0:
            raise ValueError("memory capacity M must be non-negative")
        if self.B <= 0:
            raise ValueError("bank count B must be positive")
        if self.C < 0:
            raise ValueError("bank capacity C must be non-negative")
        if self.U < 0:
            raise ValueError("update capacity U must be non-negative")
