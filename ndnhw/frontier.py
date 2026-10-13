from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import pandas as pd

from .trace import InterestTrace
from .trie import PrefixTrie
from .pct_dp import pct_dp
from .baselines import topk_closure
from .metrics import evaluate_coverage


@dataclass(frozen=True)
class FrontierPoint:
    D: int
    M: int
    algorithm: str
    certified_coverage: float
    naive_coverage: float
    exact_coverage: float
    fallback_rate: float
    selected_nodes: int
    objective_value: float


def sweep_dm_frontier(
    trie: PrefixTrie,
    trace: InterestTrace,
    D_values: list[int],
    M_values: list[int],
    algorithms: tuple[str, ...] = ("pct_dp", "topk_closure"),
) -> pd.DataFrame:
    rows: list[FrontierPoint] = []

    for D, M in product(D_values, M_values):
        if "pct_dp" in algorithms:
            res = pct_dp(trie, M=M, D=D)
            report = evaluate_coverage(trie, trace, res.selected, D)
            rows.append(FrontierPoint(
                D=D,
                M=M,
                algorithm="PCT-DP",
                certified_coverage=report.certified_coverage,
                naive_coverage=report.naive_coverage,
                exact_coverage=report.exact_coverage,
                fallback_rate=report.fallback_rate,
                selected_nodes=len(res.selected),
                objective_value=res.value,
            ))

        if "topk_closure" in algorithms:
            res = topk_closure(trie, M=M, D=D)
            report = evaluate_coverage(trie, trace, res.selected, D)
            rows.append(FrontierPoint(
                D=D,
                M=M,
                algorithm="TopK-Closure",
                certified_coverage=report.certified_coverage,
                naive_coverage=report.naive_coverage,
                exact_coverage=report.exact_coverage,
                fallback_rate=report.fallback_rate,
                selected_nodes=len(res.selected),
                objective_value=res.value,
            ))

    return pd.DataFrame([r.__dict__ for r in rows])
