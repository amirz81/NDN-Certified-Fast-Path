from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .trie import PrefixTrie
from .trace import InterestTrace
from .banking import greedy_load_balance
from .bank_repair import repair_bank_greedy
from .cert_bank_repair import cert_bank_repair
from .certificates import evaluate_certificate_aware_coverage, deterministic_action_ids


@dataclass(frozen=True)
class PortfolioResult:
    selected: set[int]
    assignment: dict[int, int]
    feasible: bool
    value: float
    certified_coverage: float
    base_certified_coverage: float
    naive_coverage: float
    certificate_gain: float
    certificates_used: int
    cert_budget: int
    prefix_budget: int
    source: str


def _value(trie: PrefixTrie, selected: set[int]) -> float:
    return sum(trie.get(v).fib_weight for v in selected if trie.get(v).is_fib_prefix)


def _evaluate_candidate(
    trie: PrefixTrie,
    trace: InterestTrace,
    selected: set[int],
    D: int,
    B: int,
    C: float,
    action_ids: dict[int, int],
    cert_budget: int,
    prefix_budget: int,
    source: str,
) -> PortfolioResult | None:
    if len(selected) > prefix_budget or not trie.is_prefix_closed(selected):
        return None

    assignment, bank_report = greedy_load_balance(trie, selected, B=B, C=C)
    if not bank_report.feasible:
        return None

    rep = evaluate_certificate_aware_coverage(
        trie, trace, selected, D,
        action_ids=action_ids,
        max_certificates=cert_budget,
    )
    return PortfolioResult(
        selected=selected,
        assignment=assignment,
        feasible=True,
        value=_value(trie, selected),
        certified_coverage=rep.certified_coverage,
        base_certified_coverage=rep.base_certified_coverage,
        naive_coverage=rep.rate(rep.naive_hits),
        certificate_gain=rep.certificate_gain,
        certificates_used=rep.certificates_used,
        cert_budget=cert_budget,
        prefix_budget=prefix_budget,
        source=source,
    )


def _weighted_exact_selection(trie: PrefixTrie, prefix_budget: int, D: int) -> set[int]:
    selected: set[int] = set()
    candidates = [
        n for n in trie.iter_nodes()
        if n.is_fib_prefix and n.depth <= D and n.fib_weight > 0
    ]
    candidates.sort(key=lambda n: n.fib_weight, reverse=True)

    for node in candidates:
        closure = trie.closure({node.node_id}, include_root=False)
        closure = {u for u in closure if trie.get(u).depth <= D}
        trial = selected | closure
        if len(trial) <= prefix_budget:
            selected = trial
    return selected


def _shallow_shortcut_selection(trie: PrefixTrie, prefix_budget: int, D: int) -> set[int]:
    selected: set[int] = set()
    candidates = [
        n for n in trie.iter_nodes()
        if n.is_fib_prefix and n.depth <= D
    ]
    # Favor shallow prefixes first, then high descendant mass.
    def subtree_weight(node_id: int) -> float:
        return sum(
            trie.get(v).fib_weight
            for v in trie.descendants(node_id)
            if trie.get(v).is_fib_prefix
        )

    candidates.sort(key=lambda n: (n.depth, -subtree_weight(n.node_id), -n.fib_weight))

    for node in candidates:
        closure = trie.closure({node.node_id}, include_root=False)
        closure = {u for u in closure if trie.get(u).depth <= D}
        trial = selected | closure
        if len(trial) <= prefix_budget:
            selected = trial
    return selected


def cert_bank_portfolio(
    trie: PrefixTrie,
    trace: InterestTrace,
    total_budget: int,
    D: int,
    B: int,
    C: float,
    action_ids: dict[int, int] | None = None,
    cert_budget_options: Iterable[int] | None = None,
    cert_cost: int = 1,
) -> PortfolioResult:
    """Portfolio heuristic for certified bank-aware placement.

    It tries several structurally different feasible constructions:
    - original joint BPC-Repair,
    - bank repair plus certificates,
    - exact-prefix greedy plus certificates,
    - shallow-shortcut greedy plus certificates.

    Then it returns the feasible candidate with the highest certified coverage.
    This is not meant as a theoretical oracle; it is a robust headliner heuristic
    for experiments where the winning regime changes with bank pressure.
    """
    if action_ids is None:
        action_ids = deterministic_action_ids(trie, mode="top_component")

    if cert_budget_options is None:
        raw = [0, 2, 4, 8, 12, 16, 24, 32]
        cert_budget_options = [k for k in raw if k * cert_cost <= total_budget]

    best: PortfolioResult | None = None

    def consider(res: PortfolioResult | None):
        nonlocal best
        if res is None:
            return
        if best is None or (
            res.certified_coverage,
            res.value,
            -len(res.selected),
        ) > (
            best.certified_coverage,
            best.value,
            -len(best.selected),
        ):
            best = res

    # Original BPC-Repair across same split options.
    bpc = cert_bank_repair(
        trie, trace, total_budget=total_budget, D=D, B=B, C=C,
        action_ids=action_ids, cert_cost=cert_cost,
        cert_budget_options=cert_budget_options,
    )
    consider(PortfolioResult(
        selected=bpc.selected,
        assignment=bpc.assignment,
        feasible=bpc.feasible,
        value=bpc.value,
        certified_coverage=bpc.certified_coverage,
        base_certified_coverage=bpc.base_certified_coverage,
        naive_coverage=bpc.naive_coverage,
        certificate_gain=bpc.certificate_gain,
        certificates_used=bpc.certificates_used,
        cert_budget=bpc.cert_budget,
        prefix_budget=bpc.prefix_budget,
        source="BPC-Repair",
    ))

    for cert_budget in cert_budget_options:
        prefix_budget = max(0, total_budget - cert_budget * cert_cost)

        # Bank-repair seed under reduced prefix budget.
        repaired = repair_bank_greedy(trie, M=prefix_budget, D=D, B=B, C=C, max_rounds=5)
        consider(_evaluate_candidate(
            trie, trace, repaired.selected, D, B, C, action_ids,
            cert_budget, prefix_budget, "BankRepair+Cert",
        ))

        # Exact-heavy greedy seed.
        exact = _weighted_exact_selection(trie, prefix_budget, D)
        consider(_evaluate_candidate(
            trie, trace, exact, D, B, C, action_ids,
            cert_budget, prefix_budget, "ExactGreedy+Cert",
        ))

        # Shortcut-heavy shallow seed.
        shallow = _shallow_shortcut_selection(trie, prefix_budget, D)
        consider(_evaluate_candidate(
            trie, trace, shallow, D, B, C, action_ids,
            cert_budget, prefix_budget, "ShallowShortcut+Cert",
        ))

    if best is None:
        assignment, _ = greedy_load_balance(trie, set(), B=B, C=C)
        rep = evaluate_certificate_aware_coverage(
            trie, trace, set(), D, action_ids=action_ids, max_certificates=0
        )
        return PortfolioResult(
            selected=set(), assignment=assignment, feasible=True, value=0.0,
            certified_coverage=rep.certified_coverage,
            base_certified_coverage=rep.base_certified_coverage,
            naive_coverage=rep.rate(rep.naive_hits),
            certificate_gain=rep.certificate_gain,
            certificates_used=0,
            cert_budget=0,
            prefix_budget=total_budget,
            source="empty",
        )
    return best
