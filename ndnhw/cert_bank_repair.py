from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .trie import PrefixTrie
from .trace import InterestTrace
from .banking import greedy_load_balance, evaluate_bank_assignment
from .certificates import evaluate_certificate_aware_coverage, deterministic_action_ids, certificate_candidates


@dataclass(frozen=True)
class CertBankResult:
    selected: set[int]
    assignment: dict[int, int]
    value: float
    feasible: bool
    bank_loads: dict[int, float]
    cert_budget: int
    prefix_budget: int
    certified_coverage: float
    base_certified_coverage: float
    naive_coverage: float
    certificate_gain: float
    certificates_used: int


def _prefix_value(trie: PrefixTrie, selected: set[int]) -> float:
    return sum(trie.get(v).fib_weight for v in selected if trie.get(v).is_fib_prefix)


def _feasible_assignment(trie: PrefixTrie, selected: set[int], B: int, C: float):
    assignment, report = greedy_load_balance(trie, selected, B=B, C=C)
    if not report.feasible:
        return None, report
    return assignment, report


def cert_bank_greedy(
    trie: PrefixTrie,
    trace: InterestTrace,
    total_budget: int,
    D: int,
    B: int,
    C: float,
    action_ids: dict[int, int] | None = None,
    cert_budget: int = 0,
    cert_cost: int = 1,
    max_rounds: int = 128,
) -> CertBankResult:
    """Joint greedy over positive prefixes and shortcut certificates.

    The total budget is split as:
        prefix_budget = total_budget - cert_budget * cert_cost

    During prefix selection, the objective is *certificate-aware* safe coverage, not
    merely exact LPM coverage. This lets the heuristic select cheap shallow prefixes
    when certificates can safely recover deeper action-equivalent LPM decisions.
    """
    if action_ids is None:
        action_ids = deterministic_action_ids(trie, mode="top_component")

    prefix_budget = total_budget - cert_budget * cert_cost
    if prefix_budget < 0:
        prefix_budget = 0

    selected: set[int] = set()
    candidates = {
        node.node_id for node in trie.iter_nodes()
        if node.is_fib_prefix and node.depth <= D
    }

    current_report = evaluate_certificate_aware_coverage(
        trie, trace, selected, D, action_ids=action_ids, max_certificates=cert_budget
    )

    for _ in range(max_rounds):
        best = None

        for v in list(candidates):
            closure = trie.closure({v}, include_root=False)
            closure = {u for u in closure if trie.get(u).depth <= D}
            missing = closure - selected

            if not missing:
                candidates.discard(v)
                continue

            trial = selected | closure
            if len(trial) > prefix_budget:
                candidates.discard(v)
                continue

            assignment, bank_report = _feasible_assignment(trie, trial, B, C)
            if assignment is None:
                continue

            trial_report = evaluate_certificate_aware_coverage(
                trie, trace, trial, D, action_ids=action_ids, max_certificates=cert_budget
            )
            gain = trial_report.certified_hits - current_report.certified_hits
            if gain <= 0:
                continue

            # Favor strong safe-coverage gain per newly stored prefix, then lower max bank load.
            bank_balance_penalty = bank_report.max_load / max(C, 1e-9)
            score = gain / (len(missing) + 0.15 * bank_balance_penalty + 1e-9)

            if best is None or score > best[0]:
                best = (score, v, trial, assignment, bank_report, trial_report)

        if best is None:
            break

        _, chosen, selected, assignment, bank_report, current_report = best
        candidates.discard(chosen)

    assignment, bank_report = greedy_load_balance(trie, selected, B=B, C=C)
    final_report = evaluate_certificate_aware_coverage(
        trie, trace, selected, D, action_ids=action_ids, max_certificates=cert_budget
    )
    return CertBankResult(
        selected=selected,
        assignment=assignment,
        value=_prefix_value(trie, selected),
        feasible=bank_report.feasible,
        bank_loads=bank_report.bank_loads,
        cert_budget=cert_budget,
        prefix_budget=prefix_budget,
        certified_coverage=final_report.certified_coverage,
        base_certified_coverage=final_report.base_certified_coverage,
        naive_coverage=final_report.rate(final_report.naive_hits),
        certificate_gain=final_report.certificate_gain,
        certificates_used=final_report.certificates_used,
    )


def cert_bank_repair(
    trie: PrefixTrie,
    trace: InterestTrace,
    total_budget: int,
    D: int,
    B: int,
    C: float,
    action_ids: dict[int, int] | None = None,
    cert_cost: int = 1,
    cert_budget_options: Iterable[int] | None = None,
) -> CertBankResult:
    """Try multiple prefix/certificate budget splits and keep the best feasible one."""
    if action_ids is None:
        action_ids = deterministic_action_ids(trie, mode="top_component")

    if cert_budget_options is None:
        # Small set of useful budget splits; avoids heavy exhaustive search.
        raw = [0, 4, 8, 16, 32, 64]
        cert_budget_options = [k for k in raw if k * cert_cost <= total_budget]

    best: CertBankResult | None = None
    for cert_budget in cert_budget_options:
        if cert_budget * cert_cost > total_budget:
            continue
        res = cert_bank_greedy(
            trie, trace, total_budget=total_budget, D=D, B=B, C=C,
            action_ids=action_ids, cert_budget=cert_budget, cert_cost=cert_cost,
        )
        if not res.feasible:
            continue
        if best is None or res.certified_coverage > best.certified_coverage:
            best = res

    if best is None:
        # Return the zero solution, which is always feasible.
        assignment, report = greedy_load_balance(trie, set(), B=B, C=C)
        final_report = evaluate_certificate_aware_coverage(
            trie, trace, set(), D, action_ids=action_ids, max_certificates=0
        )
        return CertBankResult(
            selected=set(), assignment=assignment, value=0.0, feasible=True,
            bank_loads=report.bank_loads, cert_budget=0, prefix_budget=total_budget,
            certified_coverage=final_report.certified_coverage,
            base_certified_coverage=final_report.base_certified_coverage,
            naive_coverage=final_report.rate(final_report.naive_hits),
            certificate_gain=final_report.certificate_gain,
            certificates_used=0,
        )
    return best
