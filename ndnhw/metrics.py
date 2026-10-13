from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from .name import NDNName
from .trie import PrefixTrie
from .trace import InterestTrace


@dataclass(frozen=True)
class CoverageReport:
    total: int
    parser_accepted: int
    naive_hits: int
    exact_hits: int
    certified_hits: int
    fallback: int
    unsafe_hits: int

    def rate(self, count: int) -> float:
        return 0.0 if self.total == 0 else count / self.total

    @property
    def parser_coverage(self) -> float:
        return self.rate(self.parser_accepted)

    @property
    def naive_coverage(self) -> float:
        return self.rate(self.naive_hits)

    @property
    def exact_coverage(self) -> float:
        return self.rate(self.exact_hits)

    @property
    def certified_coverage(self) -> float:
        return self.rate(self.certified_hits)

    @property
    def fallback_rate(self) -> float:
        return self.rate(self.fallback)

    @property
    def unsafe_rate(self) -> float:
        return self.rate(self.unsafe_hits)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total": self.total,
            "parser_accepted": self.parser_accepted,
            "naive_hits": self.naive_hits,
            "exact_hits": self.exact_hits,
            "certified_hits": self.certified_hits,
            "fallback": self.fallback,
            "unsafe_hits": self.unsafe_hits,
            "parser_coverage": self.parser_coverage,
            "naive_coverage": self.naive_coverage,
            "exact_coverage": self.exact_coverage,
            "certified_coverage": self.certified_coverage,
            "fallback_rate": self.fallback_rate,
            "unsafe_rate": self.unsafe_rate,
        }


def parser_depth_coverage(trace: InterestTrace, D: int) -> float:
    if len(trace) == 0:
        return 0.0
    return sum(1 for name in trace.names if name.depth <= D) / len(trace)


def selected_weight(trie: PrefixTrie, selected: set[int]) -> float:
    return sum(node.fib_weight for node in trie.nodes if node.node_id in selected and node.is_fib_prefix)


def _has_deeper_fib_prefix_on_path(trie: PrefixTrie, name: NDNName, node_id: int) -> bool:
    """Return True if there exists a deeper FIB prefix matching name below node_id."""
    node = trie.get(node_id)
    current = node_id
    for comp in name.components[node.depth:]:
        cur_node = trie.get(current)
        if comp not in cur_node.children:
            return False
        current = cur_node.children[comp]
        if trie.get(current).is_fib_prefix:
            return True
    return False


def is_certified_decision(trie: PrefixTrie, name: NDNName, selected_match: int | None) -> bool:
    """Check whether the selected match certifies the true LPM decision.

    Certification policy used in this prototype:
    - If there is no selected match, no certified fast-path decision exists.
    - If selected match equals the true LPM, it is certified.
    - Otherwise, it is not certified.

    This conservative definition is the one used for safe fast-path coverage.
    Later variants can add negative certificates to prove absence of deeper prefixes.
    """

    if selected_match is None:
        return False
    true_lpm = trie.lpm(name)
    return selected_match == true_lpm


def evaluate_coverage(
    trie: PrefixTrie,
    trace: InterestTrace,
    selected: set[int],
    D: int,
) -> CoverageReport:
    parser_accepted = 0
    naive_hits = 0
    exact_hits = 0
    certified_hits = 0
    fallback = 0
    unsafe_hits = 0

    for name in trace.names:
        if name.depth > D:
            fallback += 1
            continue

        parser_accepted += 1
        selected_match = trie.deepest_selected_match(name, selected)
        true_lpm = trie.lpm(name)

        if selected_match is not None:
            naive_hits += 1

        if selected_match is not None and selected_match == true_lpm:
            exact_hits += 1

        certified = is_certified_decision(trie, name, selected_match)
        if certified:
            certified_hits += 1
        else:
            if selected_match is not None:
                unsafe_hits += 1
            fallback += 1

    return CoverageReport(
        total=len(trace),
        parser_accepted=parser_accepted,
        naive_hits=naive_hits,
        exact_hits=exact_hits,
        certified_hits=certified_hits,
        fallback=fallback,
        unsafe_hits=unsafe_hits,
    )


def lpm_weight_map(trie: PrefixTrie, trace: InterestTrace) -> Counter[int]:
    counts: Counter[int] = Counter()
    for name in trace.names:
        node_id = trie.lpm(name)
        if node_id is not None:
            counts[node_id] += 1
    return counts
