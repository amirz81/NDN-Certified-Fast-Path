from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import random

from .name import NDNName
from .trie import PrefixTrie
from .trace import InterestTrace
from .metrics import CoverageReport


@dataclass(frozen=True)
class CertificateReport:
    total: int
    parser_accepted: int
    base_certified_hits: int
    certificate_hits: int
    certified_hits: int
    naive_hits: int
    exact_hits: int
    fallback: int
    unsafe_hits: int
    certificates_used: int
    certificate_candidates: int

    def rate(self, count: int) -> float:
        return 0.0 if self.total == 0 else count / self.total

    @property
    def certified_coverage(self) -> float:
        return self.rate(self.certified_hits)

    @property
    def base_certified_coverage(self) -> float:
        return self.rate(self.base_certified_hits)

    @property
    def certificate_gain(self) -> float:
        return self.rate(self.certificate_hits)

    @property
    def fallback_rate(self) -> float:
        return self.rate(self.fallback)

    @property
    def unsafe_rate(self) -> float:
        return self.rate(self.unsafe_hits)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "parser_accepted": self.parser_accepted,
            "base_certified_hits": self.base_certified_hits,
            "certificate_hits": self.certificate_hits,
            "certified_hits": self.certified_hits,
            "naive_hits": self.naive_hits,
            "exact_hits": self.exact_hits,
            "fallback": self.fallback,
            "unsafe_hits": self.unsafe_hits,
            "certificates_used": self.certificates_used,
            "certificate_candidates": self.certificate_candidates,
            "base_certified_coverage": self.base_certified_coverage,
            "certified_coverage": self.certified_coverage,
            "certificate_gain": self.certificate_gain,
            "fallback_rate": self.fallback_rate,
            "unsafe_rate": self.unsafe_rate,
        }


def deterministic_action_ids(
    trie: PrefixTrie,
    mode: str = "top_component",
    num_actions: int = 8,
    inherit_probability: float = 0.75,
    seed: int = 7,
) -> dict[int, int]:
    """Assign synthetic forwarding actions to FIB prefixes.

    This models whether a deeper FIB prefix semantically overrides a shallower one.
    In real NDN, action equivalence would come from next-hop/face sets. For synthetic
    traces, this gives controlled action equivalence.

    Modes:
    - top_component: all prefixes under the same first component share action.
    - hash: action is hash(prefix) mod num_actions.
    - inherited_hash: deeper FIB prefixes often inherit their closest FIB ancestor action.
    """
    rng = random.Random(seed)
    actions: dict[int, int] = {}

    for node in trie.nodes:
        if not node.is_fib_prefix:
            continue

        if mode == "top_component":
            top = node.name.components[0] if node.name.components else "root"
            digest = hashlib.sha256(top.encode("utf-8")).hexdigest()
            actions[node.node_id] = int(digest, 16) % num_actions

        elif mode == "hash":
            digest = hashlib.sha256(str(node.name).encode("utf-8")).hexdigest()
            actions[node.node_id] = int(digest, 16) % num_actions

        elif mode == "inherited_hash":
            parent = node.parent_id
            inherited = None
            while parent is not None:
                if parent in actions:
                    inherited = actions[parent]
                    break
                parent = trie.get(parent).parent_id

            if inherited is not None and rng.random() < inherit_probability:
                actions[node.node_id] = inherited
            else:
                digest = hashlib.sha256(str(node.name).encode("utf-8")).hexdigest()
                actions[node.node_id] = int(digest, 16) % num_actions
        else:
            raise ValueError(f"unknown action mode: {mode}")

    return actions


def certificate_candidates(
    trie: PrefixTrie,
    trace: InterestTrace,
    selected: set[int],
    D: int,
    action_ids: dict[int, int],
) -> Counter[tuple[int, int]]:
    """Count shortcut certificates that would safely replace deeper true LPMs.

    A certificate key (s, t) means:
    hardware matched selected shallower prefix s, true LPM is deeper prefix t,
    but both prefixes have the same forwarding action, so an equivalence certificate
    can safely let hardware use s's action for traffic whose true LPM is t.
    """
    counts: Counter[tuple[int, int]] = Counter()

    for name in trace.names:
        if name.depth > D:
            continue

        selected_match = trie.deepest_selected_match(name, selected)
        true_lpm = trie.lpm(name)

        if selected_match is None or true_lpm is None:
            continue

        if selected_match == true_lpm:
            continue

        if action_ids.get(selected_match) == action_ids.get(true_lpm):
            counts[(selected_match, true_lpm)] += 1

    return counts


def evaluate_certificate_aware_coverage(
    trie: PrefixTrie,
    trace: InterestTrace,
    selected: set[int],
    D: int,
    action_ids: dict[int, int] | None = None,
    max_certificates: int = 0,
    action_mode: str = "top_component",
    num_actions: int = 8,
    seed: int = 7,
) -> CertificateReport:
    """Evaluate safe coverage with optional action-equivalence certificates.

    Base certified hits require selected_match == true_lPM.
    Certificate hits allow selected_match != true_lPM only when a stored certificate
    proves forwarding-action equivalence for that shortcut.
    """
    if action_ids is None:
        action_ids = deterministic_action_ids(
            trie,
            mode=action_mode,
            num_actions=num_actions,
            seed=seed,
        )

    candidates = certificate_candidates(trie, trace, selected, D, action_ids)
    chosen = set(k for k, _ in candidates.most_common(max_certificates))

    parser_accepted = 0
    naive_hits = 0
    exact_hits = 0
    base_certified_hits = 0
    certificate_hits = 0
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
            base_certified_hits += 1
            continue

        if (
            selected_match is not None
            and true_lpm is not None
            and (selected_match, true_lpm) in chosen
        ):
            certificate_hits += 1
            continue

        if selected_match is not None:
            unsafe_hits += 1
        fallback += 1

    certified_hits = base_certified_hits + certificate_hits

    return CertificateReport(
        total=len(trace),
        parser_accepted=parser_accepted,
        base_certified_hits=base_certified_hits,
        certificate_hits=certificate_hits,
        certified_hits=certified_hits,
        naive_hits=naive_hits,
        exact_hits=exact_hits,
        fallback=fallback,
        unsafe_hits=unsafe_hits,
        certificates_used=len(chosen),
        certificate_candidates=len(candidates),
    )
