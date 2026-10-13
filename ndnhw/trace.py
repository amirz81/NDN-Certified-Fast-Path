from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import random
from typing import Iterable

import numpy as np

from .name import NDNName
from .trie import PrefixTrie


@dataclass(frozen=True)
class SyntheticTraceConfig:
    n_interests: int = 10_000
    n_prefixes: int = 512
    min_depth: int = 2
    max_depth: int = 8
    zipf_alpha: float = 1.1
    seed: int = 7
    component_pool: int = 64
    suffix_pool: int = 512
    p_deeper_than_prefix: float = 0.8


@dataclass
class InterestTrace:
    names: list[NDNName]

    def __len__(self) -> int:
        return len(self.names)

    def name_counts(self) -> Counter[NDNName]:
        return Counter(self.names)

    def length_distribution(self) -> Counter[int]:
        return Counter(name.depth for name in self.names)

    @property
    def total_weight(self) -> int:
        return len(self.names)


def _random_prefix(rng: random.Random, depth: int, component_pool: int) -> NDNName:
    comps = [f"c{rng.randrange(component_pool)}" for _ in range(depth)]
    return NDNName.parse(comps)


def generate_synthetic_trace(config: SyntheticTraceConfig) -> tuple[InterestTrace, PrefixTrie]:
    """Generate a synthetic NDN Interest trace and a FIB trie.

    The generator creates a FIB prefix set, samples true LPM prefixes with Zipf popularity,
    then appends a random suffix to most Interests so that names are longer than FIB prefixes.
    """

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    trie = PrefixTrie()
    fib_prefixes: list[NDNName] = []

    attempts = 0
    while len(fib_prefixes) < config.n_prefixes and attempts < config.n_prefixes * 20:
        attempts += 1
        depth = rng.randint(config.min_depth, config.max_depth)
        prefix = _random_prefix(rng, depth, config.component_pool)
        if str(prefix) == "/":
            continue
        if trie.node_id(prefix) is not None and trie.get(trie.node_id(prefix)).is_fib_prefix:
            continue
        trie.add_fib_prefix(prefix, weight=0.0)
        fib_prefixes.append(prefix)

    if not fib_prefixes:
        raise RuntimeError("failed to generate FIB prefixes")

    ranks = np.arange(1, len(fib_prefixes) + 1)
    weights = 1.0 / (ranks ** config.zipf_alpha)
    probs = weights / weights.sum()

    names: list[NDNName] = []
    prefix_choice_ids = np_rng.choice(len(fib_prefixes), size=config.n_interests, p=probs)

    lpm_counts: Counter[int] = Counter()

    for idx in prefix_choice_ids:
        prefix = fib_prefixes[int(idx)]
        comps = list(prefix.components)
        if rng.random() < config.p_deeper_than_prefix:
            suffix_len = rng.randint(1, 3)
            comps += [f"s{rng.randrange(config.suffix_pool)}" for _ in range(suffix_len)]
        name = NDNName.parse(comps)
        names.append(name)
        lpm_id = trie.lpm(name)
        if lpm_id is not None:
            lpm_counts[lpm_id] += 1

    # Update FIB weights and access load using actual trace LPM counts.
    for node in trie.nodes:
        node.fib_weight = 0.0
        node.access_load = 0.0
    for node_id, count in lpm_counts.items():
        trie.nodes[node_id].fib_weight = float(count)
        trie.nodes[node_id].access_load = float(count)

    return InterestTrace(names), trie
