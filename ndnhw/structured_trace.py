from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import random

import numpy as np

from .name import NDNName
from .trie import PrefixTrie
from .trace import InterestTrace


@dataclass(frozen=True)
class StructuredTraceConfig:
    n_interests: int = 10_000
    n_prefixes: int = 256
    seed: int = 7
    zipf_alpha: float = 1.1
    domains: tuple[str, ...] = ("video", "sensor", "campus", "repo", "app")
    max_depth: int = 8
    suffix_probability: float = 0.8
    burst_probability: float = 0.25


def _structured_prefix(rng: random.Random, domain: str, max_depth: int) -> NDNName:
    depth = rng.randint(2, max_depth)
    comps = [domain]

    vocab = {
        "video": ["course", "lecture", "segment", "quality", "dash", "clip"],
        "sensor": ["building", "floor", "room", "temp", "humidity", "camera"],
        "campus": ["dept", "lab", "user", "file", "calendar", "notice"],
        "repo": ["pkg", "version", "blob", "chunk", "manifest", "mirror"],
        "app": ["service", "api", "tenant", "object", "feed", "item"],
    }
    words = vocab.get(domain, ["x", "y", "z"])
    while len(comps) < depth:
        comps.append(f"{rng.choice(words)}{rng.randrange(32)}")
    return NDNName.parse(comps)


def generate_structured_trace(config: StructuredTraceConfig) -> tuple[InterestTrace, PrefixTrie]:
    """Generate a more NDN-like hierarchical synthetic trace.

    Compared with the flat random generator, this creates shared top-level domains,
    repeated middle components, and bursty popular subtrees.
    """
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    trie = PrefixTrie()
    prefixes: list[NDNName] = []

    attempts = 0
    while len(prefixes) < config.n_prefixes and attempts < config.n_prefixes * 50:
        attempts += 1
        domain = rng.choice(config.domains)
        prefix = _structured_prefix(rng, domain, config.max_depth)
        node_id = trie.node_id(prefix)
        if node_id is not None and trie.get(node_id).is_fib_prefix:
            continue
        trie.add_fib_prefix(prefix, weight=0.0)
        prefixes.append(prefix)

    ranks = np.arange(1, len(prefixes) + 1)
    probs = 1.0 / (ranks ** config.zipf_alpha)
    probs = probs / probs.sum()

    # Bursty windows reuse a small hot subset.
    hot_subset = list(np_rng.choice(len(prefixes), size=max(1, len(prefixes) // 10), replace=False))

    names: list[NDNName] = []
    lpm_counts: Counter[int] = Counter()

    i = 0
    while i < config.n_interests:
        if rng.random() < config.burst_probability:
            burst_len = min(rng.randint(10, 80), config.n_interests - i)
            idx = int(rng.choice(hot_subset))
            choices = [idx] * burst_len
        else:
            burst_len = 1
            choices = [int(np_rng.choice(len(prefixes), p=probs))]

        for idx in choices:
            prefix = prefixes[idx]
            comps = list(prefix.components)
            if rng.random() < config.suffix_probability:
                comps += [f"seg{rng.randrange(2048)}"]
                if rng.random() < 0.35:
                    comps += [f"chunk{rng.randrange(2048)}"]
            name = NDNName.parse(comps)
            names.append(name)
            lpm = trie.lpm(name)
            if lpm is not None:
                lpm_counts[lpm] += 1
        i += burst_len

    names = names[: config.n_interests]

    for node in trie.nodes:
        node.fib_weight = 0.0
        node.access_load = 0.0

    for node_id, count in lpm_counts.items():
        trie.nodes[node_id].fib_weight = float(count)
        trie.nodes[node_id].access_load = float(count)

    return InterestTrace(names), trie
