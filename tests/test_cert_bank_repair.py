from collections import Counter

from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.name import NDNName
from ndnhw.trie import PrefixTrie
from ndnhw.trace import InterestTrace


def build_small_stress():
    trie = PrefixTrie()
    action_ids = {}
    names = []
    for g in range(4):
        parent = f"/svc{g}/topic"
        pid = trie.add_fib_prefix(parent, weight=0.0, access_load=0.0)
        action_ids[pid] = g
        for c in range(4):
            child = f"{parent}/sub{c}"
            cid = trie.add_fib_prefix(child, weight=8.0, access_load=8.0)
            action_ids[cid] = g if c < 3 else 1000 + g
            for i in range(8):
                names.append(NDNName.parse(f"{child}/item{i}"))
    counts = Counter()
    for name in names:
        counts[trie.lpm(name)] += 1
    for node in trie.nodes:
        node.fib_weight = 0.0
        node.access_load = 0.0
    for node_id, count in counts.items():
        trie.nodes[node_id].fib_weight = float(count)
        trie.nodes[node_id].access_load = float(count)
    return InterestTrace(names), trie, action_ids


def test_cert_bank_repair_feasible_and_nonzero():
    trace, trie, action_ids = build_small_stress()
    total_load = sum(node.access_load for node in trie.nodes)
    res = cert_bank_repair(
        trie, trace, total_budget=16, D=8, B=2, C=total_load * 0.35,
        action_ids=action_ids,
        cert_budget_options=[0, 4, 8],
    )
    assert res.feasible
    assert res.certified_coverage > 0
    assert len(res.selected) <= res.prefix_budget
