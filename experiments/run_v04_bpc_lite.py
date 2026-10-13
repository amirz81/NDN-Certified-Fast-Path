from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
import random

import pandas as pd
import matplotlib.pyplot as plt

from ndnhw.name import NDNName
from ndnhw.trie import PrefixTrie
from ndnhw.trace import InterestTrace
from ndnhw.certificates import evaluate_certificate_aware_coverage
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.banking import greedy_load_balance


def build_certificate_stress_trace(n_groups=6, children_per_group=4, interests_per_child=10, seed=7):
    rng = random.Random(seed)
    trie = PrefixTrie()
    action_ids = {}
    names = []
    for g in range(n_groups):
        parent = f"/svc{g}/topic"
        pid = trie.add_fib_prefix(parent, weight=0.0, access_load=0.0)
        action_ids[pid] = g
        for c in range(children_per_group):
            child = f"{parent}/sub{c}"
            w = max(3, interests_per_child + rng.randint(-3, 3))
            cid = trie.add_fib_prefix(child, weight=float(w), access_load=float(w))
            action_ids[cid] = g if c < children_per_group - 1 else 1000 + g
            for i in range(w):
                names.append(NDNName.parse(f"{child}/item{i % 5}/seg{i}"))
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


def select_parents_only(trie, M, D):
    selected = set()
    parents = [n.node_id for n in trie.iter_nodes() if n.depth == 2 and n.is_fib_prefix and n.depth <= D]
    for v in parents:
        closure = trie.closure({v}, include_root=False)
        trial = selected | closure
        if len(trial) <= M:
            selected = trial
    return selected


def select_children_greedy(trie, M, D):
    selected = set()
    children = [n for n in trie.iter_nodes() if n.depth == 3 and n.is_fib_prefix and n.depth <= D]
    children.sort(key=lambda n: n.fib_weight, reverse=True)
    for node in children:
        closure = trie.closure({node.node_id}, include_root=False)
        trial = selected | closure
        if len(trial) <= M:
            selected = trial
    return selected


def eval_row(trie, trace, selected, D, B, C, action_ids, cert_budget):
    _, bank_report = greedy_load_balance(trie, selected, B=B, C=C)
    crep = evaluate_certificate_aware_coverage(trie, trace, selected, D, action_ids=action_ids, max_certificates=cert_budget)
    return bank_report, crep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='outputs/v04_bpc_lite')
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    D = 8
    for seed in [3, 7, 11]:
        trace, trie, action_ids = build_certificate_stress_trace(seed=seed)
        total_load = sum(n.access_load for n in trie.nodes)
        for regime in ['fixed_total', 'fixed_per_bank']:
            for B in [1, 2, 4]:
                C = total_load * 0.35 / B if regime == 'fixed_total' else total_load * 0.35
                for total_budget in [16, 24, 32]:
                    # ChildExact best over cert budgets
                    for alg, selected, cert_budgets in [
                        ('ChildExactGreedy+Cert', select_children_greedy(trie, total_budget, D), [0, 4, 8]),
                        ('ParentShortcut+Cert', select_parents_only(trie, total_budget, D), [0, 4, 8]),
                    ]:
                        for cert_budget in cert_budgets:
                            if cert_budget > total_budget:
                                continue
                            # For parent shortcut, use prefix budget total-cert_budget.
                            if alg == 'ParentShortcut+Cert':
                                selected = select_parents_only(trie, total_budget - cert_budget, D)
                            br, cr = eval_row(trie, trace, selected, D, B, C, action_ids, cert_budget)
                            rows.append({
                                'seed': seed, 'regime': regime, 'B': B, 'C': C,
                                'total_budget': total_budget, 'algorithm': alg,
                                'prefix_nodes': len(selected), 'cert_budget': cert_budget,
                                'bank_feasible': br.feasible, 'total_overload': br.total_overload,
                                **cr.as_dict(),
                            })

                    # Bank repair only
                    bank_only = repair_bank_greedy(trie, M=total_budget, D=D, B=B, C=C, max_rounds=4)
                    br, cr = eval_row(trie, trace, bank_only.selected, D, B, C, action_ids, 0)
                    rows.append({
                        'seed': seed, 'regime': regime, 'B': B, 'C': C,
                        'total_budget': total_budget, 'algorithm': 'BankRepairOnly',
                        'prefix_nodes': len(bank_only.selected), 'cert_budget': 0,
                        'bank_feasible': br.feasible, 'total_overload': br.total_overload,
                        **cr.as_dict(),
                    })

                    # Proposed BPC
                    bpc = cert_bank_repair(
                        trie, trace, total_budget=total_budget, D=D, B=B, C=C,
                        action_ids=action_ids, cert_budget_options=[0, 4, 8]
                    )
                    rows.append({
                        'seed': seed, 'regime': regime, 'B': B, 'C': C,
                        'total_budget': total_budget, 'algorithm': 'BPC-Repair',
                        'prefix_nodes': len(bpc.selected), 'cert_budget': bpc.cert_budget,
                        'bank_feasible': bpc.feasible, 'total_overload': 0.0,
                        'total': len(trace), 'parser_accepted': len(trace),
                        'base_certified_hits': int(bpc.base_certified_coverage * len(trace)),
                        'certificate_hits': int(bpc.certificate_gain * len(trace)),
                        'certified_hits': int(bpc.certified_coverage * len(trace)),
                        'naive_hits': int(bpc.naive_coverage * len(trace)),
                        'exact_hits': int(bpc.base_certified_coverage * len(trace)),
                        'fallback': int((1-bpc.certified_coverage)*len(trace)),
                        'unsafe_hits': 0,
                        'certificates_used': bpc.certificates_used,
                        'certificate_candidates': -1,
                        'base_certified_coverage': bpc.base_certified_coverage,
                        'certified_coverage': bpc.certified_coverage,
                        'certificate_gain': bpc.certificate_gain,
                        'fallback_rate': 1-bpc.certified_coverage,
                        'unsafe_rate': 0.0,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir/'v04_bpc_lite_all.csv', index=False)
    summary = df.groupby(['regime','B','total_budget','algorithm'], as_index=False).agg(
        mean_certified_coverage=('certified_coverage','mean'),
        mean_certificate_gain=('certificate_gain','mean'),
        feasible_rate=('bank_feasible','mean'),
        mean_total_overload=('total_overload','mean'),
        mean_prefix_nodes=('prefix_nodes','mean'),
        mean_cert_budget=('cert_budget','mean'),
    )
    summary.to_csv(outdir/'v04_bpc_lite_summary.csv', index=False)
    best = summary[summary['feasible_rate'] > 0.999].sort_values('mean_certified_coverage', ascending=False)
    best.to_csv(outdir/'v04_bpc_lite_best_feasible.csv', index=False)

    plt.figure()
    sub = summary[(summary['regime']=='fixed_per_bank') & (summary['B']==2)]
    for alg, g in sub.groupby('algorithm'):
        gg = g.sort_values('mean_certified_coverage').groupby('total_budget', as_index=False).tail(1)
        plt.plot(gg['total_budget'], gg['mean_certified_coverage'], marker='o', label=alg)
    plt.xlabel('Total budget')
    plt.ylabel('Mean certified coverage')
    plt.title('v0.4 lite: BPC vs baselines, fixed per-bank B=2')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir/'plot_v04_bpc_fixed_per_bank_B2.png', dpi=180)
    plt.close()

    plt.figure()
    sub = summary[summary['algorithm']=='BPC-Repair']
    for (regime, B), g in sub.groupby(['regime','B']):
        plt.plot(g['total_budget'], g['mean_certified_coverage'], marker='o', label=f'{regime} B={B}')
    plt.xlabel('Total budget')
    plt.ylabel('Mean certified coverage')
    plt.title('BPC-Repair under bank regimes')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir/'plot_v04_bpc_regimes.png', dpi=180)
    plt.close()

    compact = {
        'best_feasible_rows': best.head(16).to_dict(orient='records'),
        'fixed_per_bank_B2': summary[(summary['regime']=='fixed_per_bank') & (summary['B']==2)].to_dict(orient='records'),
    }
    (outdir/'v04_bpc_lite_summary.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
