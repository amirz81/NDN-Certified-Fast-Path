from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from ndnhw.structured_trace import StructuredTraceConfig, generate_structured_trace
from ndnhw.pct_dp import pct_dp
from ndnhw.baselines import topk_closure, coverage_first
from ndnhw.bank_greedy import bank_greedy_ls
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.banking import hashbank
from ndnhw.metrics import evaluate_coverage, parser_depth_coverage
from ndnhw.certificates import deterministic_action_ids, evaluate_certificate_aware_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='outputs/v03_smoke')
    parser.add_argument('--n-interests', type=int, default=1200)
    parser.add_argument('--n-prefixes', type=int, default=48)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trace, trie = generate_structured_trace(StructuredTraceConfig(
        n_interests=args.n_interests,
        n_prefixes=args.n_prefixes,
        seed=7,
        zipf_alpha=1.1,
    ))

    parser_rows = []
    for D in range(1, 12):
        parser_rows.append({'D': D, 'parser_coverage': parser_depth_coverage(trace, D)})
    parser_df = pd.DataFrame(parser_rows)
    parser_df.to_csv(outdir / 'parser_smoke.csv', index=False)

    D, M = 8, 96
    dp = pct_dp(trie, M=M, D=D)
    topk = topk_closure(trie, M=M, D=D)
    action_ids = deterministic_action_ids(trie, mode='top_component', seed=7)

    cert_rows = []
    for alg, selected in [('PCT-DP', dp.selected), ('TopK-Closure', topk.selected)]:
        base = evaluate_coverage(trie, trace, selected, D)
        for K in [0, 4, 8, 16, 32]:
            crep = evaluate_certificate_aware_coverage(
                trie, trace, selected, D, action_ids=action_ids, max_certificates=K
            )
            cert_rows.append({
                'algorithm': alg,
                'certificate_budget': K,
                'base_certified_coverage': base.certified_coverage,
                'naive_coverage': base.naive_coverage,
                **crep.as_dict(),
            })
    cert_df = pd.DataFrame(cert_rows)
    cert_df.to_csv(outdir / 'certificate_smoke.csv', index=False)

    bank_rows = []
    D_bank, M_bank = 7, 64
    total_load = sum(node.access_load for node in trie.nodes)
    fixed_total_C1 = total_load * 0.35
    for regime in ['fixed_total', 'fixed_per_bank']:
        for B in [1, 2, 4]:
            C = fixed_total_C1 / B if regime == 'fixed_total' else fixed_total_C1

            dp_bank = pct_dp(trie, M=M_bank, D=D_bank)
            _, hrep = hashbank(trie, dp_bank.selected, B=B, C=C)
            hcov = evaluate_coverage(trie, trace, dp_bank.selected, D_bank)
            bank_rows.append({'regime': regime, 'B': B, 'algorithm': 'PCT-DP+HashBank', 'feasible': hrep.feasible, 'coverage': hcov.certified_coverage, 'overload': hrep.total_overload})

            cf = coverage_first(trie, M=M_bank, D=D_bank, B=B, C=C)
            cfcov = evaluate_coverage(trie, trace, cf.selected, D_bank)
            bank_rows.append({'regime': regime, 'B': B, 'algorithm': 'CoverageFirst', 'feasible': cf.bank_report.feasible, 'coverage': cfcov.certified_coverage, 'overload': cf.bank_report.total_overload})

            bg = bank_greedy_ls(trie, M=M_bank, D=D_bank, B=B, C=C, max_rounds=2)
            bgcov = evaluate_coverage(trie, trace, bg.selected, D_bank)
            bank_rows.append({'regime': regime, 'B': B, 'algorithm': 'BankGreedy-LS', 'feasible': bg.feasible, 'coverage': bgcov.certified_coverage, 'overload': 0.0 if bg.feasible else None})

            repair = repair_bank_greedy(trie, M=M_bank, D=D_bank, B=B, C=C, max_rounds=2)
            rcov = evaluate_coverage(trie, trace, repair.selected, D_bank)
            bank_rows.append({'regime': regime, 'B': B, 'algorithm': 'BankRepair', 'feasible': repair.feasible, 'coverage': rcov.certified_coverage, 'overload': 0.0 if repair.feasible else None})

    bank_df = pd.DataFrame(bank_rows)
    bank_df.to_csv(outdir / 'bank_regimes_smoke.csv', index=False)

    plt.figure()
    plt.plot(parser_df['D'], parser_df['parser_coverage'], marker='o')
    plt.xlabel('Parser depth D')
    plt.ylabel('Parser coverage')
    plt.title('v0.3 smoke parser curve')
    plt.tight_layout()
    plt.savefig(outdir / 'plot_parser_smoke.png', dpi=160)
    plt.close()

    plt.figure()
    for alg, g in cert_df.groupby('algorithm'):
        plt.plot(g['certificate_budget'], g['certified_coverage'], marker='o', label=alg)
    plt.xlabel('Certificate budget')
    plt.ylabel('Certified coverage')
    plt.title('Certificate-aware gain')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'plot_certificate_smoke.png', dpi=160)
    plt.close()

    plt.figure()
    for (regime, alg), g in bank_df.groupby(['regime', 'algorithm']):
        if alg in ['CoverageFirst', 'BankGreedy-LS', 'BankRepair']:
            plt.plot(g['B'], g['coverage'], marker='o', label=f'{regime} {alg}')
    plt.xlabel('Bank count B')
    plt.ylabel('Certified coverage')
    plt.title('Bank regimes smoke')
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / 'plot_bank_regimes_smoke.png', dpi=160)
    plt.close()

    summary = {
        'trace_size': len(trace),
        'fib_prefixes': sum(1 for n in trie.nodes if n.is_fib_prefix),
        'trie_nodes': len(trie),
        'max_parser_coverage': float(parser_df['parser_coverage'].max()),
        'best_certificate_coverage': float(cert_df['certified_coverage'].max()),
        'best_bank_feasible_coverage': float(bank_df[bank_df['feasible']]['coverage'].max()),
    }
    (outdir / 'v03_smoke_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
