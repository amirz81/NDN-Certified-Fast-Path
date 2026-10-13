from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ndnhw.trace import SyntheticTraceConfig, generate_synthetic_trace
from ndnhw.pct_dp import pct_dp
from ndnhw.baselines import coverage_first
from ndnhw.bank_greedy import bank_greedy, bank_greedy_ls
from ndnhw.banking import hashbank
from ndnhw.metrics import evaluate_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/bank_sweep.csv")
    parser.add_argument("--n-interests", type=int, default=20_000)
    parser.add_argument("--n-prefixes", type=int, default=512)
    parser.add_argument("--D", type=int, default=7)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--bank-capacity-frac", type=float, default=0.35)
    args = parser.parse_args()

    trace, trie = generate_synthetic_trace(SyntheticTraceConfig(
        n_interests=args.n_interests,
        n_prefixes=args.n_prefixes,
    ))

    rows = []
    for B in [1, 2, 4, 8, 16]:
        total_load = sum(node.access_load for node in trie.nodes)
        C = (total_load / B) * args.bank_capacity_frac

        dp = pct_dp(trie, M=args.M, D=args.D)
        hash_assignment, hash_report = hashbank(trie, dp.selected, B=B, C=C)
        hash_cov = evaluate_coverage(trie, trace, dp.selected, args.D)
        rows.append({
            "algorithm": "PCT-DP+HashBank",
            "B": B,
            "C": C,
            "value": dp.value,
            "selected_nodes": len(dp.selected),
            "bank_feasible": hash_report.feasible,
            "total_overload": hash_report.total_overload,
            "max_load": hash_report.max_load,
            "certified_coverage": hash_cov.certified_coverage,
            "fallback_rate": hash_cov.fallback_rate,
        })

        cf = coverage_first(trie, M=args.M, D=args.D, B=B, C=C)
        cf_cov = evaluate_coverage(trie, trace, cf.selected, args.D)
        rows.append({
            "algorithm": "CoverageFirst",
            "B": B,
            "C": C,
            "value": cf.value,
            "selected_nodes": len(cf.selected),
            "bank_feasible": cf.bank_report.feasible,
            "total_overload": cf.bank_report.total_overload,
            "max_load": cf.bank_report.max_load,
            "certified_coverage": cf_cov.certified_coverage,
            "fallback_rate": cf_cov.fallback_rate,
        })

        bg = bank_greedy(trie, M=args.M, D=args.D, B=B, C=C)
        bg_cov = evaluate_coverage(trie, trace, bg.selected, args.D)
        rows.append({
            "algorithm": "BankGreedy",
            "B": B,
            "C": C,
            "value": bg.value,
            "selected_nodes": len(bg.selected),
            "bank_feasible": bg.feasible,
            "total_overload": 0.0 if bg.feasible else None,
            "max_load": max(bg.bank_loads.values()) if bg.bank_loads else 0.0,
            "certified_coverage": bg_cov.certified_coverage,
            "fallback_rate": bg_cov.fallback_rate,
        })

        bls = bank_greedy_ls(trie, M=args.M, D=args.D, B=B, C=C)
        bls_cov = evaluate_coverage(trie, trace, bls.selected, args.D)
        rows.append({
            "algorithm": "BankGreedy-LS",
            "B": B,
            "C": C,
            "value": bls.value,
            "selected_nodes": len(bls.selected),
            "bank_feasible": bls.feasible,
            "total_overload": 0.0 if bls.feasible else None,
            "max_load": max(bls.bank_loads.values()) if bls.bank_loads else 0.0,
            "certified_coverage": bls_cov.certified_coverage,
            "fallback_rate": bls_cov.fallback_rate,
        })

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
