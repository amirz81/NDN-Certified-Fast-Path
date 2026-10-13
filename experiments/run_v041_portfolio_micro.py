from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from experiments.run_v04_bpc_lite import (
    build_certificate_stress_trace,
    select_children_greedy,
    select_parents_only,
    eval_row,
)
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.cert_bank_portfolio import cert_bank_portfolio


def emit_bpc_row(trace, bpc, seed, regime, B, C, total_budget, algorithm):
    return {
        "seed": seed, "regime": regime, "B": B, "C": C,
        "total_budget": total_budget, "algorithm": algorithm,
        "source": getattr(bpc, "source", algorithm),
        "prefix_nodes": len(bpc.selected), "cert_budget": bpc.cert_budget,
        "bank_feasible": bpc.feasible, "total_overload": 0.0,
        "total": len(trace), "parser_accepted": len(trace),
        "base_certified_hits": int(bpc.base_certified_coverage * len(trace)),
        "certificate_hits": int(bpc.certificate_gain * len(trace)),
        "certified_hits": int(bpc.certified_coverage * len(trace)),
        "naive_hits": int(bpc.naive_coverage * len(trace)),
        "exact_hits": int(bpc.base_certified_coverage * len(trace)),
        "fallback": int((1-bpc.certified_coverage)*len(trace)),
        "unsafe_hits": 0,
        "certificates_used": bpc.certificates_used,
        "certificate_candidates": -1,
        "base_certified_coverage": bpc.base_certified_coverage,
        "certified_coverage": bpc.certified_coverage,
        "certificate_gain": bpc.certificate_gain,
        "fallback_rate": 1-bpc.certified_coverage,
        "unsafe_rate": 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/v041_portfolio_micro")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    D = 8
    for seed in [3, 7]:
        trace, trie, action_ids = build_certificate_stress_trace(seed=seed)
        total_load = sum(n.access_load for n in trie.nodes)

        for regime in ["fixed_per_bank", "fixed_total"]:
            B = 2
            C = total_load * 0.35 if regime == "fixed_per_bank" else total_load * 0.35 / B

            for total_budget in [16, 24, 32]:
                # Best of exact/parent baselines over tiny cert splits.
                for cert_budget in [0, 4, 8]:
                    if cert_budget > total_budget:
                        continue
                    prefix_budget = total_budget - cert_budget

                    child = select_children_greedy(trie, prefix_budget, D)
                    br, cr = eval_row(trie, trace, child, D, B, C, action_ids, cert_budget)
                    rows.append({
                        "seed": seed, "regime": regime, "B": B, "C": C,
                        "total_budget": total_budget, "algorithm": "ChildExactGreedy+Cert",
                        "source": "ChildExactGreedy+Cert",
                        "prefix_nodes": len(child), "cert_budget": cert_budget,
                        "bank_feasible": br.feasible, "total_overload": br.total_overload,
                        **cr.as_dict(),
                    })

                    parent = select_parents_only(trie, prefix_budget, D)
                    br, cr = eval_row(trie, trace, parent, D, B, C, action_ids, cert_budget)
                    rows.append({
                        "seed": seed, "regime": regime, "B": B, "C": C,
                        "total_budget": total_budget, "algorithm": "ParentShortcut+Cert",
                        "source": "ParentShortcut+Cert",
                        "prefix_nodes": len(parent), "cert_budget": cert_budget,
                        "bank_feasible": br.feasible, "total_overload": br.total_overload,
                        **cr.as_dict(),
                    })

                bank_only = repair_bank_greedy(trie, M=total_budget, D=D, B=B, C=C, max_rounds=3)
                br, cr = eval_row(trie, trace, bank_only.selected, D, B, C, action_ids, 0)
                rows.append({
                    "seed": seed, "regime": regime, "B": B, "C": C,
                    "total_budget": total_budget, "algorithm": "BankRepairOnly",
                    "source": "BankRepairOnly",
                    "prefix_nodes": len(bank_only.selected), "cert_budget": 0,
                    "bank_feasible": br.feasible, "total_overload": br.total_overload,
                    **cr.as_dict(),
                })

                bpc = cert_bank_repair(
                    trie, trace, total_budget=total_budget, D=D, B=B, C=C,
                    action_ids=action_ids, cert_budget_options=[0, 4, 8]
                )
                rows.append(emit_bpc_row(trace, bpc, seed, regime, B, C, total_budget, "BPC-Repair"))

                portfolio = cert_bank_portfolio(
                    trie, trace, total_budget=total_budget, D=D, B=B, C=C,
                    action_ids=action_ids, cert_budget_options=[0, 4, 8]
                )
                rows.append(emit_bpc_row(trace, portfolio, seed, regime, B, C, total_budget, "BPC-Portfolio"))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "v041_portfolio_micro_all.csv", index=False)

    summary = df.groupby(["regime", "B", "total_budget", "algorithm"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        mean_certificate_gain=("certificate_gain", "mean"),
        feasible_rate=("bank_feasible", "mean"),
        mean_total_overload=("total_overload", "mean"),
        mean_prefix_nodes=("prefix_nodes", "mean"),
        mean_cert_budget=("cert_budget", "mean"),
    )
    summary.to_csv(outdir / "v041_portfolio_micro_summary.csv", index=False)
    best = summary[summary["feasible_rate"] > 0.999].sort_values("mean_certified_coverage", ascending=False)
    best.to_csv(outdir / "v041_portfolio_micro_best_feasible.csv", index=False)

    source_counts = df[df["algorithm"] == "BPC-Portfolio"].groupby(
        ["regime", "B", "total_budget", "source"]
    ).size().reset_index(name="count")
    source_counts.to_csv(outdir / "v041_portfolio_micro_source_counts.csv", index=False)

    plt.figure()
    sub = summary[(summary["regime"] == "fixed_per_bank") & (summary["B"] == 2)]
    for alg, g in sub.groupby("algorithm"):
        g = g.sort_values("total_budget")
        plt.plot(g["total_budget"], g["mean_certified_coverage"], marker="o", label=alg)
    plt.xlabel("Total budget")
    plt.ylabel("Mean certified coverage")
    plt.title("v0.4.1 micro: portfolio vs baselines")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "plot_v041_portfolio_micro.png", dpi=180)
    plt.close()

    compact = {
        "best_feasible_rows": best.head(16).to_dict(orient="records"),
        "fixed_per_bank_B2": summary[(summary["regime"] == "fixed_per_bank") & (summary["B"] == 2)].to_dict(orient="records"),
        "portfolio_source_counts": source_counts.to_dict(orient="records"),
    }
    (outdir / "v041_portfolio_micro_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
