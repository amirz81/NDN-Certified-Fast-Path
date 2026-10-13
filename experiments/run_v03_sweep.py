from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from ndnhw.structured_trace import StructuredTraceConfig, generate_structured_trace
from ndnhw.trace import SyntheticTraceConfig, generate_synthetic_trace
from ndnhw.pct_dp import pct_dp
from ndnhw.baselines import topk_closure, coverage_first
from ndnhw.bank_greedy import bank_greedy_ls
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.banking import hashbank
from ndnhw.metrics import evaluate_coverage, parser_depth_coverage
from ndnhw.certificates import deterministic_action_ids, evaluate_certificate_aware_coverage


def make_workload(kind: str, seed: int, alpha: float, n_interests: int, n_prefixes: int):
    if kind == "structured":
        return generate_structured_trace(StructuredTraceConfig(
            n_interests=n_interests,
            n_prefixes=n_prefixes,
            seed=seed,
            zipf_alpha=alpha,
        ))
    if kind == "random":
        return generate_synthetic_trace(SyntheticTraceConfig(
            n_interests=n_interests,
            n_prefixes=n_prefixes,
            seed=seed,
            zipf_alpha=alpha,
        ))
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/v03_sweep")
    parser.add_argument("--n-interests", type=int, default=6000)
    parser.add_argument("--n-prefixes", type=int, default=160)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seeds = [3, 7, 11]
    alphas = [0.8, 1.1, 1.4]
    kinds = ["random", "structured"]

    parser_rows = []
    cert_rows = []
    bank_rows = []
    heat_rows = []

    for kind in kinds:
        for seed in seeds:
            for alpha in alphas:
                trace, trie = make_workload(kind, seed, alpha, args.n_interests, args.n_prefixes)

                for D in range(1, 12):
                    parser_rows.append({
                        "kind": kind,
                        "seed": seed,
                        "zipf_alpha": alpha,
                        "D": D,
                        "parser_coverage": parser_depth_coverage(trace, D),
                    })

                # Certificate-aware gap recovery at a strong memory/parser point.
                D_cert, M_cert = 8, 256
                dp = pct_dp(trie, M=M_cert, D=D_cert)
                topk = topk_closure(trie, M=M_cert, D=D_cert)
                action_ids = deterministic_action_ids(trie, mode="top_component", num_actions=8, seed=seed)

                for alg, selected in [("PCT-DP", dp.selected), ("TopK-Closure", topk.selected)]:
                    base = evaluate_coverage(trie, trace, selected, D_cert)
                    for K in [0, 4, 8, 16, 32, 64]:
                        crep = evaluate_certificate_aware_coverage(
                            trie, trace, selected, D_cert,
                            action_ids=action_ids,
                            max_certificates=K,
                        )
                        cert_rows.append({
                            "kind": kind,
                            "seed": seed,
                            "zipf_alpha": alpha,
                            "algorithm": alg,
                            "D": D_cert,
                            "M": M_cert,
                            "cert_budget": K,
                            "base_certified_coverage": base.certified_coverage,
                            "naive_coverage": base.naive_coverage,
                            **crep.as_dict(),
                        })

                # Heatmap frontier points for PCT-DP only.
                for D in [4, 6, 8]:
                    for M in [32, 64, 128, 256]:
                        res = pct_dp(trie, M=M, D=D)
                        rep = evaluate_coverage(trie, trace, res.selected, D)
                        heat_rows.append({
                            "kind": kind,
                            "seed": seed,
                            "zipf_alpha": alpha,
                            "D": D,
                            "M": M,
                            "algorithm": "PCT-DP",
                            "certified_coverage": rep.certified_coverage,
                            "naive_coverage": rep.naive_coverage,
                            "unsafe_rate": rep.unsafe_rate,
                        })

                # Bank regimes: fixed total and fixed per-bank.
                D_bank, M_bank = 7, 96
                total_load = sum(node.access_load for node in trie.nodes)
                base_per_bank = total_load * 0.35

                for regime in ["fixed_total", "fixed_per_bank"]:
                    for B in [1, 2, 4, 8]:
                        if regime == "fixed_total":
                            C = (total_load / B) * 0.35
                        else:
                            C = base_per_bank

                        dp = pct_dp(trie, M=M_bank, D=D_bank)
                        _, hash_report = hashbank(trie, dp.selected, B=B, C=C)
                        hash_cov = evaluate_coverage(trie, trace, dp.selected, D_bank)
                        bank_rows.append({
                            "kind": kind,
                            "seed": seed,
                            "zipf_alpha": alpha,
                            "regime": regime,
                            "algorithm": "PCT-DP+HashBank",
                            "B": B,
                            "C": C,
                            "bank_feasible": hash_report.feasible,
                            "total_overload": hash_report.total_overload,
                            "value": dp.value,
                            "certified_coverage": hash_cov.certified_coverage,
                        })

                        cf = coverage_first(trie, M=M_bank, D=D_bank, B=B, C=C)
                        cf_cov = evaluate_coverage(trie, trace, cf.selected, D_bank)
                        bank_rows.append({
                            "kind": kind,
                            "seed": seed,
                            "zipf_alpha": alpha,
                            "regime": regime,
                            "algorithm": "CoverageFirst",
                            "B": B,
                            "C": C,
                            "bank_feasible": cf.bank_report.feasible,
                            "total_overload": cf.bank_report.total_overload,
                            "value": cf.value,
                            "certified_coverage": cf_cov.certified_coverage,
                        })

                        bls = bank_greedy_ls(trie, M=M_bank, D=D_bank, B=B, C=C, max_rounds=3)
                        bls_cov = evaluate_coverage(trie, trace, bls.selected, D_bank)
                        bank_rows.append({
                            "kind": kind,
                            "seed": seed,
                            "zipf_alpha": alpha,
                            "regime": regime,
                            "algorithm": "BankGreedy-LS",
                            "B": B,
                            "C": C,
                            "bank_feasible": bls.feasible,
                            "total_overload": 0.0 if bls.feasible else None,
                            "value": bls.value,
                            "certified_coverage": bls_cov.certified_coverage,
                        })

                        repair = repair_bank_greedy(trie, M=M_bank, D=D_bank, B=B, C=C, max_rounds=6)
                        repair_cov = evaluate_coverage(trie, trace, repair.selected, D_bank)
                        bank_rows.append({
                            "kind": kind,
                            "seed": seed,
                            "zipf_alpha": alpha,
                            "regime": regime,
                            "algorithm": "BankRepair",
                            "B": B,
                            "C": C,
                            "bank_feasible": repair.feasible,
                            "total_overload": 0.0 if repair.feasible else None,
                            "value": repair.value,
                            "certified_coverage": repair_cov.certified_coverage,
                        })

    parser_df = pd.DataFrame(parser_rows)
    cert_df = pd.DataFrame(cert_rows)
    bank_df = pd.DataFrame(bank_rows)
    heat_df = pd.DataFrame(heat_rows)

    parser_df.to_csv(outdir / "parser_all.csv", index=False)
    cert_df.to_csv(outdir / "certificate_all.csv", index=False)
    bank_df.to_csv(outdir / "bank_regimes_all.csv", index=False)
    heat_df.to_csv(outdir / "frontier_heatmap_all.csv", index=False)

    parser_summary = parser_df.groupby(["kind", "zipf_alpha", "D"], as_index=False).agg(
        mean_parser_coverage=("parser_coverage", "mean"),
        std_parser_coverage=("parser_coverage", "std"),
    )
    cert_summary = cert_df.groupby(["kind", "zipf_alpha", "algorithm", "cert_budget"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        mean_base_certified_coverage=("base_certified_coverage", "mean"),
        mean_certificate_gain=("certificate_gain", "mean"),
        mean_naive_coverage=("naive_coverage", "mean"),
    )
    bank_summary = bank_df.groupby(["kind", "zipf_alpha", "regime", "algorithm", "B"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        feasible_rate=("bank_feasible", "mean"),
        mean_total_overload=("total_overload", "mean"),
        mean_value=("value", "mean"),
    )
    heat_summary = heat_df.groupby(["kind", "zipf_alpha", "D", "M"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        mean_naive_coverage=("naive_coverage", "mean"),
        mean_unsafe_rate=("unsafe_rate", "mean"),
    )

    parser_summary.to_csv(outdir / "parser_summary.csv", index=False)
    cert_summary.to_csv(outdir / "certificate_summary.csv", index=False)
    bank_summary.to_csv(outdir / "bank_regimes_summary.csv", index=False)
    heat_summary.to_csv(outdir / "frontier_heatmap_summary.csv", index=False)

    # Plots
    plt.figure()
    for (kind, alpha), g in parser_summary.groupby(["kind", "zipf_alpha"]):
        plt.plot(g["D"], g["mean_parser_coverage"], marker="o", label=f"{kind} a={alpha}")
    plt.xlabel("Parser depth D")
    plt.ylabel("Mean parser coverage")
    plt.title("v0.3 parser-depth coverage")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "plot_parser_v03.png", dpi=180)
    plt.close()

    plt.figure()
    sub = cert_summary[(cert_summary["kind"] == "structured") & (cert_summary["algorithm"] == "PCT-DP")]
    for alpha, g in sub.groupby("zipf_alpha"):
        plt.plot(g["cert_budget"], g["mean_certified_coverage"], marker="o", label=f"alpha={alpha}")
    plt.xlabel("Certificate budget")
    plt.ylabel("Mean certified coverage")
    plt.title("Certificate-aware coverage gain, structured/PCT-DP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "plot_certificate_gain.png", dpi=180)
    plt.close()

    plt.figure()
    sub = bank_summary[(bank_summary["kind"] == "structured") & (bank_summary["zipf_alpha"] == 1.1)]
    for (regime, alg), g in sub.groupby(["regime", "algorithm"]):
        if alg in ["CoverageFirst", "BankGreedy-LS", "BankRepair"]:
            plt.plot(g["B"], g["mean_certified_coverage"], marker="o", label=f"{regime} {alg}")
    plt.xlabel("Bank count B")
    plt.ylabel("Mean certified coverage")
    plt.title("Bank regimes expose different hardware bottlenecks")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "plot_bank_regimes.png", dpi=180)
    plt.close()

    # Heatmaps for structured alpha=1.1
    import numpy as np
    target = heat_summary[(heat_summary["kind"] == "structured") & (heat_summary["zipf_alpha"] == 1.1)]
    pivot = target.pivot(index="D", columns="M", values="mean_certified_coverage")
    plt.figure()
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Mean certified coverage")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Memory M")
    plt.ylabel("Parser depth D")
    plt.title("Realizability frontier heatmap, structured alpha=1.1")
    plt.tight_layout()
    plt.savefig(outdir / "plot_frontier_heatmap_structured_a11.png", dpi=180)
    plt.close()

    summary = {
        "runs": len(kinds) * len(seeds) * len(alphas),
        "kinds": kinds,
        "seeds": seeds,
        "zipf_alphas": alphas,
        "certificate_best": cert_summary.sort_values("mean_certified_coverage", ascending=False).head(12).to_dict(orient="records"),
        "bank_best_feasible": bank_summary[bank_summary["feasible_rate"] > 0].sort_values("mean_certified_coverage", ascending=False).head(12).to_dict(orient="records"),
        "frontier_best": heat_summary.sort_values("mean_certified_coverage", ascending=False).head(12).to_dict(orient="records"),
    }
    (outdir / "v03_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
