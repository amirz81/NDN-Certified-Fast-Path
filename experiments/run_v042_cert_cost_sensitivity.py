from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from experiments.run_v04_bpc_lite import build_certificate_stress_trace, eval_row
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.cert_bank_portfolio import cert_bank_portfolio


def cert_budget_options(total_budget: int, gamma: float) -> list[int]:
    raw = [0, 2, 4, 8, 12, 16, 24, 32]
    return [k for k in raw if gamma * k <= total_budget]


def emit_row(
    *,
    seed,
    regime,
    B,
    C,
    total_budget,
    gamma,
    algorithm,
    source,
    prefix_nodes,
    cert_budget,
    bank_feasible,
    total_overload,
    report,
    runtime_s,
):
    return {
        "seed": seed,
        "regime": regime,
        "B": B,
        "C": C,
        "total_budget": total_budget,
        "gamma": gamma,
        "algorithm": algorithm,
        "source": source,
        "prefix_nodes": prefix_nodes,
        "cert_budget": cert_budget,
        "bank_feasible": bank_feasible,
        "total_overload": total_overload,
        "runtime_s": runtime_s,
        **report.as_dict(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/v042_cert_cost")
    parser.add_argument("--mode", choices=["micro", "lite"], default="micro")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "micro":
        seeds = [3, 7]
        B_values = [2]
        budgets = [16, 24, 32]
        gammas = [0.25, 0.5, 1.0, 2.0]
    else:
        seeds = [3, 7, 11]
        B_values = [1, 2, 4]
        budgets = [16, 24, 32, 48]
        gammas = [0.25, 0.5, 1.0, 2.0]

    rows = []
    D = 8

    for seed in seeds:
        trace, trie, action_ids = build_certificate_stress_trace(seed=seed)
        total_load = sum(n.access_load for n in trie.nodes)

        for regime in ["fixed_per_bank", "fixed_total"]:
            for B in B_values:
                C = total_load * 0.35 if regime == "fixed_per_bank" else total_load * 0.35 / B

                for total_budget in budgets:
                    t0 = time.time()
                    bank = repair_bank_greedy(trie, M=total_budget, D=D, B=B, C=C, max_rounds=3)
                    bank_runtime = time.time() - t0
                    br, cr = eval_row(trie, trace, bank.selected, D, B, C, action_ids, 0)

                    for gamma in gammas:
                        rows.append(emit_row(
                            seed=seed, regime=regime, B=B, C=C,
                            total_budget=total_budget, gamma=gamma,
                            algorithm="BankRepairOnly", source="BankRepairOnly",
                            prefix_nodes=len(bank.selected), cert_budget=0,
                            bank_feasible=br.feasible, total_overload=br.total_overload,
                            report=cr, runtime_s=bank_runtime,
                        ))

                        opts = cert_budget_options(total_budget, gamma)

                        t0 = time.time()
                        bpc = cert_bank_repair(
                            trie, trace, total_budget=total_budget, D=D, B=B, C=C,
                            action_ids=action_ids, cert_budget_options=opts, cert_cost=gamma,
                        )
                        bpc_runtime = time.time() - t0
                        br_bpc, cr_bpc = eval_row(trie, trace, bpc.selected, D, B, C, action_ids, bpc.cert_budget)
                        rows.append(emit_row(
                            seed=seed, regime=regime, B=B, C=C,
                            total_budget=total_budget, gamma=gamma,
                            algorithm="BPC-Repair", source="BPC-Repair",
                            prefix_nodes=len(bpc.selected), cert_budget=bpc.cert_budget,
                            bank_feasible=br_bpc.feasible, total_overload=br_bpc.total_overload,
                            report=cr_bpc, runtime_s=bpc_runtime,
                        ))

                        t0 = time.time()
                        port = cert_bank_portfolio(
                            trie, trace, total_budget=total_budget, D=D, B=B, C=C,
                            action_ids=action_ids, cert_budget_options=opts, cert_cost=gamma,
                        )
                        port_runtime = time.time() - t0
                        br_port, cr_port = eval_row(trie, trace, port.selected, D, B, C, action_ids, port.cert_budget)
                        rows.append(emit_row(
                            seed=seed, regime=regime, B=B, C=C,
                            total_budget=total_budget, gamma=gamma,
                            algorithm="BPC-Portfolio", source=port.source,
                            prefix_nodes=len(port.selected), cert_budget=port.cert_budget,
                            bank_feasible=br_port.feasible, total_overload=br_port.total_overload,
                            report=cr_port, runtime_s=port_runtime,
                        ))

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cert_cost_sensitivity_all.csv", index=False)

    summary = df.groupby(["regime", "B", "total_budget", "gamma", "algorithm"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        std_certified_coverage=("certified_coverage", "std"),
        mean_certificate_gain=("certificate_gain", "mean"),
        feasible_rate=("bank_feasible", "mean"),
        mean_cert_budget=("cert_budget", "mean"),
        mean_prefix_nodes=("prefix_nodes", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    summary.to_csv(outdir / "cert_cost_sensitivity_summary.csv", index=False)

    source_counts = df[df["algorithm"] == "BPC-Portfolio"].groupby(
        ["regime", "B", "total_budget", "gamma", "source"]
    ).size().reset_index(name="count")
    source_counts.to_csv(outdir / "cert_cost_sensitivity_source_counts.csv", index=False)

    for regime in ["fixed_per_bank", "fixed_total"]:
        sub = summary[(summary["regime"] == regime) & (summary["B"] == 2) & (summary["total_budget"] == 32)]
        if len(sub):
            plt.figure()
            for alg, g in sub.groupby("algorithm"):
                g = g.sort_values("gamma")
                plt.plot(g["gamma"], g["mean_certified_coverage"], marker="o", label=alg)
            plt.xlabel("Certificate cost gamma")
            plt.ylabel("Mean certified coverage")
            plt.title(f"Certificate-cost sensitivity, {regime}, B=2, budget=32")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"plot_gamma_coverage_{regime}_B2_M32.png", dpi=220)
            plt.close()

    sub = summary[(summary["algorithm"] == "BPC-Portfolio") & (summary["B"] == 2) & (summary["total_budget"] == 32)]
    if len(sub):
        plt.figure()
        for regime, g in sub.groupby("regime"):
            g = g.sort_values("gamma")
            plt.plot(g["gamma"], g["mean_cert_budget"], marker="o", label=regime)
        plt.xlabel("Certificate cost gamma")
        plt.ylabel("Mean selected certificate budget")
        plt.title("BPC-Portfolio adapts certificate budget to certificate cost")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "plot_gamma_selected_cert_budget_B2_M32.png", dpi=220)
        plt.close()

    rt = summary.groupby(["algorithm"], as_index=False)["mean_runtime_s"].mean()
    plt.figure()
    plt.bar(rt["algorithm"], rt["mean_runtime_s"])
    plt.ylabel("Mean runtime per instance (s)")
    plt.title("Runtime comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "plot_runtime_by_algorithm.png", dpi=220)
    plt.close()

    key = summary[(summary["B"] == 2) & (summary["total_budget"] == 32)].copy()
    pivot = key.pivot_table(
        index=["regime", "gamma"], columns="algorithm",
        values="mean_certified_coverage", aggfunc="first"
    ).reset_index()
    if "BPC-Portfolio" in pivot and "BankRepairOnly" in pivot:
        pivot["portfolio_minus_bankrepair"] = pivot["BPC-Portfolio"] - pivot["BankRepairOnly"]
    pivot.to_csv(outdir / "cert_cost_key_B2_M32.csv", index=False)

    compact = {
        "mode": args.mode,
        "seeds": seeds,
        "B_values": B_values,
        "budgets": budgets,
        "gammas": gammas,
        "key_B2_M32": pivot.to_dict(orient="records"),
        "portfolio_source_counts_total": (
            df[df["algorithm"] == "BPC-Portfolio"].groupby("source").size().reset_index(name="count").to_dict(orient="records")
        ),
        "runtime_mean_by_algorithm": rt.to_dict(orient="records"),
    }
    (outdir / "cert_cost_sensitivity_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
