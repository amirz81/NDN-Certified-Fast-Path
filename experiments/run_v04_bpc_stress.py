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
from ndnhw.metrics import evaluate_coverage
from ndnhw.certificates import evaluate_certificate_aware_coverage
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.banking import greedy_load_balance


def build_certificate_stress_trace(
    n_groups: int = 18,
    children_per_group: int = 5,
    interests_per_child: int = 18,
    seed: int = 7,
) -> tuple[InterestTrace, PrefixTrie, dict[int, int]]:
    rng = random.Random(seed)
    trie = PrefixTrie()
    action_ids: dict[int, int] = {}
    names = []

    for g in range(n_groups):
        parent = f"/svc{g}/topic"
        parent_id = trie.add_fib_prefix(parent, weight=0.0, access_load=0.0)
        action_ids[parent_id] = g

        for c in range(children_per_group):
            child = f"{parent}/sub{c}"
            w = max(3, interests_per_child + rng.randint(-4, 4))
            child_id = trie.add_fib_prefix(child, weight=float(w), access_load=float(w))

            # Most children are action-equivalent, last child is a true override.
            action_ids[child_id] = g if c < children_per_group - 1 else 10000 + g

            for i in range(w):
                names.append(NDNName.parse(f"{child}/item{i % 7}/seg{i}"))

    counts = Counter()
    for name in names:
        lpm = trie.lpm(name)
        if lpm is not None:
            counts[lpm] += 1
    for node in trie.nodes:
        node.fib_weight = 0.0
        node.access_load = 0.0
    for node_id, count in counts.items():
        trie.nodes[node_id].fib_weight = float(count)
        trie.nodes[node_id].access_load = float(count)

    return InterestTrace(names), trie, action_ids


def select_parents_only(trie: PrefixTrie, M: int, D: int) -> set[int]:
    selected = set()
    parents = [
        node.node_id for node in trie.iter_nodes()
        if node.depth == 2 and node.is_fib_prefix and node.depth <= D
    ]
    for node_id in parents:
        closure = trie.closure({node_id}, include_root=False)
        trial = selected | closure
        if len(trial) <= M:
            selected = trial
    return selected


def select_children_greedy(trie: PrefixTrie, M: int, D: int) -> set[int]:
    selected = set()
    children = [
        node for node in trie.iter_nodes()
        if node.is_fib_prefix and node.depth == 3 and node.depth <= D
    ]
    children.sort(key=lambda n: n.fib_weight, reverse=True)
    for node in children:
        closure = trie.closure({node.node_id}, include_root=False)
        trial = selected | closure
        if len(trial) <= M:
            selected = trial
    return selected


def eval_selected(trie, trace, selected, D, B, C, action_ids, cert_budget):
    assignment, bank_report = greedy_load_balance(trie, selected, B=B, C=C)
    crep = evaluate_certificate_aware_coverage(
        trie, trace, selected, D, action_ids=action_ids, max_certificates=cert_budget
    )
    return bank_report, crep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/v04_bpc_stress")
    parser.add_argument("--n-groups", type=int, default=18)
    parser.add_argument("--children-per-group", type=int, default=5)
    parser.add_argument("--interests-per-child", type=int, default=18)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    seeds = [3, 7, 11]
    total_budgets = [24, 32, 48, 64]
    D = 8

    for seed in seeds:
        trace, trie, action_ids = build_certificate_stress_trace(
            n_groups=args.n_groups,
            children_per_group=args.children_per_group,
            interests_per_child=args.interests_per_child,
            seed=seed,
        )
        total_load = sum(node.access_load for node in trie.nodes)

        for regime in ["fixed_total", "fixed_per_bank"]:
            for B in [1, 2, 4]:
                C = total_load * 0.35 / B if regime == "fixed_total" else total_load * 0.35

                for total_budget in total_budgets:
                    # Baseline 1: exact child storage, certificates optional but usually less helpful.
                    exact_selected = select_children_greedy(trie, M=total_budget, D=D)
                    for cert_budget in [0, 8, 16, 32]:
                        if cert_budget > total_budget:
                            continue
                        bank_report, crep = eval_selected(
                            trie, trace, exact_selected, D, B, C, action_ids, cert_budget
                        )
                        rows.append({
                            "seed": seed,
                            "regime": regime,
                            "B": B,
                            "C": C,
                            "total_budget": total_budget,
                            "algorithm": "ChildExactGreedy+Cert",
                            "prefix_nodes": len(exact_selected),
                            "cert_budget": cert_budget,
                            "bank_feasible": bank_report.feasible,
                            "total_overload": bank_report.total_overload,
                            **crep.as_dict(),
                        })

                    # Baseline 2: parent shortcut + certificates.
                    for cert_budget in [0, 8, 16, 32]:
                        if cert_budget > total_budget:
                            continue
                        prefix_budget = total_budget - cert_budget
                        parent_selected = select_parents_only(trie, M=prefix_budget, D=D)
                        bank_report, crep = eval_selected(
                            trie, trace, parent_selected, D, B, C, action_ids, cert_budget
                        )
                        rows.append({
                            "seed": seed,
                            "regime": regime,
                            "B": B,
                            "C": C,
                            "total_budget": total_budget,
                            "algorithm": "ParentShortcut+Cert",
                            "prefix_nodes": len(parent_selected),
                            "cert_budget": cert_budget,
                            "bank_feasible": bank_report.feasible,
                            "total_overload": bank_report.total_overload,
                            **crep.as_dict(),
                        })

                    # Baseline 3: bank repair only, no certificates.
                    br = repair_bank_greedy(trie, M=total_budget, D=D, B=B, C=C, max_rounds=6)
                    bank_report, crep = eval_selected(
                        trie, trace, br.selected, D, B, C, action_ids, cert_budget=0
                    )
                    rows.append({
                        "seed": seed,
                        "regime": regime,
                        "B": B,
                        "C": C,
                        "total_budget": total_budget,
                        "algorithm": "BankRepairOnly",
                        "prefix_nodes": len(br.selected),
                        "cert_budget": 0,
                        "bank_feasible": bank_report.feasible,
                        "total_overload": bank_report.total_overload,
                        **crep.as_dict(),
                    })

                    # Proposed: joint prefix-certificate search under bank feasibility.
                    bpc = cert_bank_repair(
                        trie, trace, total_budget=total_budget, D=D, B=B, C=C,
                        action_ids=action_ids,
                        cert_budget_options=[0, 4, 8, 16, 32],
                    )
                    rows.append({
                        "seed": seed,
                        "regime": regime,
                        "B": B,
                        "C": C,
                        "total_budget": total_budget,
                        "algorithm": "BPC-Repair",
                        "prefix_nodes": len(bpc.selected),
                        "cert_budget": bpc.cert_budget,
                        "bank_feasible": bpc.feasible,
                        "total_overload": 0.0 if bpc.feasible else None,
                        "total": len(trace),
                        "parser_accepted": len(trace),
                        "base_certified_hits": int(bpc.base_certified_coverage * len(trace)),
                        "certificate_hits": int(bpc.certificate_gain * len(trace)),
                        "certified_hits": int(bpc.certified_coverage * len(trace)),
                        "naive_hits": int(bpc.naive_coverage * len(trace)),
                        "exact_hits": int(bpc.base_certified_coverage * len(trace)),
                        "fallback": int((1.0 - bpc.certified_coverage) * len(trace)),
                        "unsafe_hits": 0,
                        "certificates_used": bpc.certificates_used,
                        "certificate_candidates": -1,
                        "base_certified_coverage": bpc.base_certified_coverage,
                        "certified_coverage": bpc.certified_coverage,
                        "certificate_gain": bpc.certificate_gain,
                        "fallback_rate": 1.0 - bpc.certified_coverage,
                        "unsafe_rate": 0.0,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "v04_bpc_stress_all.csv", index=False)

    summary = df.groupby(["regime", "B", "total_budget", "algorithm"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        mean_base_certified_coverage=("base_certified_coverage", "mean"),
        mean_certificate_gain=("certificate_gain", "mean"),
        feasible_rate=("bank_feasible", "mean"),
        mean_total_overload=("total_overload", "mean"),
        mean_prefix_nodes=("prefix_nodes", "mean"),
        mean_cert_budget=("cert_budget", "mean"),
    )
    summary.to_csv(outdir / "v04_bpc_stress_summary.csv", index=False)

    # Best feasible rows per regime/budget.
    feasible = summary[summary["feasible_rate"] > 0.999].copy()
    best = feasible.sort_values("mean_certified_coverage", ascending=False)
    best.to_csv(outdir / "v04_bpc_stress_best_feasible.csv", index=False)

    # Plot 1: fixed per-bank, B=2: total budget vs coverage.
    plt.figure()
    sub = summary[(summary["regime"] == "fixed_per_bank") & (summary["B"] == 2)]
    for alg, g in sub.groupby("algorithm"):
        # For baselines with multiple cert budgets, keep best per total budget.
        gg = g.sort_values("mean_certified_coverage").groupby("total_budget", as_index=False).tail(1)
        plt.plot(gg["total_budget"], gg["mean_certified_coverage"], marker="o", label=alg)
    plt.xlabel("Total hardware-object budget")
    plt.ylabel("Mean certified coverage")
    plt.title("v0.4 BPC stress: best-of-both-worlds, fixed per-bank B=2")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "plot_v04_bpc_fixed_per_bank_B2.png", dpi=180)
    plt.close()

    # Plot 2: regimes for BPC only.
    plt.figure()
    sub = summary[summary["algorithm"] == "BPC-Repair"]
    for (regime, B), g in sub.groupby(["regime", "B"]):
        plt.plot(g["total_budget"], g["mean_certified_coverage"], marker="o", label=f"{regime} B={B}")
    plt.xlabel("Total hardware-object budget")
    plt.ylabel("Mean certified coverage")
    plt.title("BPC-Repair under bank regimes")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(outdir / "plot_v04_bpc_regimes.png", dpi=180)
    plt.close()

    compact = {
        "seeds": seeds,
        "D": D,
        "total_budgets": total_budgets,
        "best_feasible_rows": best.head(16).to_dict(orient="records"),
    }
    (outdir / "v04_bpc_stress_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
