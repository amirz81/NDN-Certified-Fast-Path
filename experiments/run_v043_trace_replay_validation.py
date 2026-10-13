from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

from ndnhw.name import NDNName
from ndnhw.trie import PrefixTrie
from ndnhw.trace import InterestTrace
from ndnhw.bank_repair import repair_bank_greedy
from ndnhw.cert_bank_repair import cert_bank_repair
from ndnhw.cert_bank_portfolio import cert_bank_portfolio
from experiments.run_v04_bpc_lite import eval_row


def normalize_name(s: str) -> str:
    s = str(s).strip().strip('"').strip("'")
    if " " in s:
        for tok in s.split():
            if tok.startswith("/"):
                s = tok
                break
    if not s:
        return "/"
    if not s.startswith("/"):
        s = "/" + s
    while "//" in s:
        s = s.replace("//", "/")
    return s.split("?")[0].split("#")[0]


def first_col(row: dict, names: list[str]) -> str:
    lower = {k.lower(): k for k in row.keys()}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return next(iter(row.keys()))


def read_interests(path: Path, limit: int | None = None) -> list[NDNName]:
    names = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        first = sample.splitlines()[0].lower() if sample.splitlines() else ""
        has_header = any(x in first for x in ["name", "interest", "uri"])
        if path.suffix.lower() == ".csv" or has_header:
            reader = csv.DictReader(f)
            for row in reader:
                col = first_col(row, ["name", "interest", "interest_name", "ndn_name", "uri"])
                nm = normalize_name(row[col])
                if nm != "/":
                    names.append(NDNName.parse(nm))
                if limit and len(names) >= limit:
                    break
        else:
            for line in f:
                nm = normalize_name(line)
                if nm != "/":
                    names.append(NDNName.parse(nm))
                if limit and len(names) >= limit:
                    break
    return names


def read_fib(path: Path) -> tuple[list[str], dict[str, str]]:
    prefixes, actions = [], {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        first = sample.splitlines()[0].lower() if sample.splitlines() else ""
        has_header = any(x in first for x in ["prefix", "name", "action", "face", "nexthop"])
        if path.suffix.lower() == ".csv" or has_header:
            reader = csv.DictReader(f)
            for row in reader:
                pcol = first_col(row, ["prefix", "name", "fib_prefix", "route", "uri"])
                acol = None
                for c in ["action", "face", "faces", "nexthop", "next_hop", "next_hops"]:
                    if c in {k.lower(): k for k in row.keys()}:
                        acol = {k.lower(): k for k in row.keys()}[c]
                        break
                p = normalize_name(row[pcol])
                if p != "/":
                    prefixes.append(p)
                    actions[p] = str(row[acol]) if acol and row.get(acol) else NDNName.parse(p).components[0]
        else:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                p = normalize_name(parts[0])
                if p != "/":
                    prefixes.append(p)
                    actions[p] = parts[1] if len(parts) > 1 else NDNName.parse(p).components[0]
    return prefixes, actions


def prefix_at_depth(name: NDNName, depth: int) -> str:
    return str(NDNName.parse(name.components[:max(1, min(depth, name.depth))]))


def derive_fib(names: list[NDNName], depth: int, min_count: int) -> tuple[list[str], dict[str, str]]:
    counts = Counter(prefix_at_depth(n, depth) for n in names if n.depth)
    prefixes = [p for p, c in counts.items() if c >= min_count and p != "/"]
    actions = {p: (NDNName.parse(p).components[0] if NDNName.parse(p).components else "root") for p in prefixes}
    return prefixes, actions


def build_trace_trie(names: list[NDNName], prefixes: list[str], actions: dict[str, str]):
    trie = PrefixTrie()
    action_map, action_ids = {}, {}
    for p in sorted(set(prefixes), key=lambda x: (NDNName.parse(x).depth, x)):
        nid = trie.add_fib_prefix(p, weight=0.0, access_load=0.0)
        act = str(actions.get(p, NDNName.parse(p).components[0]))
        action_map.setdefault(act, len(action_map))
        action_ids[nid] = action_map[act]
    for n in names:
        trie.insert_path(n)
    counts, kept = Counter(), []
    for n in names:
        lpm = trie.lpm(n)
        if lpm is not None:
            counts[lpm] += 1
            kept.append(n)
    for node in trie.nodes:
        node.fib_weight = 0.0
        node.access_load = 0.0
    for nid, c in counts.items():
        trie.nodes[nid].fib_weight = float(c)
        trie.nodes[nid].access_load = float(c)
    return InterestTrace(kept), trie, action_ids


def cert_options(total_budget: int, gamma: float):
    return [k for k in [0, 2, 4, 8, 12, 16, 24, 32, 48] if gamma * k <= total_budget]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interests", required=True)
    parser.add_argument("--fib", default=None)
    parser.add_argument("--outdir", default="outputs/v043_trace_replay")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--derive-fib-depth", type=int, default=3)
    parser.add_argument("--min-fib-count", type=int, default=1)
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--budgets", default="16,24,32,48")
    parser.add_argument("--B-values", default="1,2,4")
    parser.add_argument("--gammas", default="1.0")
    parser.add_argument("--regimes", default="fixed_per_bank,fixed_total")
    parser.add_argument("--capacity-frac", type=float, default=0.35)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    names = read_interests(Path(args.interests), args.limit)
    if args.fib:
        prefixes, actions = read_fib(Path(args.fib))
    else:
        prefixes, actions = derive_fib(names, args.derive_fib_depth, args.min_fib_count)

    trace, trie, action_ids = build_trace_trie(names, prefixes, actions)
    total_load = sum(n.access_load for n in trie.nodes)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    B_values = [int(x) for x in args.B_values.split(",") if x.strip()]
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]

    meta = {
        "loaded_interests": len(names),
        "usable_interests_with_lpm": len(trace),
        "fib_prefixes": sum(1 for n in trie.nodes if n.is_fib_prefix),
        "trie_nodes": len(trie),
        "total_load": total_load,
        "D": args.D,
        "budgets": budgets,
        "B_values": B_values,
        "gammas": gammas,
        "regimes": regimes,
    }
    (outdir / "trace_replay_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rows = []
    for regime in regimes:
        for B in B_values:
            C = total_load * args.capacity_frac if regime == "fixed_per_bank" else total_load * args.capacity_frac / B
            for M in budgets:
                for gamma in gammas:
                    t0 = time.time()
                    bank = repair_bank_greedy(trie, M=M, D=args.D, B=B, C=C, max_rounds=3)
                    br, cr = eval_row(trie, trace, bank.selected, args.D, B, C, action_ids, 0)
                    rows.append({"regime": regime, "B": B, "C": C, "total_budget": M, "gamma": gamma,
                                 "algorithm": "BankRepairOnly", "source": "BankRepairOnly",
                                 "runtime_s": time.time() - t0, "prefix_nodes": len(bank.selected),
                                 "cert_budget": 0, "bank_feasible": br.feasible, "total_overload": br.total_overload,
                                 **cr.as_dict()})
                    opts = cert_options(M, gamma)
                    t0 = time.time()
                    bpc = cert_bank_repair(trie, trace, total_budget=M, D=args.D, B=B, C=C,
                                           action_ids=action_ids, cert_budget_options=opts, cert_cost=gamma)
                    br, cr = eval_row(trie, trace, bpc.selected, args.D, B, C, action_ids, bpc.cert_budget)
                    rows.append({"regime": regime, "B": B, "C": C, "total_budget": M, "gamma": gamma,
                                 "algorithm": "BPC-Repair", "source": "BPC-Repair",
                                 "runtime_s": time.time() - t0, "prefix_nodes": len(bpc.selected),
                                 "cert_budget": bpc.cert_budget, "bank_feasible": br.feasible, "total_overload": br.total_overload,
                                 **cr.as_dict()})
                    t0 = time.time()
                    port = cert_bank_portfolio(trie, trace, total_budget=M, D=args.D, B=B, C=C,
                                               action_ids=action_ids, cert_budget_options=opts, cert_cost=gamma)
                    br, cr = eval_row(trie, trace, port.selected, args.D, B, C, action_ids, port.cert_budget)
                    rows.append({"regime": regime, "B": B, "C": C, "total_budget": M, "gamma": gamma,
                                 "algorithm": "BPC-Portfolio", "source": port.source,
                                 "runtime_s": time.time() - t0, "prefix_nodes": len(port.selected),
                                 "cert_budget": port.cert_budget, "bank_feasible": br.feasible, "total_overload": br.total_overload,
                                 **cr.as_dict()})

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "trace_replay_all.csv", index=False)
    summary = df.groupby(["regime", "B", "total_budget", "gamma", "algorithm"], as_index=False).agg(
        mean_certified_coverage=("certified_coverage", "mean"),
        mean_certificate_gain=("certificate_gain", "mean"),
        feasible_rate=("bank_feasible", "mean"),
        mean_cert_budget=("cert_budget", "mean"),
        mean_prefix_nodes=("prefix_nodes", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    )
    summary.to_csv(outdir / "trace_replay_summary.csv", index=False)
    source_counts = df[df["algorithm"] == "BPC-Portfolio"].groupby(
        ["regime", "B", "total_budget", "gamma", "source"]
    ).size().reset_index(name="count")
    source_counts.to_csv(outdir / "trace_replay_source_counts.csv", index=False)

    for regime in regimes:
        for B in B_values:
            sub = summary[(summary["regime"] == regime) & (summary["B"] == B)]
            if sub.empty:
                continue
            plt.figure()
            for alg, g in sub.groupby("algorithm"):
                g = g.sort_values("total_budget")
                plt.plot(g["total_budget"], g["mean_certified_coverage"] * 100, marker="o", label=alg)
            plt.xlabel("Total budget")
            plt.ylabel("Certified coverage (%)")
            plt.title(f"Trace replay: {regime}, B={B}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(outdir / f"plot_trace_replay_{regime}_B{B}.png", dpi=220)
            plt.close()

    compact = {
        "metadata": meta,
        "best_rows": summary.sort_values("mean_certified_coverage", ascending=False).head(12).to_dict(orient="records"),
        "portfolio_source_totals": df[df["algorithm"] == "BPC-Portfolio"].groupby("source").size().reset_index(name="count").to_dict(orient="records"),
    }
    (outdir / "trace_replay_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
