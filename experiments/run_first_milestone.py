from __future__ import annotations

import argparse
import json
from pathlib import Path

from ndnhw.trace import SyntheticTraceConfig, generate_synthetic_trace
from ndnhw.metrics import parser_depth_coverage, evaluate_coverage
from ndnhw.pct_dp import pct_dp
from ndnhw.baselines import topk_closure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/first_milestone.json")
    parser.add_argument("--n-interests", type=int, default=20_000)
    parser.add_argument("--n-prefixes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = SyntheticTraceConfig(
        n_interests=args.n_interests,
        n_prefixes=args.n_prefixes,
        seed=args.seed,
    )
    trace, trie = generate_synthetic_trace(cfg)

    parser_sweep = {
        D: parser_depth_coverage(trace, D)
        for D in range(1, cfg.max_depth + 4)
    }

    D = 6
    budgets = [16, 32, 64, 128, 256]
    memory_sweep = []

    for M in budgets:
        dp = pct_dp(trie, M=M, D=D)
        topk = topk_closure(trie, M=M, D=D)

        dp_report = evaluate_coverage(trie, trace, dp.selected, D)
        topk_report = evaluate_coverage(trie, trace, topk.selected, D)

        memory_sweep.append({
            "M": M,
            "D": D,
            "PCT-DP": {
                "value": dp.value,
                "selected_nodes": len(dp.selected),
                **dp_report.as_dict(),
            },
            "TopK-Closure": {
                "value": topk.value,
                "selected_nodes": len(topk.selected),
                **topk_report.as_dict(),
            },
        })

    result = {
        "config": cfg.__dict__,
        "trace_size": len(trace),
        "trie_nodes": len(trie),
        "fib_prefixes": sum(1 for node in trie.nodes if node.is_fib_prefix),
        "parser_sweep": parser_sweep,
        "memory_sweep": memory_sweep,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
