from __future__ import annotations

import argparse
from pathlib import Path

from ndnhw.trace import SyntheticTraceConfig, generate_synthetic_trace
from ndnhw.frontier import sweep_dm_frontier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="outputs/frontier.csv")
    parser.add_argument("--n-interests", type=int, default=20_000)
    parser.add_argument("--n-prefixes", type=int, default=512)
    args = parser.parse_args()

    trace, trie = generate_synthetic_trace(SyntheticTraceConfig(
        n_interests=args.n_interests,
        n_prefixes=args.n_prefixes,
    ))

    df = sweep_dm_frontier(
        trie=trie,
        trace=trace,
        D_values=[2, 3, 4, 5, 6, 7, 8],
        M_values=[16, 32, 64, 128, 256],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
