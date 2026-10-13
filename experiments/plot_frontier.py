from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="outputs/frontier.csv")
    parser.add_argument("--out", type=str, default="outputs/frontier.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    plt.figure()
    for alg, group in df.groupby("algorithm"):
        sub = group[group["D"] == group["D"].max()]
        plt.plot(sub["M"], sub["certified_coverage"], marker="o", label=alg)
    plt.xlabel("Memory budget M")
    plt.ylabel("Certified safe coverage")
    plt.title("Memory sweep at maximum parser depth")
    plt.legend()
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
