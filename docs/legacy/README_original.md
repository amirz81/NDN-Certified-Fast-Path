# Safe Bank-Aware Hardware Realizability of NDN Forwarding Lookup

This repository implements a trace-driven simulator and algorithmic framework for studying
which NDN forwarding lookup decisions are safely hardware-realizable under parser, memory,
banking, and update constraints.

## Current version

`v0.2-experiment-ready`

Implemented:

- NDN name representation
- Prefix trie construction
- Longest-prefix-match logic
- Synthetic NDN workload generation
- Parser-depth coverage
- Naive / exact / certified FIB coverage
- Prefix-closed trie dynamic programming, `PCT-DP`
- TopK-Closure baseline
- CoverageFirst baseline
- HashBank baseline
- BankGreedy heuristic
- BankGreedy-LS local-search heuristic
- Exact small-instance bank-aware oracle
- Realizability frontier sweep over parser depth and memory
- Bank sweep experiment
- Adversarial example generators
- PIT dynamic feasibility extension
- Tests for core correctness

## Core object

For a workload trace `T` and resource budget `(D, M, B, C, U)`:

```text
R_T(D, M, B, C, U) = maximum certified safe fast-path coverage
```

The framework distinguishes:

- naive prefix hits,
- exact longest-prefix hits,
- certified safe hits,
- fallback decisions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest tests
```

## Run first milestone

```bash
python -m experiments.run_first_milestone --out outputs/first_milestone.json
```

## Run parser/memory frontier

```bash
python -m experiments.run_frontier --out outputs/frontier.csv
python -m experiments.plot_frontier --csv outputs/frontier.csv --out outputs/frontier.png
```

## Run bank sweep

```bash
python -m experiments.run_bank_sweep --out outputs/bank_sweep.csv
```

## Run adversarial cases

```bash
python -m experiments.run_adversarial_cases
```

## Run exact oracle demo

```bash
python -m experiments.run_oracle_demo
```

## Run PIT feasibility demo

```bash
python -m experiments.run_pit_demo
```

## Main algorithms

| Algorithm | Purpose |
|---|---|
| `TopK-Closure` | Popularity baseline with ancestor closure |
| `PCT-DP` | Exact memory-only prefix-closed trie DP |
| `HashBank` | Deterministic hash-to-bank assignment |
| `CoverageFirst` | Optimize prefix coverage first, assign banks second |
| `BankGreedy` | Bank-aware greedy prefix placement |
| `BankGreedy-LS` | BankGreedy plus conservative local search |
| `ExactBankOracle` | Exponential exact solver for small bank-aware instances |
| `PIT feasibility` | Dynamic occupancy/update-budget feasibility |

## Research status

The codebase is ready for controlled synthetic experiments.
The next research upgrade is real-trace ingestion and paper-grade plotting.


## v0.3 dangerous upgrades

Added after the first result analysis:

- certificate-aware safe shortcut model
- synthetic forwarding-action equivalence model
- structured NDN-like trace generator
- two bank-capacity regimes:
  - fixed total bandwidth
  - fixed per-bank bandwidth
- stronger feasible repair heuristic: `BankRepair`
- v0.3 sweep script with parser, certificate, bank-regime, and heatmap outputs

Run:

```bash
python -m experiments.run_v03_sweep --outdir outputs/v03_sweep
```

## v0.4 BPC upgrades

Added joint prefix-certificate placement under bank constraints:

- `ndnhw/cert_bank_repair.py`
- `experiments/run_v04_bpc_stress.py`
- `tests/test_cert_bank_repair.py`

The main proposed algorithm is **BPC-Repair**: Budgeted Prefix-Certificate Repair.
It chooses exact prefixes and certified shortcuts jointly under a total hardware-object budget and bank-feasibility constraints.

Run:

```bash
python -m experiments.run_v04_bpc_stress --outdir outputs/v04_bpc_stress
```


## v0.4.1 portfolio upgrade

Adds `BPC-Portfolio`, a robust certified bank-aware portfolio heuristic that tries:
- original BPC-Repair,
- bank repair plus certificates,
- exact greedy plus certificates,
- shallow shortcut plus certificates,

then returns the best feasible certified placement.

Run:

```bash
python -m experiments.run_v041_portfolio_lite --outdir outputs/v041_portfolio_lite
```


## v0.4.2 Certificate-cost sensitivity

```bash
python -m experiments.run_v042_cert_cost_sensitivity --outdir outputs/v042_cert_cost_micro --mode micro
```


## v0.4.3 Trace-replay validation

```bash
python -m experiments.run_v043_trace_replay_validation --interests data/example_interests.csv --fib data/example_fib.csv --outdir outputs/v043_example --D 8 --budgets 8,16,24 --B-values 1,2 --gammas 1.0
```


## v0.4.4 fast big real trace

See `RUN_BIG_REAL_TRACE_FAST.md`.
