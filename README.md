# Certified Bank-Aware Fast-Path Coverage for NDN Forwarding Lookup

Research code and reproducibility artifacts for **Certified Bank-Aware Fast-Path Coverage for
Hardware-Constrained NDN Forwarding Lookup**.

The project studies safe fast-path coverage for Named Data Networking (NDN) longest-prefix lookup
under bounded parser depth, prefix-closed trie memory, abstract bank constraints, and a shared
prefix/certificate budget. The proposed BPC family jointly values selected trie state and
same-action shortcut certificates while sending uncertain traffic to an exact fallback path.

## Main ideas

- prefix-closed parser-visible trie placement;
- exact memory-only `PCT-DP` for the tree-closure subproblem;
- bank-aware exact-prefix baselines and local repair;
- certificate-aware `BPC-Greedy` (legacy code label: `BPC-Repair`);
- `BPC-Portfolio`, combining certificate-aware, exact-heavy, repair-based, and shallow-anchor biases;
- trace replay with exact/certificate/fallback classification.

For fixed selected prefix state and unit certificate cost, eligible certificate classes are disjoint,
so the best fixed-state certificate set is obtained by top-k pair weight. The difficult coupling is that
changing selected trie state changes the deepest hardware match and therefore changes the certificate
opportunity set.

## Validation status

The cleaned snapshot was checked with:

```bash
PYTHONPATH=. python -m pytest -q
```

Result:

```text
13 passed
```

The repository also preserves the extracted summary/evidence from the 50k real-name replay under:

```text
outputs/v044_real_trace_parallel_pruned_50k/
```

For object-equivalent budgets `M = 16, 32, 48`, the archived run reports best exact-only
BankRepairOnly coverage of **23.36%, 36.45%, 46.48%**, while BPC-Portfolio reaches
**39.45%, 55.26%, 64.06%**, respectively. These are modeled coverage results under the paper's
stated oracle-style guard and abstract bank/certificate-cost assumptions; they are not synthesized
latency, area, energy, or throughput measurements.

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run tests

```bash
PYTHONPATH=. python -m pytest -q
```

PowerShell:

```powershell
$env:PYTHONPATH = "."
python -m pytest -q
```

## Key experiment entry points

### BPC portfolio

```bash
python -m experiments.run_v041_portfolio_lite --outdir outputs/v041_portfolio_lite
```

### Certificate-cost sensitivity

```bash
python -m experiments.run_v042_cert_cost_sensitivity   --outdir outputs/v042_cert_cost_micro --mode micro
```

### Trace-replay validation

```bash
python -m experiments.run_v043_trace_replay_validation   --interests data/example_interests.csv   --fib data/example_fib.csv   --outdir outputs/v043_example   --D 8   --budgets 8,16,24   --B-values 1,2   --gammas 1.0
```

### Fast parallel real-trace sweep

See `RUN_BIG_REAL_TRACE_FAST.md` and:

```bash
python -m experiments.run_v044_trace_replay_parallel --help
```

## Repository layout

```text
ndnhw/          core trie, placement, bank, certificate and workload logic
experiments/    sweep, replay, plotting and validation drivers
tests/          unit/regression tests
data/           small examples and preprocessed FIB inputs
outputs/        archived compact result/evidence files
tools/          auxiliary tooling
docs/legacy/    original working-snapshot documentation
```

## Important modeling boundaries

The current evaluation is deliberately an offline design-space characterization.

- The evaluator uses known true LPM as an oracle-style target-class guard.
- Certificate-table/guard bank traffic is not modeled in the archived bank abstraction.
- `gamma = 1` is an object-equivalent reference normalization, not measured equality of silicon cost.
- There is no RTL synthesis, place-and-route, calibrated latency, area, power, or throughput result.

See the extended preprint for the exact semantic and architectural scope.

## Large trace file

The original working snapshot contained `data/real_interests_multi.csv` (~109.7 MB). It is excluded
from this GitHub-ready package because it exceeds GitHub's ordinary 100 MiB per-file limit. The
compact 50k replay summaries are retained. See `data/README.md` for details.

## arXiv vs GitHub

Git/GitHub is not required for arXiv. This repository is the research-code artifact; the arXiv upload
should remain a separate TeX/figure source package.

## License

No software license is selected in this snapshot. Add a license before advertising the repository as
open-source software.
