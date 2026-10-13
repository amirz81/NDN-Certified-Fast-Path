# v0.4.3 Trace-Replay Validation Layer

This adds a validation path for external NDN Interest-name traces and optional FIB/action files.

## Why this is the extra validation layer

The current paper has strong synthetic/structured-synthetic evidence. A realistic trace-replay path directly addresses the reviewer question:

> Does the framework survive outside your generator?

NIST/ACM ICN 2023 describes an openly available non-synthetic NDN traffic trace dataset captured directly from official NDN testbed routers. Mini-NDN is a practical fallback for emulation-generated traces.

## Input: Interest names

CSV columns accepted:

```text
name
interest
interest_name
ndn_name
uri
```

Or one name per line.

## Input: optional FIB

CSV columns accepted:

```text
prefix
name
fib_prefix
route
uri
```

Optional action/next-hop columns:

```text
action
face
faces
nexthop
next_hop
next_hops
```

If no FIB is provided, the script derives FIB prefixes by truncating Interest names to `--derive-fib-depth`.

## Example

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 -m experiments.run_v043_trace_replay_validation `
  --interests data/example_interests.csv `
  --fib data/example_fib.csv `
  --outdir outputs/v043_example `
  --D 8 `
  --budgets 8,16,24 `
  --B-values 1,2 `
  --gammas 1.0
```

## Real/testbed trace run

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 -m experiments.run_v043_trace_replay_validation `
  --interests data/real_interests.csv `
  --fib data/real_fib.csv `
  --outdir outputs/v043_real_trace `
  --D 8 `
  --budgets 16,24,32,48 `
  --B-values 1,2,4 `
  --gammas 1.0
```

Without FIB:

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 -m experiments.run_v043_trace_replay_validation `
  --interests data/real_interests.csv `
  --outdir outputs/v043_real_trace_derived_fib `
  --derive-fib-depth 3 `
  --min-fib-count 2 `
  --D 8 `
  --budgets 16,24,32,48 `
  --B-values 1,2,4 `
  --gammas 1.0
```

## Outputs

```text
trace_replay_metadata.json
trace_replay_all.csv
trace_replay_summary.csv
trace_replay_source_counts.csv
trace_replay_summary.json
plot_trace_replay_*.png
```
