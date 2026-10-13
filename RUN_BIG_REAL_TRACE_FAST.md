# v0.4.4 Fast Big Real-Trace Validation

Your `real_interests_multi.csv` has about 266k Interest names. Running every config sequentially is too slow.

This update adds:

- `tools/make_real_fib_multidepth_pruned.py`
- `experiments/run_v044_trace_replay_parallel.py`

## 1. Build a pruned multi-depth FIB

```powershell
cd E:\ndn_hw_realizability_v0_4_3_trace_replay\ndn_hw_v041_work

py -3.12 tools\make_real_fib_multidepth_pruned.py `
  --interests data\real_interests_multi.csv `
  --out data\real_fib_multi_pruned.csv `
  --depths 1,2,3,4 `
  --min-count 20 `
  --max-prefixes 1000 `
  --max-per-depth 300 `
  --action-depth 1
```

If still slow, make it smaller:

```powershell
py -3.12 tools\make_real_fib_multidepth_pruned.py `
  --interests data\real_interests_multi.csv `
  --out data\real_fib_multi_pruned.csv `
  --depths 1,2,3,4 `
  --min-count 50 `
  --max-prefixes 500 `
  --max-per-depth 150 `
  --action-depth 1
```

## 2. Smoke parallel run

```powershell
$env:PYTHONPATH=(Get-Location).Path

py -3.12 -m experiments.run_v044_trace_replay_parallel `
  --interests data\real_interests_multi.csv `
  --fib data\real_fib_multi_pruned.csv `
  --outdir outputs\v044_real_trace_parallel_smoke `
  --D 8 `
  --budgets 16,32 `
  --B-values 1,2 `
  --gammas 1.0 `
  --workers 4 `
  --max-interests-sample 10000
```

Zip:

```powershell
Compress-Archive -Path outputs\v044_real_trace_parallel_smoke -DestinationPath v044_real_trace_parallel_smoke.zip -Force
```

## 3. Paper-grade run

```powershell
$env:PYTHONPATH=(Get-Location).Path

py -3.12 -m experiments.run_v044_trace_replay_parallel `
  --interests data\real_interests_multi.csv `
  --fib data\real_fib_multi_pruned.csv `
  --outdir outputs\v044_real_trace_parallel_pruned_50k `
  --D 8 `
  --budgets 16,32,48 `
  --B-values 1,2,4 `
  --gammas 1.0 `
  --workers 6 `
  --max-interests-sample 50000
```

Zip:

```powershell
Compress-Archive -Path outputs\v044_real_trace_parallel_pruned_50k -DestinationPath v044_real_trace_parallel_pruned_50k.zip -Force
```

## Notes

- `--workers 6` means 6 Python processes.
- Use `--workers 4` if RAM gets tight.
- `--max-interests-sample 50000` is already enough for paper validation.
- Use `--max-interests-sample 0` only if attempting the full 266k.
- Chunk CSVs are saved as workers finish, so partial progress is not lost.
