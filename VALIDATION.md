# Validation

Clean GitHub-ready snapshot validation:

- `PYTHONPATH=. python -m pytest -q`
- result: **13 passed**
- no API keys/secrets detected by a basic repository text scan
- the >100 MiB derived local trace was excluded from the public package
- compact archived 50k replay evidence remains under `outputs/v044_real_trace_parallel_pruned_50k/`

Archived coverage checks from `parallel_trace_all.csv`:

| Budget M | BankRepairOnly | BPC-Portfolio |
|---:|---:|---:|
| 16 | 23.36% | 39.45% |
| 32 | 36.45% | 55.26% |
| 48 | 46.48% | 64.06% |

These values are modeled certified coverage under the paper's stated assumptions.
