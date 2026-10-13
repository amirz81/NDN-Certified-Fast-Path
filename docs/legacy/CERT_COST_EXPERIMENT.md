# v0.4.2 Certificate-Cost Sensitivity

Run micro:

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 -m experiments.run_v042_cert_cost_sensitivity --outdir outputs/v042_cert_cost_micro --mode micro
```

Run lite/heavier:

```powershell
$env:PYTHONPATH=(Get-Location).Path
py -3.12 -m experiments.run_v042_cert_cost_sensitivity --outdir outputs/v042_cert_cost_lite --mode lite
```

Outputs:

```text
cert_cost_sensitivity_all.csv
cert_cost_sensitivity_summary.csv
cert_cost_sensitivity_source_counts.csv
cert_cost_key_B2_M32.csv
plot_gamma_coverage_fixed_per_bank_B2_M32.png
plot_gamma_coverage_fixed_total_B2_M32.png
plot_gamma_selected_cert_budget_B2_M32.png
plot_runtime_by_algorithm.png
```
