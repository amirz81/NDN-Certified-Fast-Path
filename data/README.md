# Data

Small example/preprocessed files needed for smoke and validation paths are included in this repository.

The original working snapshot also contained `real_interests_multi.csv` (~109.7 MB). It is intentionally
excluded from the GitHub-ready repository because it exceeds GitHub's normal 100 MiB per-file limit and
is derived from the public NDN traffic-trace workflow rather than being source code.

The extended preprint uses public NDN testbed names from the TNTech-NGIN dataset under a reproducible
synthetic forwarding-action snapshot. See the paper and `docs/legacy/TRACE_REPLAY_VALIDATION.md` for
the trace-replay methodology. The 50k-run summary/evidence files are retained under:

`outputs/v044_real_trace_parallel_pruned_50k/`
