# Repository provenance

This public package is derived from the user-supplied working snapshot:

`ndn_hw_v041_work.zip`

Scientific Python source, tests, experiment drivers, and compact archived evidence are preserved.
Public cleanup is limited to repository hygiene/documentation:

- removed Python bytecode/cache directories;
- excluded `data/real_interests_multi.csv` because the file is ~109.7 MB and exceeds GitHub's
  ordinary 100 MiB per-file limit;
- removed redundant nested ZIP copies of already-extracted output directories;
- replaced the stale root README with a paper-facing public README while preserving the original
  documentation under `docs/legacy/`.

No algorithmic source edits were required to obtain the reported 13/13 passing tests.
