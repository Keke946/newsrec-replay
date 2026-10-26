# v1 delivery checklist

| Item | State |
|---|---|
| Scope and selected/comparator policies frozen before final | Complete; local tag v1.0.0-rc1 |
| Snapshot implementation, corruption fallback and atomicity checks | Complete |
| Real-log full-state/next-response snapshot check | Complete;22158-batch prefix and103 next requests |
| Scoped release acceptance |102 passed,0 failures/errors/skips |
| Once-only locked final timeline evaluation | Complete;114477 requests, unchanged selected policy |
| Independent source/cohort/schedule/model/reuse verification | Complete |
| Synthetic no-data demo | Complete; not a quality benchmark |
| Public-safe source, license, reproduction, results and interview material | Prepared and independently checked from staged source |
| Full local development history and private evidence | Delivered separately; not included in public ZIP |
| GitHub publication / remote CI | Target: Keke946/newsrec-replay; remote publication and CI status must be verified on GitHub |
| Personal teaching / interview rehearsal | Deferred until user finishes midterm |

The public ZIP is an initial publication snapshot, not a fabricated history of prior GitHub uploads. The private Git bundle retains actual local commit history and failed experiments. The owner approved creation of Keke946/newsrec-replay. Publish the reviewed public snapshot and use honest future commits. Do not push the private operational repository or its data/artifacts into an unrelated remote.

For a new, empty repository selected by the owner, extract the public ZIP, inspect it, initialize Git, make a real current commit, add that exact repository as origin, and push. For an existing repository, inspect its state and use a dedicated branch/PR; do not overwrite unrelated files or history. No credentials are packaged.
