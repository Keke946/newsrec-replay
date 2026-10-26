# NewsRec v1 project map

The supported CPU path is click-data audit → chronological feature/candidate replay → LightGBM LambdaRank → six-hour full-prefix refit policy → immutable model registry → closed-timestamp-batch API → append-only journal → optional JSON/gzip snapshot.

| Layer | Main entry | Responsibility |
|---|---|---|
| Data contracts | `src/newsrec/data.py`, `cli.py` | Read-only input audit, fingerprints, role separation |
| Candidate/features | `time_ranker.py`, `baseline.py` | Trend50 plus recent3, max53; causal15 features |
| Refit experiments | `tools/run_refresh_backtest.py` | Fixed/refresh matched controls, label arrival cutoff, delayed switch |
| Runtime registry | `refresh_replay.py` | Artifact/feature/data identity, activation and fallback |
| Batch API | `refresh_api.py`, `tools/serve_refresh_replay.py` | Explicit opt-in loopback replay; no live training |
| Persistence | `append_refresh_journal.py` | Commit-before-reply, exact retries, pinned historical model choice |
| Snapshot | `state_snapshot.py` | Prefix-bound state, atomic write, tail replay, full-log fallback |
| Acceptance/demo | `tools/check_v1.py`, `tools/demo_v1.py` | Core tests and self-contained synthetic walkthrough |

The request is immediately after a click. Its own click is visible in user history; other events tied at that timestamp are revealed after all queries in the batch. All source clicks advance the query cache even if they do not have a scored label. The next strictly later observed click is the proxy target; equal-timestamp next transitions are excluded. Repeated items remain eligible. Article availability uses supplied publication metadata, with known anomalies.

The selected policy uses no supplied embeddings. Two-tower, mixed-negative, sequence, cold-candidate and ANN branches remain separately documented research. Their existence is not evidence they improve the selected ranker. The rejected dynamic SQLite refresh-journal trials remain available in private evidence; the launcher uses the accepted append backend. The older static service is retained separately.

This is a reproducible event-time evaluation and replay system. Registry models are precomputed. Actual ingestion delays, online impressions/CTR, asynchronous feedback worker and distributed serving are outside v1. All final scores and negative findings are reported in V1_RESULTS.md and V1_LIMITATIONS.md.
