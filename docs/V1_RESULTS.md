# v1 results: frozen final evaluation and systems acceptance

All accuracy figures below are offline next-click proxy metrics, not CTR or online A/B results. The refresh policy was selected before final scoring (freeze commit a081cf7, tag v1.0.0-rc1). Final evaluation executed once, with no interval/parameter/sample changes after viewing scores.

|Stage|Requests|Fixed HR5|Refresh HR5|Delta (percentage points)|
|---|---:|---:|---:|---:|
|Three internal development windows|321036|32.8228%|33.0340%|+0.2112|
|Previously explored validation confirmation|114855|33.2942%|33.4004%|+0.1062|
|Locked final timeline|114477|29.3386%|29.3325%|-0.0061|

The final comparison is essentially flat:437 gains,444 losses, net−7;400 user-cluster bootstrap interval [−0.0517,+0.0480] percentage points. Earlier validation gains did not generalize. No positive final accuracy gain is claimed. The selected policy and fixed comparator are both retained; final results are not used to pick a new method or tune cadence.

|Final metric|Fixed|Preselected refresh|
|---|---:|---:|
|mrr@5|16.4053%|16.4091%|
|hit_rate@10|39.8141%|39.8473%|

Final cohort:114477 requests/52410 users;8184 unavailable targets retained;66387 targets in the shared candidate pool. Source1112623 train-role clicks,364047 article metadata records; testA/B excluded. Initial fit cutoff1507980021008, natural final-end1510603454886; only revealed earlier labels may update the selected prequential policy. Source SQL independently verified all request fields/history, schedules, latest eligible cutoff, model identities and exact group-set equivalence for no-new-label reuses. Source DB unchanged.122 scheduled model entries comprise20 actual fits and102 exact no-new-training-group reuses. Total CPU-run wall time351.77s; this is not online serving latency.

The historical static-service33.3055% used a different query clock and is not the matched final comparator. Scores across time windows are not interchangeable. The selected v1 policy uses no supplied embeddings.

## Engineering evidence

- 114855/114855 validation replay requests matched offline Top10/history/model cutoff after945729 source events.
- Append persistence checked on16315 real requests across two restarts around a refresh switch;22158 committed batches passed an independent disk-chain check. Separate synthetic child tests exercised abrupt exit after commit.
- Snapshot real-prefix check: full constructor30.58s versus snapshot8.75s,6.29MB checkpoint; all mutable state and103 subsequent responses matched. Second snapshot constructor8.97s. This is one paired observation, not an SLA or repeated benchmark; complete log/retry index retained.
- 102 scoped v1 acceptance tests passed without skips; optional neural/ANN research tests and rejected dynamic SQLite backend are outside that claim.
- The synthetic demo ran successfully, including scheduled model activation, snapshot restore and exact retry; it supplies no quality benchmark.

## Accepted and rejected scope

Supported: the auditable CPU evaluation pipeline, immutable scheduled model replay, opt-in batch API, append journal and safe snapshot. Dynamic SQLite WAL/DELETE trials failed real-prefix integrity tests and remain rejected with root cause unconfirmed. Neural/sequence/cold-start experiments did not establish selected-baseline gains. No production live trainer, online lift, distributed serving, or final-test retuning is claimed.
