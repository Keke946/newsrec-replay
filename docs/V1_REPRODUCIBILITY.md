# Reproduce v1

Python3.12, NumPy2.3.5, LightGBM4.6.0, FastAPI0.115.12, httpx0.28.1 and Uvicorn0.34.3 were used for core acceptance. Linux/macOS are supported by the single-owner advisory file lock. Research Torch/Faiss dependencies are optional and excluded from the core acceptance claim.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-service.txt
python tools/check_v1.py --report acceptance.json
python tools/demo_v1.py --output demo-output
```

The demo generates synthetic data and two local toy models, records a scheduled switch, writes a state snapshot and verifies an exact retry after reopening. It does not download data or benchmark accuracy. Its output directory must be new.

For actual data, obtain authorized Tianchi input files separately. Audit with `python run.py audit --help` and place the resulting audit SQLite at `runs/recovered-audit/audit.sqlite`, or adapt the input path explicitly before freezing a new experiment. Never deserialize inherited pickle/joblib files. Core refit experiments use only train-role clicks and article metadata. Optional research embeddings are not needed for the selected policy.

Historical reproducibility commands:

```bash
python tools/run_refresh_backtest.py
python tools/run_refresh_backtest.py --confirm-validation
python tools/verify_refresh_backtest.py --confirm-validation
```

These produce immutable run directories; do not overwrite them. The archived final protocol is `protocols/v1_final_v1.json`; `--final-once` is the frozen evaluation entry, not a tuning loop. In this release that final result is already recorded. Any new development requires a separate protocol/data horizon, never reuse final results to adjust this release's parameters.

To replay the owner's supplied immutable registry/models, after obtaining the private artifacts and identical source data:

```bash
python tools/serve_refresh_replay.py --database runs/recovered-audit/audit.sqlite --registry runs/refresh-runtime-v0/registry.json --journal runs/local-refresh/batches.jsonl --snapshot runs/local-refresh/state.json.gz --port 8001
```

The endpoint accepts complete timestamp batches at `/replay/batch`. `/replay/checkpoint` writes the configured snapshot; graceful launcher shutdown also checkpoints. The journal remains the authority and is not compacted. An unusable snapshot falls back to full journal replay after journal identity/chain checks. Read-only original data and immutable models must remain available; retain original source/environment for older journals. This does not silently migrate another journal format or bypass identity guards.

The GitHub Actions workflow mirrors the local acceptance/demo steps; it has not run remotely until this source is actually published to a selected repository. Actions references were checked against their official repositories on2026-10-26: https://github.com/actions/checkout and https://github.com/actions/setup-python .
