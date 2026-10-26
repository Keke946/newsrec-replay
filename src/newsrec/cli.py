"""P0 diagnostics and P1 baseline commands. No external dependencies."""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from .baseline import replay
from .data import audit, discover, environment, file_hash, load_database, source_hash, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'baseline'))
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new directory, never an input directory')
    parser.add_argument('--scope', choices=('linked-sample', 'local-unverified'), required=True)
    parser.add_argument('--skip-embeddings', action='store_true')
    parser.add_argument('--train-end', type=int, help='explicit training cutoff in milliseconds')
    parser.add_argument('--eval-end', type=int, help='explicit evaluation cutoff in milliseconds')
    parser.add_argument('--sample-smoke-cutoffs', action='store_true',
                        help='diagnostic quantiles; allowed only for linked-sample')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--rolling-trend', action='store_true',
                        help='add causal 6h/5min-refresh trend baseline; separate live-statistics setting')
    args = parser.parse_args(argv)
    files = discover(args.data_dir)
    source_dir, output = args.data_dir.resolve(), args.output.resolve()
    if output == source_dir or output in source_dir.parents or source_dir in output.parents:
        parser.error('output must be separate from, not inside or above, the input directory')
    if args.sample_smoke_cutoffs and args.scope != 'linked-sample':
        parser.error('automatic cutoffs are sample-smoke only; full local baseline needs explicit cutoffs')
    if args.command == 'baseline' and not args.sample_smoke_cutoffs and (args.train_end is None or args.eval_end is None):
        parser.error('baseline requires train-end and eval-end or sample-smoke-cutoffs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'command': args.command,
                'scope': args.scope, 'source_sha256': source_hash(), 'environment': environment(),
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'inputs': {}, 'status': 'running'}
    write_json(output / 'manifest.json', manifest)
    connection = None
    try:
        for role, path in files.items():
            manifest['inputs'][role] = {'name': path.name, 'bytes': path.stat().st_size,
                                        'sha256': file_hash(path)}
        connection = load_database(files, output / 'audit.sqlite')
        diagnostics = audit(connection, files, args.skip_embeddings)
        diagnostics['scope'] = args.scope
        write_json(output / 'data_audit.json', diagnostics)
        if args.command == 'baseline':
            cutoffs = diagnostics.get('proposed_time_cutoffs', {})
            train_end = cutoffs.get('train_end_ms') if args.sample_smoke_cutoffs else args.train_end
            eval_end = cutoffs.get('validation_end_ms') if args.sample_smoke_cutoffs else args.eval_end
            with (output / 'predictions.jsonl').open('w') as stream:
                def emit(row):
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                metrics, _ = replay(connection, train_end, eval_end, args.k, on_prediction=emit,
                                    include_trend=args.rolling_trend)
            metrics['scope'] = args.scope
            metrics['evaluation_status'] = 'smoke-only' if args.sample_smoke_cutoffs else 'exploratory-unfrozen'
            write_json(output / 'metrics.json', metrics)
        manifest['status'] = 'succeeded'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if connection:
            connection.close()
        manifest['wall_seconds'] = time.perf_counter() - started
        write_json(output / 'manifest.json', manifest)
    print(f'Completed {args.command}; scope={args.scope}; output={output}', flush=True)


if __name__ == '__main__':
    main()
