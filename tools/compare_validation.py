"""Paired validation-only HR@5 uncertainty; needs NumPy, no model training.

python tools/compare_validation.py runs/.../predictions.jsonl --output report.json
User clusters preserve within-user dependence; time blocks are a separate
robustness diagnostic, not a proof that all temporal dependence is removed.
"""
import argparse
from collections import defaultdict
from itertools import groupby
import json
from pathlib import Path

import numpy as np


def cluster_bootstrap(rows, replicas=400, seed=2026):
    """Rows: [request count, reference hit count, challenger hit count]."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values) or (values[:, 0] <= 0).any():
        raise ValueError('Need nonempty positive-size clusters with three columns')
    if replicas < 2:
        raise ValueError('Need at least two bootstrap replicas')
    counts = values[:, 0]
    differences = values[:, 2] - values[:, 1]
    macro = differences / counts
    result = {'clusters': len(values), 'requests': int(counts.sum()),
              'request_weighted_delta': float(differences.sum() / counts.sum()),
              'cluster_macro_delta': float(macro.mean()),
              'replicas': replicas, 'seed': seed}
    if len(values) < 2:
        result['interval_status'] = 'insufficient independent cluster units'
        return result
    rng = np.random.default_rng(seed)
    weighted, balanced = [], []
    for start in range(0, replicas, 16):
        indices = rng.integers(0, len(values), size=(min(16, replicas - start), len(values)))
        weighted.extend((differences[indices].sum(axis=1) / counts[indices].sum(axis=1)).tolist())
        balanced.extend(macro[indices].mean(axis=1).tolist())
    result['request_weighted_95pct_percentile_interval'] = np.quantile(weighted, [.025, .975]).tolist()
    result['cluster_macro_95pct_percentile_interval'] = np.quantile(balanced, [.025, .975]).tolist()
    return result


def compare(path, reference='itemcf', challenger='rolling_trend', replicas=400):
    users = defaultdict(lambda: [0, 0, 0])
    blocks = defaultdict(lambda: [0, 0, 0])
    request_ids = set()
    with Path(path).open() as stream:
        records = (json.loads(line) for line in stream)
        for request_id, group in groupby(records, key=lambda row: row['request_id']):
            if request_id in request_ids:
                raise ValueError('Repeated or noncontiguous request ID')
            request_ids.add(request_id)
            rows = list(group)
            indexed = {row['model']: row for row in rows}
            if len(indexed) != len(rows):
                raise ValueError('Duplicate model prediction for one request')
            first, second = indexed[reference], indexed[challenger]
            for key in ('user_id', 'request_timestamp', 'target_id', 'target_timestamp'):
                if first[key] != second[key]:
                    raise ValueError(f'Misaligned request {request_id}: {key}')
            target = first['target_id']
            ref_hit = int(target in first['article_ids'][:5])
            new_hit = int(target in second['article_ids'][:5])
            for row in (users[first['user_id']], blocks[first['request_timestamp'] // 21600000]):
                row[0] += 1
                row[1] += ref_hit
                row[2] += new_hit
    return {'scope': 'validation-only paired HR@5 analysis', 'reference': reference,
            'challenger': challenger, 'unique_requests': len(request_ids),
            'user_cluster_bootstrap': cluster_bootstrap(list(users.values()), replicas),
            'six_hour_block_bootstrap': cluster_bootstrap(list(blocks.values()), replicas),
            'limitations': ['different frozen vs live global statistics explicitly declared',
                           'bootstrap is descriptive validation uncertainty, not a causal A/B test',
                           'few time blocks and serial dependence limit inference',
                           'no final-region scores were used']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    args.output.write_text(json.dumps(compare(args.predictions), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
