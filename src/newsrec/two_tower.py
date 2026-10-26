"""Optional CPU PyTorch two-tower prototype; no legacy model deserialization.

Fixed first-run configuration, provided 250-D vectors, train-only causal
prefixes, masked in-batch negatives, exact catalog retrieval. A prototype,
not a replacement for freshness baselines, ranking, ANN or final evaluation.
"""
import argparse
from collections import deque
from itertools import groupby
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import file_hash, source_hash, write_json
from .evaluation import RankingMetrics


class VectorStore:
    def __init__(self, directory, connection):
        directory = Path(directory)
        info = json.loads((directory / 'VECTOR_MANIFEST.json').read_text())
        n, dim = info['rows'], info['dimensions']
        if (info['vectors_dtype'], info['article_ids_dtype'], dim) != ('<f4', '<i8', 250):
            raise ValueError('Unexpected vector contract')
        if (directory / 'article_vectors.f32').stat().st_size != n * dim * 4:
            raise ValueError('Truncated or wrong-sized vector file')
        if (directory / 'article_ids.i64').stat().st_size != n * 8:
            raise ValueError('Truncated or wrong-sized ID file')
        self.ids = np.memmap(directory / 'article_ids.i64', dtype='<i8', mode='r', shape=(n,))
        self.raw = np.memmap(directory / 'article_vectors.f32', dtype='<f4', mode='r', shape=(n, dim))
        if len(np.unique(self.ids)) != n or (self.ids < 0).any():
            raise ValueError('Duplicate or negative vector IDs')
        if not np.isfinite(self.raw).all():
            raise ValueError('Non-finite input vectors')
        self.norms = np.linalg.norm(self.raw, axis=1)
        if (self.norms <= 0).any():
            raise ValueError('Zero input vector')
        self.lookup = {int(item): i for i, item in enumerate(self.ids)}
        metadata = dict(connection.execute('SELECT article,created FROM articles'))
        if set(metadata) != set(self.lookup):
            raise ValueError('Vector IDs and metadata catalog differ')
        self.created = np.asarray([metadata[int(item)] for item in self.ids], dtype=np.int64)

    def vectors(self, rows):
        return np.asarray(self.raw[rows], dtype=np.float32) / self.norms[rows, None]


class TwoTower(nn.Module):
    def __init__(self, input_dim=250, hidden=128, dimension=64):
        super().__init__()
        def tower():
            return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, dimension))
        self.user_tower = tower()
        self.item_tower = tower()

    def users(self, vectors):
        return F.normalize(self.user_tower(vectors), dim=-1)

    def items(self, vectors):
        return F.normalize(self.item_tower(vectors), dim=-1)


def negative_mask(target_indices, candidate_created, request_times):
    """True means usable logit; duplicate targets/future candidates off diagonal masked."""
    allowed = candidate_created[None, :] <= request_times[:, None]
    distinct = target_indices[:, None] != target_indices[None, :]
    diagonal = torch.eye(len(target_indices), dtype=torch.bool, device=target_indices.device)
    if not torch.all(allowed.diagonal()):
        raise ValueError('Training positive was unavailable at its own request time')
    return allowed & (distinct | diagonal)


def build_examples(connection, store, output, end, request_after=None, training=False):
    # Reserve at most one row per observed event; actual n is written separately.
    capacity = connection.execute('SELECT count(*) FROM clicks WHERE role="train" AND timestamp<=?', (end,)).fetchone()[0]
    if capacity == 0:
        raise ValueError('No source events')
    query_vectors = np.lib.format.open_memmap(output / 'queries.npy', mode='w+', dtype='float32', shape=(capacity, 250))
    # user, request ordinal, request time, target vector index, target time, history length
    records = np.lib.format.open_memmap(output / 'records.npy', mode='w+', dtype='int64', shape=(capacity, 6))
    n = unavailable_excluded = tied = 0
    cursor = connection.execute('SELECT user,article,timestamp,ordinal FROM clicks WHERE role="train" '
                                'AND timestamp<=? ORDER BY user,timestamp,ordinal', (end,))
    for user, events in groupby(cursor, key=lambda row: row[0]):
        history = deque(maxlen=3)
        pending = []
        previous = None
        history_length = 0
        previous_query = None
        previous_length = 0
        for _, article, timestamp, ordinal in events:
            index = store.lookup[article]
            if previous is not None:
                prev_time, prev_ordinal = previous
                if timestamp == prev_time:
                    tied += 1
                elif request_after is None or prev_time > request_after:
                    legal_positive = store.created[index] <= prev_time
                    if training and not legal_positive:
                        unavailable_excluded += 1
                    else:
                        query_vectors[n] = previous_query
                        records[n] = (user, prev_ordinal, prev_time, index, timestamp, previous_length)
                        n += 1
            # Match replay: expose earlier timestamp batches, then this event;
            # other clicks tied with this request remain hidden.
            if previous is not None and timestamp != previous[0]:
                history.extend(pending)
                history_length += len(pending)
                pending = []
            query_history = (list(history) + [index])[-3:]
            weights = np.asarray([.7 ** distance for distance in range(len(query_history)-1, -1, -1)], dtype=np.float32)
            pooled = (store.vectors(np.asarray(query_history)) * weights[:, None]).sum(axis=0)
            pooled /= max(float(np.linalg.norm(pooled)), 1e-12)
            previous_query = pooled
            previous_length = history_length + 1
            pending.append(index)
            previous = (timestamp, ordinal)
    query_vectors.flush()
    records.flush()
    info = {'examples': n, 'capacity': capacity, 'unavailable_training_positives_excluded': unavailable_excluded,
            'same_timestamp_transitions_in_source': tied, 'training': training,
            'request_after': request_after, 'end': end, 'history_encoder': 'last3 normalized provided vectors, .7 decay'}
    write_json(output / 'examples.json', info)
    return query_vectors[:n], records[:n], info


def train(model, store, queries, records, epochs=1, batch_size=256, seed=2026):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    log = []
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(records))
        loss_sum = count = updates = 0
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            if len(chosen) < 2:
                continue
            rows = np.asarray(records[chosen])
            item_rows = rows[:, 3]
            query = torch.from_numpy(np.array(queries[chosen], copy=True))
            items = torch.from_numpy(store.vectors(item_rows))
            indices = torch.from_numpy(item_rows.copy())
            mask = negative_mask(indices, torch.from_numpy(store.created[item_rows].copy()),
                                 torch.from_numpy(rows[:, 2].copy()))
            if int(mask.sum()) <= len(chosen):
                continue
            logits = model.users(query) @ model.items(items).T / .1
            logits = logits.masked_fill(~mask, -torch.inf)
            loss = F.cross_entropy(logits, torch.arange(len(chosen)))
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(chosen)
            count += len(chosen)
            updates += 1
        entry = {'epoch': epoch + 1, 'updates': updates, 'examples_in_updates': count,
                 'mean_inbatch_loss': loss_sum / count if count else None}
        log.append(entry)
        print(json.dumps(entry), flush=True)
    return log


@torch.inference_mode()
def evaluate(model, store, queries, records, output, batch_size=64, k=10, dimension=64,
             model_name='two_tower_content_v0'):
    model.eval()
    encoded = torch.empty((len(store.ids), dimension), dtype=torch.float32)
    for start in range(0, len(store.ids), 4096):
        end = min(start + 4096, len(store.ids))
        encoded[start:end] = model.items(torch.from_numpy(store.vectors(np.arange(start, end))))
    created = torch.from_numpy(store.created)
    metric = RankingMetrics((5, k))
    available_metric = RankingMetrics((5, k))
    available = 0
    with (output / 'predictions.jsonl').open('w') as stream:
        for start in range(0, len(records), batch_size):
            rows = np.asarray(records[start:start + batch_size])
            user = model.users(torch.from_numpy(np.array(queries[start:start + batch_size], copy=True)))
            scores = user @ encoded.T
            legal = created[None, :] <= torch.from_numpy(rows[:, 2].copy())[:, None]
            scores.masked_fill_(~legal, -torch.inf)
            values, indices = scores.topk(min(k, len(store.ids)), dim=1)
            for row, scores_row, picked in zip(rows, values.tolist(), indices.tolist()):
                user_id, ordinal, timestamp, target_index, target_time, length = map(int, row)
                predictions = [int(store.ids[index]) for score, index in zip(scores_row, picked) if math.isfinite(score)]
                target = int(store.ids[target_index])
                is_available = bool(store.created[target_index] <= timestamp)
                available += is_available
                metric.add(target, predictions, user_id=user_id)
                if is_available:
                    available_metric.add(target, predictions)
                stream.write(json.dumps({'request_id': ordinal, 'user_id': user_id, 'request_timestamp': timestamp,
                                         'target_id': target, 'target_timestamp': target_time,
                                         'target_available': is_available, 'model': model_name,
                                         'history_length': length, 'article_ids': predictions}) + '\n')
            if start % 8192 == 0:
                print(f'Exact catalog retrieval: {min(start + batch_size, len(records))}/{len(records)}', flush=True)
    return {'overall': metric.result(), 'target_available': available_metric.result(),
            'available_targets': available, 'retrieval': 'exact cosine over all metadata-available catalog items',
            'catalog_rows': len(store.ids), 'dimension': dimension}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    torch.manual_seed(2026)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    connection = sqlite3.connect(f'file:{args.database.resolve()}?mode=ro', uri=True)
    manifest = {'status': 'running', 'source_sha256': source_hash(), 'torch': torch.__version__,
                'protocol': protocol, 'seed': 2026, 'threads': args.threads,
                'config': {'epochs': 1, 'batch_size': 256, 'hidden': 128, 'dimension': 64,
                           'lr': .001, 'temperature': .1, 'history': 3, 'history_decay': .7},
                'limitations': ['provided embeddings have unknown training and availability provenance',
                               'first content-assisted prototype; no ANN, freshness feature, or learned ranker',
                               'validation only; no final-region scores']}
    write_json(args.output / 'manifest.json', manifest)
    clock = time.perf_counter()
    try:
        # Verify input CSV identity; model cannot silently switch datasets.
        for name, key in [('train_click_log.csv', 'train_csv_sha256'), ('articles.csv', 'articles_csv_sha256')]:
            if file_hash(args.data_dir / name) != protocol[key]:
                raise ValueError(f'{name} differs from protocol')
        store = VectorStore(args.data_dir, connection)
        manifest['input_hashes'] = {name: file_hash(args.data_dir / name) for name in
                                    ('article_vectors.f32', 'article_ids.i64', 'VECTOR_MANIFEST.json')}
        manifest['database_sha256'] = file_hash(args.database)
        write_json(args.output / 'manifest.json', manifest)
        train_dir, valid_dir = args.output / 'train', args.output / 'validation'
        train_dir.mkdir(); valid_dir.mkdir()
        queries, records, info = build_examples(connection, store, train_dir, protocol['train_end_ms'], training=True)
        manifest['training_examples'] = info
        model = TwoTower()
        train_clock = time.perf_counter()
        manifest['training_log'] = train(model, store, queries, records)
        manifest['training_seconds'] = time.perf_counter() - train_clock
        torch.save(model.state_dict(), args.output / 'model_state.pt')
        del queries, records
        queries, records, info = build_examples(connection, store, valid_dir, protocol['validation_end_ms'],
                                                 request_after=protocol['train_end_ms'])
        if len(records) != 114855:
            raise ValueError(f'Validation cohort differs from verified P0: {len(records)}')
        manifest['validation_examples'] = info
        inference_clock = time.perf_counter()
        metrics = evaluate(model, store, queries, records, args.output)
        manifest['inference_seconds'] = time.perf_counter() - inference_clock
        write_json(args.output / 'metrics.json', metrics)
        manifest['status'] = 'succeeded'
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        connection.close()
        manifest['wall_seconds'] = time.perf_counter() - clock
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
