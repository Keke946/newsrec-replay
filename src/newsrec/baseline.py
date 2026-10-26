"""Frozen, bounded-window ItemCF and timestamped next-click replay.

This is an independent NEW baseline, not a faithful reproduction of the old
all-pairs model. Train-only item pairs are fixed; user histories update in replay.
All events with the same timestamp are revealed as a batch after requests.
"""
import math
from collections import Counter, defaultdict, deque
from itertools import groupby

from .evaluation import RankingMetrics


class RollingPopularity:
    """Causal event-window popularity with a bounded refresh cadence.

    Fixed defaults were chosen before its first full-data score was viewed.
    This is a separate live-statistics baseline, not a relabelled frozen model.
    """
    def __init__(self, window_ms=6 * 60 * 60 * 1000, refresh_ms=5 * 60 * 1000):
        if window_ms <= 0 or refresh_ms <= 0:
            raise ValueError('Trend window and refresh must be positive')
        self.window_ms = window_ms
        self.refresh_ms = refresh_ms
        self.events = deque()
        self.counts = Counter()
        self.cached = []
        self.bucket = None
        self.last_observed_timestamp = None

    def observe(self, timestamp, item):
        if self.last_observed_timestamp is not None and timestamp < self.last_observed_timestamp:
            raise ValueError('Trend observations must be chronological')
        self.last_observed_timestamp = timestamp
        self.events.append((timestamp, item))
        self.counts[item] += 1

    def recommend(self, metadata, timestamp, k):
        if self.last_observed_timestamp is not None and self.last_observed_timestamp > timestamp:
            raise ValueError('Trend state contains a future event')
        while self.events and self.events[0][0] <= timestamp - self.window_ms:
            _, item = self.events.popleft()
            self.counts[item] -= 1
            if self.counts[item] == 0:
                del self.counts[item]
        bucket = timestamp // self.refresh_ms
        if bucket != self.bucket:
            self.cached = sorted(self.counts, key=lambda item: (-self.counts[item], item))
            self.bucket = bucket
        # Do not serve expired items from a cached ranking during its bucket.
        result = []
        for item in self.cached:
            if self.counts.get(item, 0) and item in metadata and metadata[item]['created'] <= timestamp:
                result.append(item)
                if len(result) >= k:
                    break
        return result


class WindowedItemCF:
    def __init__(self, window=3):
        if window < 1:
            raise ValueError('window must be positive')
        self.window = window
        self.neighbors = {}
        self.popularity = Counter()
        self.popular_items = []

    def fit(self, rows, top_neighbors=200):
        # rows must be sorted by user, timestamp, ordinal; enforced by SQL caller.
        pairs = defaultdict(Counter)
        for _, events in groupby(rows, lambda row: row[0]):
            recent = deque(maxlen=self.window)
            for _, article, _ in events:
                self.popularity[article] += 1
                for distance, previous in enumerate(reversed(recent), 1):
                    if previous != article:
                        weight = .9 ** (distance - 1)
                        pairs[previous][article] += weight
                        pairs[article][previous] += .7 * weight
                recent.append(article)
        for source, related in pairs.items():
            scores = [(item, count / math.sqrt(self.popularity[source] * self.popularity[item]))
                      for item, count in related.items()]
            self.neighbors[source] = sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:top_neighbors]
        # Frozen popularity must not be re-sorted for every replay request.
        self.popular_items = [item for item, _ in sorted(self.popularity.items(),
                                                        key=lambda pair: (-pair[1], pair[0]))]
        return self

    def recommend(self, history, metadata, timestamp, k=10, model='itemcf'):
        if model not in ('itemcf', 'popularity'):
            raise ValueError('Unknown model')
        scores = Counter()
        # Repeated earlier items remain eligible: this task allows repeat clicks.
        if model == 'itemcf':
            for position, article in enumerate(reversed(list(history)[-2:])):
                for item, similarity in self.neighbors.get(article, ()):
                    if item in metadata and metadata[item]['created'] <= timestamp:
                        scores[item] += .7 ** position * similarity
        result = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:k]]
        selected = set(result)
        # Stable, train-only popularity; future-unavailable items are filtered.
        for item in self.popular_items:
            if len(result) >= k:
                break
            if item not in selected and item in metadata and metadata[item]['created'] <= timestamp:
                result.append(item)
                selected.add(item)
        return result


def replay(connection, train_end, eval_end, k=10, window=3, on_prediction=None, include_trend=False):
    if train_end >= eval_end:
        raise ValueError('train_end must precede eval_end')
    if k < 1:
        raise ValueError('k must be positive')
    metadata = {article: {'category': category, 'created': created, 'words': words}
                for article, category, created, words in connection.execute('SELECT * FROM articles')}
    train_rows = connection.execute(
        'SELECT user,article,timestamp FROM clicks WHERE role="train" AND timestamp<=? '
        'ORDER BY user,timestamp,ordinal', (train_end,))
    model = WindowedItemCF(window).fit(train_rows)
    if not model.popularity:
        raise ValueError('No training events before train_end')
    # Each request occurs immediately after a known event. The target is next event.
    # Ambiguous same-timestamp transitions are counted but not scored.
    connection.execute('DROP TABLE IF EXISTS replay_targets')
    connection.execute('CREATE TEMP TABLE replay_targets AS SELECT ordinal,user,article,timestamp,'
                       'lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,'
                       'lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time '
                       'FROM clicks WHERE role="train"')
    rows = connection.execute('SELECT ordinal,user,article,timestamp,target,target_time '
                              'FROM replay_targets WHERE timestamp<=? ORDER BY timestamp,ordinal', (eval_end,))
    histories = defaultdict(lambda: deque(maxlen=20))
    history_counts = Counter()
    names = ['popularity', 'itemcf'] + (['rolling_trend'] if include_trend else [])
    overall = {name: RankingMetrics((5, k)) for name in names}
    trend = RollingPopularity() if include_trend else None
    groups = {name: defaultdict(lambda: RankingMetrics((5, k))) for name in overall}
    available = tied = 0
    predictions = []
    emit = on_prediction if on_prediction is not None else predictions.append
    # First-click map is diagnostic only; never used to score or build features.
    first_click = dict(connection.execute('SELECT article,min(timestamp) FROM clicks WHERE role="train" GROUP BY article'))
    for timestamp, batch in groupby(rows, lambda row: row[3]):
        events = list(batch)
        for ordinal, user, article, _, target, target_time in events:
            if timestamp <= train_end or target is None or target_time > eval_end:
                continue
            if target_time <= timestamp:
                tied += 1
                continue
            # Include only this user's current event, not other same-time events.
            history = (list(histories[user]) + [article])[-20:]
            is_available = target in metadata and metadata[target]['created'] <= timestamp
            available += is_available
            cold = first_click[target] > timestamp
            length = history_counts[user] + 1
            length_bin = '1' if length == 1 else '2-5' if length <= 5 else '6-10' if length <= 10 else '>10'
            for name in overall:
                candidates = (trend.recommend(metadata, timestamp, max(k, 5)) if name == 'rolling_trend'
                              else model.recommend(history, metadata, timestamp, max(k, 5), name))
                overall[name].add(target, candidates, user_id=user)
                groups[name][f'history={length_bin}'].add(target, candidates)
                groups[name]['zero-behavior' if cold else 'warm'].add(target, candidates)
                if is_available:
                    groups[name]['target-available'].add(target, candidates)
                emit({'request_id': ordinal, 'user_id': user, 'request_timestamp': timestamp,
                                    'target_id': target, 'target_timestamp': target_time, 'model': name,
                                    'target_available': is_available, 'zero_behavior_target': cold,
                                    'history_length': length, 'article_ids': candidates})
        for _, user, article, _, _, _ in events:
            histories[user].append(article)
            history_counts[user] += 1
            if trend:
                trend.observe(timestamp, article)
    result = {'protocol': 'next-click proxy, train-only roles, frozen graph/popularity, replayed user histories',
              'train_end_ms': train_end, 'eval_end_ms': eval_end, 'window': window,
              'training_clicks': sum(model.popularity.values()), 'tie_transitions_excluded': tied,
              'available_targets': available, 'scope_warning': 'proxy requests; no online experiment',
              'models': {name: {'overall': metric.result(),
                                 'groups': {group: values.result() for group, values in groups[name].items()}}
                         for name, metric in overall.items()}}
    if trend:
        result['rolling_trend_config'] = {'window_ms': trend.window_ms, 'refresh_ms': trend.refresh_ms,
                                         'event_update': 'reveal same-time batch after all its requests',
                                         'statistics': 'live, train-role observed events only'}
    return result, predictions
