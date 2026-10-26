"""Ranking metrics with the complete request denominator and strict contracts."""
import math


class RankingMetrics:
    def __init__(self, ks=(5, 10)):
        self.ks = tuple(sorted(set(ks)))
        if not self.ks or any(k < 1 for k in self.ks):
            raise ValueError('K must be positive')
        self.count = 0
        self.sums = {k: [0., 0., 0.] for k in self.ks}
        self.users = {}

    def add(self, target, predictions, user_id=None):
        if len(predictions) != len(set(predictions)):
            raise ValueError('Predictions must be unique')
        self.count += 1
        rank = predictions.index(target) + 1 if target in predictions else math.inf
        user = None
        if user_id is not None:
            user = self.users.setdefault(user_id, [0.] * (1 + 3 * len(self.ks)))
            user[0] += 1
        for position, k in enumerate(self.ks):
            if rank <= k:
                self.sums[k][0] += 1
                self.sums[k][1] += 1 / rank
                self.sums[k][2] += 1 / math.log2(rank + 1)
                if user is not None:
                    offset = 1 + 3 * position
                    user[offset] += 1
                    user[offset + 1] += 1 / rank
                    user[offset + 2] += 1 / math.log2(rank + 1)

    def result(self):
        output = {'requests': self.count}
        for k in self.ks:
            for name, value in zip(('hit_rate', 'mrr', 'ndcg'), self.sums[k]):
                output[f'{name}@{k}'] = value / self.count if self.count else None
        if self.users:
            # Both units are reported. Active users no longer silently dominate
            # the only reported score, while request-averaged evidence is kept.
            user_output = {'users': len(self.users),
                           'requests_with_user_id': int(sum(row[0] for row in self.users.values()))}
            for position, k in enumerate(self.ks):
                for index, name in enumerate(('hit_rate', 'mrr', 'ndcg')):
                    column = 1 + 3 * position + index
                    user_output[f'{name}@{k}'] = sum(row[column] / row[0]
                                                     for row in self.users.values()) / len(self.users)
            output['user_balanced'] = user_output
        return output


def minmax(scores):
    if any(not math.isfinite(v) for v in scores.values()):
        raise ValueError('Scores must be finite')
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    return {item: 1. if high == low else (value - low) / (high - low)
            for item, value in scores.items()}
