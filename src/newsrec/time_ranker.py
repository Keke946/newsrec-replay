"""Causal live-trend candidates and optional LightGBM LambdaRank.

No provided embeddings or neural model outputs enter ranker training. All
features are observable at the proxy request; the next click is only a label.
"""
from collections import Counter, defaultdict, deque
from itertools import groupby
import math
from pathlib import Path
import time
import json

from .baseline import RollingPopularity
from .data import write_json
from .evaluation import RankingMetrics

FEATURES = ['log_count_6h', 'log_count_1h', 'log_count_15m', 'recent_share_1h',
            'recent_share_15m', 'log_age_seconds', 'log_words', 'category_share_20',
            'category_match_last', 'item_match_last', 'repeat_recency', 'log_history_length',
            'cached_trend_reciprocal_rank', 'category_share_3', 'log_previous_gap_seconds']
CONFIG = {'candidate_trend_k': 50, 'history_candidates': 3, 'train_ordinal_modulus': 13,
          'n_estimators': 60, 'num_leaves': 15, 'learning_rate': .05,
          'min_child_samples': 100, 'reg_lambda': 1., 'lambdarank_truncation_level': 10,
          'seed': 2026, 'threads': 4}


class EventWindow:
    def __init__(self, duration):
        self.duration = duration
        self.events = deque()
        self.counts = Counter()

    def observe(self, timestamp, item):
        self.events.append((timestamp, item))
        self.counts[item] += 1
        self.expire(timestamp)

    def expire(self, timestamp):
        while self.events and self.events[0][0] <= timestamp - self.duration:
            _, item = self.events.popleft()
            self.counts[item] -= 1
            if not self.counts[item]:
                del self.counts[item]


class CausalCandidates:
    def __init__(self, metadata, config=None):
        self.metadata = metadata
        self.config = config or CONFIG
        self.trend = RollingPopularity()
        self.hour = EventWindow(60*60*1000)
        self.quarter = EventWindow(15*60*1000)
        self.last_query_timestamp = None
        self.seen = set()
        self.fresh = None
        if self.config.get('fresh_slots',0):
            from .catalog import FreshCatalog
            self.fresh = FreshCatalog(metadata,self.config.get('fresh_max_age_ms',6*60*60*1000))

    def observe(self, timestamp, item):
        if self.last_query_timestamp is not None and timestamp<self.last_query_timestamp:
            raise ValueError('Late observation precedes an already served request')
        self.trend.observe(timestamp, item)
        self.hour.observe(timestamp, item)
        self.quarter.observe(timestamp, item)
        self.seen.add(item)

    def query(self, history, timestamp, history_length):
        if self.last_query_timestamp is not None and timestamp<self.last_query_timestamp:
            raise ValueError('Query time must not move backwards')
        if any(item not in self.metadata or event_time>timestamp for item,event_time in history):
            raise ValueError('Unknown or future history event')
        if self.trend.last_observed_timestamp is not None and self.trend.last_observed_timestamp>timestamp:
            raise ValueError('Trend state contains a future event')
        self.last_query_timestamp=timestamp
        self.hour.expire(timestamp)
        self.quarter.expire(timestamp)
        trend = self.trend.recommend(self.metadata, timestamp, self.config['candidate_trend_k'])
        rank = {item: 1/(index+1) for index, item in enumerate(trend)}
        candidates = list(trend)
        if self.fresh:
            candidates.extend(self.fresh.recommend(history,timestamp,self.seen,self.config['fresh_slots']))
        for item, _ in reversed(history[-self.config['history_candidates']:]):
            if item not in rank and item not in candidates and self.metadata[item]['created'] <= timestamp:
                candidates.append(item)
        categories = Counter(self.metadata[item]['category'] for item, _ in history)
        short = Counter(self.metadata[item]['category'] for item, _ in history[-3:])
        previous_gap = max(0, timestamp-history[-2][1])/1000 if len(history)>1 else 0
        features = []
        for item in candidates:
            data = self.metadata[item]
            count = self.trend.counts.get(item, 0)
            hour = self.hour.counts.get(item, 0)
            quarter = self.quarter.counts.get(item, 0)
            repeat = next((1/index for index,(earlier,_) in enumerate(reversed(history),1) if earlier==item), 0.)
            features.append([math.log1p(count), math.log1p(hour), math.log1p(quarter),
                             hour/max(count,1), quarter/max(count,1),
                             math.log1p(max(0,timestamp-data['created'])/1000), math.log1p(data['words']),
                             categories[data['category']]/max(len(history),1),
                             float(bool(history) and data['category']==self.metadata[history[-1][0]]['category']),
                             float(bool(history) and item==history[-1][0]), repeat, math.log1p(history_length),
                             rank.get(item,0.), short[data['category']]/max(min(3,len(history)),1),
                             math.log1p(previous_gap)])
        return trend, candidates, features


def run(connection, output, protocol, config=None):
    import numpy as np
    import lightgbm as lgb
    config = dict(config or CONFIG)
    columns = config.get('feature_columns', list(range(len(FEATURES))))
    if not columns or len(set(columns))!=len(columns) or any(index<0 or index>=len(FEATURES) for index in columns):
        raise ValueError('Invalid active feature columns')
    active_features = [FEATURES[index] for index in columns]
    train_end, eval_end = protocol['train_end_ms'], protocol['validation_end_ms']
    metadata = {article:dict(category=category,created=created,words=words)
                for article,category,created,words in connection.execute('SELECT * FROM articles')}
    state = CausalCandidates(metadata, config)
    histories = defaultdict(lambda:deque(maxlen=20))
    lengths = Counter()
    first = dict(connection.execute('SELECT article,min(timestamp) FROM clicks WHERE role="train" GROUP BY article'))
    connection.execute('CREATE TEMP TABLE rank_targets AS SELECT ordinal,user,article,timestamp,'
                       'lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,'
                       'lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time '
                       'FROM clicks WHERE role="train"')
    rows = connection.execute('SELECT * FROM rank_targets WHERE timestamp<=? ORDER BY timestamp,ordinal',(eval_end,))
    train_info = Counter()
    group_sizes = []
    feature_stream = (output/'train_features.f32').open('wb')
    label_stream = (output/'train_labels.i8').open('wb')
    metric = {name:RankingMetrics((5,10)) for name in ('rolling_trend','same_pool_6h','lambdarank_time')}
    cohort = {name:{key:RankingMetrics((5,10)) for key in ('zero-behavior','warm','history<=5','history>5','target-available')}
              for name in metric}
    pending, pending_features = [], []
    model = None
    fitted = False
    statistics = {'train_selection':train_info,'candidate_budget_max':config['candidate_trend_k']+config['history_candidates']+config.get('fresh_slots',0),
                  'training_cutoff':train_end,'validation_cutoff':eval_end,'feature_names':FEATURES,
                  'active_feature_names':active_features}
    predictions = (output/'predictions.jsonl').open('w')

    def fit():
        nonlocal model
        feature_stream.close();label_stream.close()
        if not group_sizes:
            raise ValueError('No informative training queries')
        x = np.memmap(output/'train_features.f32',dtype='<f4',mode='r',shape=(sum(group_sizes),len(FEATURES)))
        y = np.memmap(output/'train_labels.i8',dtype='i1',mode='r',shape=(sum(group_sizes),))
        clock = time.perf_counter()
        model = lgb.LGBMRanker(objective='lambdarank', n_estimators=config['n_estimators'],
                              num_leaves=config['num_leaves'],learning_rate=config['learning_rate'],
                              min_child_samples=config['min_child_samples'],reg_lambda=config['reg_lambda'],
                              lambdarank_truncation_level=config['lambdarank_truncation_level'],
                              random_state=config['seed'],n_jobs=config['threads'],
                              deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(x[:,columns],y,group=group_sizes,feature_name=active_features)
        model.booster_.save_model(str(output/'ranker.txt'))
        statistics['fit_seconds'] = time.perf_counter()-clock
        statistics['feature_gain'] = dict(zip(active_features,map(float,model.booster_.feature_importance(importance_type='gain'))))
        statistics['training_rows'] = sum(group_sizes)
        write_json(output/'training.json',statistics)
        print(f'Fitted {len(group_sizes)} groups, {sum(group_sizes)} candidate rows',flush=True)

    def flush():
        if not pending:
            return
        # Booster avoids one sklearn dataframe validation warning per batch.
        scores = model.booster_.predict(np.asarray(pending_features,dtype=np.float32)[:,columns],num_threads=config['threads']) if pending_features else []
        offset = 0
        for row,trend,candidates,features in pending:
            ordinal,user,article,timestamp,target,target_time = row
            current = scores[offset:offset+len(candidates)]
            offset += len(candidates)
            ranked = sorted(zip(candidates,current),key=lambda pair:(-pair[1],pair[0]))
            sixhour = sorted(zip(candidates,features),key=lambda pair:(-pair[1][0],pair[0]))
            outputs = {'rolling_trend':trend[:10], 'same_pool_6h':[item for item,_ in sixhour[:10]],
                       'lambdarank_time':[item for item,_ in ranked[:10]]}
            available = target in metadata and metadata[target]['created']<=timestamp
            cold = first[target]>timestamp
            length = len_for_request[ordinal]
            for name,picked in outputs.items():
                metric[name].add(target,picked,user_id=user)
                keys = ['zero-behavior' if cold else 'warm', 'history<=5' if length<=5 else 'history>5']
                if available:keys.append('target-available')
                for key in keys:cohort[name][key].add(target,picked,user_id=user)
                predictions.write(json.dumps(dict(request_id=ordinal,user_id=user,request_timestamp=timestamp,
                                                   target_id=target,target_timestamp=target_time,target_available=available,
                                                   zero_behavior_target=cold,history_length=length,model=name,
                                                   article_ids=picked))+'\n')
            statistics['validation_requests'] = statistics.get('validation_requests',0)+1
            statistics['validation_candidate_hits'] = statistics.get('validation_candidate_hits',0)+int(target in candidates)
            statistics['total_candidates'] = statistics.get('total_candidates',0)+len(candidates)
            if cold:
                statistics['zero_behavior_requests'] = statistics.get('zero_behavior_requests',0)+1
                statistics['zero_behavior_candidate_hits'] = statistics.get('zero_behavior_candidate_hits',0)+int(target in candidates)
            del len_for_request[ordinal]
        pending.clear();pending_features.clear()

    len_for_request = {}
    try:
        for timestamp,batch in groupby(rows,lambda row:row[3]):
            events = list(batch)
            for row in events:
                ordinal,user,article,_,target,target_time = row
                if target is None or target_time<=timestamp or target_time>eval_end:
                    continue
                valid = timestamp>train_end
                train = not valid and target_time<=train_end and ordinal%config['train_ordinal_modulus']==0
                if not (train or valid):continue
                if valid and not fitted:
                    fit();fitted=True
                    # Match original baseline's first validation ranking snapshot.
                    state.trend.bucket = None
                history = (list(histories[user])+[(article,timestamp)])[-20:]
                length = lengths[user]+1
                trend,candidates,features = state.query(history,timestamp,length)
                if train:
                    train_info['sampled_queries']+=1
                    if target not in metadata or metadata[target]['created']>timestamp:
                        train_info['target_unavailable']+=1
                    elif target not in candidates:
                        train_info['target_not_retrieved']+=1
                    elif len(candidates)<2:
                        train_info['no_negative_candidate']+=1
                    else:
                        np.asarray(features,dtype='<f4').tofile(feature_stream)
                        np.asarray([int(item==target) for item in candidates],dtype='i1').tofile(label_stream)
                        group_sizes.append(len(candidates))
                        train_info['informative_queries']+=1
                else:
                    pending.append((row,trend,candidates,features));pending_features.extend(features)
                    len_for_request[ordinal]=length
                    if len(pending)>=512:flush()
            for _,user,article,_,_,_ in events:
                histories[user].append((article,timestamp));lengths[user]+=1
                state.observe(timestamp,article)
        if pending:flush()
        if not fitted:raise ValueError('No validation requests')
    finally:
        feature_stream.close();label_stream.close();predictions.close()
    statistics['candidate_coverage'] = statistics['validation_candidate_hits']/statistics['validation_requests']
    write_json(output/'training.json',statistics)
    return {'models':{name:{'overall':values.result(),'groups':{key:value.result() for key,value in cohort[name].items()}}
                      for name,values in metric.items()},'diagnostics':statistics,
            'warning':'exploratory validation, no actual impressions; updated features and learned ranking, not an isolated architecture comparison'}
