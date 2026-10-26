"""Single-process replay service engine sharing offline candidate/features.

Explicit timestamp batches prevent same-time global information leakage.
This in-memory prototype is not durable or safe for multi-worker deployment.
"""
from collections import Counter, OrderedDict, defaultdict, deque
import hashlib
import json
import math
import threading

from .time_ranker import CausalCandidates, CONFIG, FEATURES


class ContractError(ValueError):
    pass


def integer(value,name):
    if type(value) is not int or value<0:
        raise ContractError(f'{name} must be a nonnegative integer')


class ReplayEngine:
    def __init__(self,metadata,model,config=None,version='local-prototype'):
        if model.feature_name()!=FEATURES:
            raise ContractError('Model feature schema differs from shared feature encoder')
        self.config=dict(config or CONFIG)
        self.state=CausalCandidates(metadata,self.config)
        self.metadata=metadata
        self.model=model
        self.version=version
        self.histories=defaultdict(lambda:deque(maxlen=20))
        self.lengths=Counter()
        self.observed_through=None
        self.query_clock=None
        self.batches=OrderedDict()
        self.lock=threading.RLock()

    def ingest(self,timestamp,events):
        with self.lock:
            integer(timestamp,'timestamp')
            if not events or len(events)>10000:
                raise ContractError('Batch size must be in 1..10000')
            for row in events:
                if set(row)!={'event_id','user_id','article_id'}:
                    raise ContractError('Unexpected event schema')
                for key,value in row.items():integer(value,key)
                if row['article_id'] not in self.metadata:
                    raise ContractError('Unknown article')
            ordered=sorted(events,key=lambda row:row['event_id'])
            if len({row['event_id'] for row in ordered})!=len(ordered):
                raise ContractError('Duplicate event_id within batch')
            digest=hashlib.sha256(json.dumps(ordered,sort_keys=True).encode()).hexdigest()
            if timestamp in self.batches:
                if self.batches[timestamp]!=digest:
                    raise ContractError('Closed timestamp batch has different contents')
                return {'accepted':0,'duplicate_batch':True,'observed_through':self.observed_through}
            if self.observed_through is not None and timestamp<=self.observed_through:
                raise ContractError('Timestamp batch already closed or outside replay cache')
            if self.query_clock is not None and timestamp<self.query_clock:
                raise ContractError('Late batch precedes already served request')
            # Validate the whole batch before changing any observed state.
            for row in ordered:
                self.state.observe(timestamp,row['article_id'])
                self.histories[row['user_id']].append((row['article_id'],timestamp))
                self.lengths[row['user_id']]+=1
            self.observed_through=timestamp
            self.batches[timestamp]=digest
            if len(self.batches)>1024:self.batches.popitem(last=False)
            return {'accepted':len(ordered),'duplicate_batch':False,'observed_through':timestamp}

    def recommend(self,user_id,timestamp,current_article=None,k=10):
        import numpy as np
        with self.lock:
            integer(user_id,'user_id');integer(timestamp,'timestamp');integer(k,'k')
            if not 1<=k<=10:raise ContractError('k must be in 1..10')
            if current_article is not None:
                integer(current_article,'current_article')
                if current_article not in self.metadata:raise ContractError('Unknown current article')
            if self.observed_through is not None and timestamp<=self.observed_through:
                raise ContractError('Request must precede reveal of its timestamp batch')
            if self.query_clock is not None and timestamp<self.query_clock:
                raise ContractError('Request time moved backwards')
            history=list(self.histories.get(user_id,()))
            length=self.lengths[user_id]
            if current_article is not None:
                history=(history+[(current_article,timestamp)])[-20:];length+=1
            trend,items,features=self.state.query(history,timestamp,length)
            self.query_clock=timestamp
            if not items:
                picked=[];scores=[];mode='empty_candidates'
            elif not history:
                picked=trend[:k];scores=[None]*len(picked);mode='cold_user_trend'
            else:
                values=self.model.predict(np.asarray(features,dtype=np.float32),num_threads=1)
                if len(values)!=len(items):raise ContractError('Model score count differs from candidates')
                if not all(math.isfinite(float(value)) for value in values):
                    raise ContractError('Model emitted nonfinite scores')
                ranked=sorted(zip(items,map(float,values)),key=lambda pair:(-pair[1],pair[0]))[:k]
                picked=[item for item,_ in ranked];scores=[score for _,score in ranked];mode='ranker'
            return dict(user_id=user_id,timestamp=timestamp,article_ids=picked,scores=scores,
                        model_version=self.version,mode=mode,candidate_count=len(items),history_length=length)


def prime(engine,connection,through):
    from itertools import groupby
    rows=connection.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" AND timestamp<=? '
                            'ORDER BY timestamp,ordinal',(through,))
    count=0
    for timestamp,batch in groupby(rows,lambda row:row[0]):
        events=[dict(event_id=ordinal,user_id=user,article_id=article) for _,ordinal,user,article in batch]
        engine.ingest(timestamp,events);count+=len(events)
    return count
