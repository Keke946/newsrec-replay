"""Opt-in single-process replay of immutable, precomputed scheduled models.

No training, durable journal, or production deployment. Whole timestamp batches
query every click proxy before revealing any same-time global events.
"""
from collections import Counter,defaultdict,deque
import hashlib,json,threading
from pathlib import Path
import numpy as np
from .data import file_hash
from .serving import ContractError,integer
from .time_ranker import CausalCandidates,CONFIG,FEATURES


def metadata_hash(metadata):
    return hashlib.sha256(json.dumps(sorted(metadata.items()),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def encoder_hashes():
    root=Path(__file__).parent
    return {name:file_hash(root/name) for name in ['time_ranker.py','baseline.py']}


class ModelRegistry:
    def __init__(self,path,database_sha256):
        self.path=Path(path);self.root=self.path.parent.resolve();blob=self.path.read_bytes();self.info=json.loads(blob)
        info=self.info
        if info.get('schema')!='precomputed-refresh-replay-v1' or info.get('feature_names')!=FEATURES or info.get('config')!=CONFIG:
            raise ContractError('Registry schema/config differs')
        if info['encoder_sha256']!=encoder_hashes() or info['database_sha256']!=database_sha256:raise ContractError('Registry source/data identity differs')
        self.start=info['start_ms'];self.entries=info['models'];self.cache={};self.clock=None;self.current=None
        if not self.entries:raise ContractError('Empty registry')
        for n,e in enumerate(self.entries):
            integer(e['cutoff'],'cutoff');integer(e['activate_ms'],'activate_ms')
            cutoff=self.start if n==0 else self.start+n*21600000-1
            activate=self.start if n==0 else self.start+n*21600000+60000
            if e['cutoff']!=cutoff or e['activate_ms']!=activate or e['max_label_time']>cutoff:raise ContractError('Future labels or irregular activation')
            p=Path(e['file'])
            if p.is_absolute() or self.root not in (self.root/p).resolve().parents:raise ContractError('Model path escapes registry')
        # Initial model is mandatory; refresh models may arrive later.
        self._load(self.entries[0],required=True)
        self.version=hashlib.sha256(blob).hexdigest()

    def _load(self,e,required=False):
        cut=e['cutoff']
        if cut in self.cache:return self.cache[cut]
        path=self.root/e['file']
        if not path.is_file():
            if required:raise ContractError('Initial model missing')
            return None
        if file_hash(path)!=e['sha256']:raise ContractError('Model checksum differs')
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(path))
        if model.feature_name()!=FEATURES:raise ContractError('Model feature schema differs')
        self.cache[cut]=model;return model

    def select(self,timestamp):
        if timestamp<self.start or self.clock is not None and timestamp<self.clock:raise ContractError('Model clock moved backwards/before initial model')
        for e in reversed(self.entries):
            if e['activate_ms']>timestamp:continue
            model=self._load(e)
            if model is not None:
                self.clock=timestamp;self.current=e
                return model,e
        raise ContractError('No ready model')


    def select_pinned(self,timestamp,cutoff):
        """Restore a committed fallback choice despite later artifact arrival."""
        if timestamp<self.start or self.clock is not None and timestamp<self.clock:raise ContractError('Pinned model clock moved backwards')
        entry=next((e for e in self.entries if e['cutoff']==cutoff),None)
        if entry is None or entry['activate_ms']>timestamp:raise ContractError('Pinned model was not eligible')
        if self.current and cutoff<self.current['cutoff']:raise ContractError('Pinned model moved backwards')
        model=self._load(entry,required=True);self.clock=timestamp;self.current=entry
        return model,entry


class RefreshReplayEngine:
    def __init__(self,metadata,registry):
        if metadata_hash(metadata)!=registry.info['metadata_sha256']:raise ContractError('Metadata identity differs')
        self.registry=registry;self.metadata=metadata;self.state=CausalCandidates(metadata,CONFIG)
        self.histories=defaultdict(lambda:deque(maxlen=20));self.lengths=Counter();self.observed_through=None
        self.lock=threading.RLock();self.closed=False;self.switches=[]

    def process_batch(self,timestamp,events,score_event_ids=None,*,_pinned_cutoff=None):
        with self.lock:
            if self.closed:raise ContractError('Replay engine closed after processing failure')
            integer(timestamp,'timestamp')
            if self.observed_through is not None and timestamp<=self.observed_through:raise ContractError('Batch must advance time; retries require external replay')
            if not events or len(events)>10000:raise ContractError('Batch size must be in 1..10000')
            for row in events:
                if set(row)!={'event_id','user_id','article_id'}:raise ContractError('Unexpected event schema')
                for k,v in row.items():integer(v,k)
                if row['article_id'] not in self.metadata:raise ContractError('Unknown article')
            ordered=sorted(events,key=lambda r:r['event_id']);ids={r['event_id'] for r in ordered}
            if len(ids)!=len(ordered):raise ContractError('Duplicate event IDs')
            wanted=ids if score_event_ids is None else set(score_event_ids)
            if score_event_ids is not None:
                for i in score_event_ids:integer(i,'score_event_id')
                if len(wanted)!=len(score_event_ids):raise ContractError('Duplicate score IDs')
            if not wanted<=ids:raise ContractError('Score IDs not in batch')
            if wanted and timestamp<=self.registry.start:raise ContractError('Scored batch must be after initial fit cutoff')
            try:
                model=entry=None
                if timestamp>self.registry.start:
                    model,entry=self.registry.select(timestamp) if _pinned_cutoff is None else self.registry.select_pinned(timestamp,_pinned_cutoff)
                pending=[];matrices=[]
                for r in ordered:
                    if r['event_id'] in wanted:
                        u=r['user_id'];a=r['article_id'];h=(list(self.histories[u])+[(a,timestamp)])[-20:];length=self.lengths[u]+1
                        _,items,x=self.state.query(h,timestamp,length)
                        pending.append((r,length,items));matrices.append(np.asarray(x,dtype=np.float32).reshape(-1,15))
                    else:
                        # Same query side effects; avoid unused feature arithmetic.
                        self.state.last_query_timestamp=timestamp
                        self.state.hour.expire(timestamp);self.state.quarter.expire(timestamp)
                        self.state.trend.recommend(self.metadata,timestamp,CONFIG['candidate_trend_k'])
                scores=model.predict(np.concatenate(matrices),num_threads=1) if matrices and sum(len(x) for x in matrices) else []
                if not np.isfinite(scores).all():raise ContractError('Nonfinite model scores')
                results=[];offset=0
                for r,length,items in pending:
                    ranked=sorted(zip(items,scores[offset:offset+len(items)]),key=lambda v:(-float(v[1]),v[0]));offset+=len(items)
                    results.append(dict(event_id=r['event_id'],user_id=r['user_id'],history_length=length,candidate_count=len(items),article_ids=[i for i,_ in ranked[:10]]))
                for r in ordered:
                    self.state.observe(timestamp,r['article_id']);self.histories[r['user_id']].append((r['article_id'],timestamp));self.lengths[r['user_id']]+=1
                self.observed_through=timestamp
                if entry and (not self.switches or self.switches[-1]['cutoff']!=entry['cutoff']):self.switches.append(dict(timestamp=timestamp,cutoff=entry['cutoff']))
                return dict(timestamp=timestamp,accepted=len(events),model_cutoff=entry['cutoff'] if entry else None,registry_version=self.registry.version,predictions=results,mode='precomputed_replay')
            except Exception:
                self.closed=True
                raise


def prime_refresh(engine,connection,through):
    """Warm the all-click proxy clock without scoring or revealing labels."""
    from itertools import groupby
    if through!=engine.registry.start:raise ContractError('Warmup must stop at initial fit cutoff')
    count=0
    for t,rows in groupby(connection.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" AND timestamp<=? ORDER BY timestamp,ordinal',(through,)),lambda r:r[0]):
        events=[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in rows]
        engine.process_batch(t,events,[]);count+=len(events)
    return count
