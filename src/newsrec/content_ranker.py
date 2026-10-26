"""Fixed-pool content features and paired train-internal ranking experiment.

No Torch, text encoder, learned retrieval output, or final-region evaluation.
"""
from collections import Counter,defaultdict,deque
from itertools import groupby
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .data import file_hash,write_json
from .evaluation import RankingMetrics
from .time_ranker import CausalCandidates,CONFIG,FEATURES

CONTENT_FEATURES=['cosine_last','cosine_max_last3','cosine_decay_mean_last3']


def predict_matrix(model,matrix,width,threads):
    return model.predict(matrix[:,:width],num_threads=threads) if len(matrix) else np.empty(0,dtype=np.float64)


class ContentVectors:
    def __init__(self,ids,values,created):
        ids=np.asarray(ids)
        if len(ids)!=len(set(map(int,ids))) or len(ids)!=len(values):raise ValueError('Vector ID contract')
        if set(map(int,ids))!=set(created):raise ValueError('Vector catalog differs')
        if not np.isfinite(values).all():raise ValueError('Nonfinite vectors')
        norms=np.linalg.norm(values,axis=1)
        if (norms<=0).any():raise ValueError('Zero vector')
        self.normalized=np.asarray(values/norms[:,None],dtype=np.float32)
        self.lookup={int(item):i for i,item in enumerate(ids)}
        self.created=created

    @classmethod
    def load(cls,directory,metadata,protocol):
        directory=Path(directory);info=json.loads((directory/'VECTOR_MANIFEST.json').read_text())
        if (info['rows'],info['dimensions'],info['vectors_dtype'],info['article_ids_dtype'])!=(364047,250,'<f4','<i8'):
            raise ValueError('Unexpected full vector manifest')
        for name,size,expected in [('article_vectors.f32',364047000,protocol['vector_sha256']),
                                   ('article_ids.i64',364047*8,protocol['ids_sha256'])]:
            path=directory/name
            if path.stat().st_size!=size or file_hash(path)!=expected:raise ValueError('Vector length/hash differs')
        ids=np.memmap(directory/'article_ids.i64',dtype='<i8',mode='r',shape=(364047,))
        values=np.memmap(directory/'article_vectors.f32',dtype='<f4',mode='r',shape=(364047,250))
        return cls(ids,values,{i:d['created'] for i,d in metadata.items()})

    def features(self,items,history,timestamp):
        if any(t>timestamp for _,t in history):raise ValueError('Future history')
        if any(i not in self.lookup or self.created[i]>timestamp for i in items):raise ValueError('Future/unknown candidate')
        if any(i not in self.lookup for i,_ in history):raise ValueError('Unknown history')
        if not items or not history:return np.zeros((len(items),3),dtype=np.float32)
        recent=history[-3:]
        candidates=self.normalized[[self.lookup[i] for i in items]]
        earlier=self.normalized[[self.lookup[i] for i,_ in recent]]
        similarity=np.clip(candidates@earlier.T,-1,1)
        weights=np.asarray([.7**d for d in range(len(recent)-1,-1,-1)],dtype=np.float32)
        weights/=weights.sum()
        return np.column_stack((similarity[:,-1],similarity.max(axis=1),similarity@weights)).astype(np.float32)


def validate_protocol(protocol,allow_confirmation=False):
    train_end=protocol['train_end_ms'];end=protocol['validation_end_ms']
    if train_end>=end:raise ValueError('Invalid protected boundary')
    if protocol.get('mode')=='validation_confirmation':
        if not allow_confirmation:raise ValueError('Explicit validation confirmation authorization required')
        if train_end!=protocol['old_validation_start_ms'] or end!=protocol['old_validation_end_ms']:
            raise ValueError('Confirmation must use exactly the frozen validation boundary')
    elif end>protocol['old_validation_start_ms']:
        raise ValueError('Development crosses protected boundary')


def run_pair(connection,vectors,metadata,output,protocol,config=None,allow_confirmation=False,candidate_factory=None):
    import lightgbm as lgb
    config=dict(config or CONFIG)
    train_end=protocol['train_end_ms'];end=protocol['validation_end_ms']
    validate_protocol(protocol,allow_confirmation)
    state=(candidate_factory or CausalCandidates)(metadata,config)
    histories=defaultdict(lambda:deque(maxlen=20));lengths=Counter()
    # Restrict source rows BEFORE constructing labels and cold diagnostics.
    connection.execute('CREATE TEMP TABLE content_targets AS SELECT ordinal,user,article,timestamp,'
                       'lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,'
                       'lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time '
                       'FROM clicks WHERE role="train" AND timestamp<=?',(end,))
    first=dict(connection.execute('SELECT article,min(timestamp) FROM clicks WHERE role="train" AND timestamp<=? GROUP BY article',(end,)))
    stream=(output/'train_features.f32').open('wb');labels=(output/'train_labels.i8').open('wb')
    prediction=(output/'predictions.jsonl').open('w')
    groups=[];models={};pending=[];selection=Counter();stats=Counter();fit_times={}
    metrics={name:defaultdict(lambda:RankingMetrics((5,10))) for name in ('base15','content18')}
    paired_users=defaultdict(lambda:np.zeros(3,dtype=np.float64))
    blocks=defaultdict(lambda:np.zeros(3,dtype=np.float64))
    candidate_digest=hashlib.sha256();base_digest=hashlib.sha256()
    features_seconds=0.

    def fit():
        stream.close();labels.close()
        if not groups:raise ValueError('No informative train queries')
        np.save(output/'group_sizes.npy',np.asarray(groups,dtype=np.int32),allow_pickle=False)
        x=np.memmap(output/'train_features.f32',dtype='<f4',mode='r',shape=(sum(groups),18))
        y=np.memmap(output/'train_labels.i8',dtype='i1',mode='r',shape=(sum(groups),))
        if hashlib.sha256(x[:,:15].astype('<f4').tobytes()).hexdigest()!=base_digest.hexdigest():
            raise ValueError('Baseline columns changed in persisted shared matrix')
        for name,width in [('base15',15),('content18',18)]:
            start=time.perf_counter()
            ranker=lgb.LGBMRanker(objective='lambdarank',n_estimators=config['n_estimators'],num_leaves=config['num_leaves'],
                                  learning_rate=config['learning_rate'],min_child_samples=config['min_child_samples'],
                                  reg_lambda=config['reg_lambda'],lambdarank_truncation_level=config['lambdarank_truncation_level'],
                                  random_state=config['seed'],n_jobs=config['threads'],deterministic=True,force_col_wise=True,verbosity=-1)
            ranker.fit(x[:,:width],y,group=groups,feature_name=(FEATURES+CONTENT_FEATURES)[:width])
            models[name]=ranker.booster_;models[name].save_model(str(output/(name+'.txt')))
            fit_times[name]=time.perf_counter()-start
        print(f'Fit paired models: {len(groups)} groups/{sum(groups)} rows',flush=True)

    def flush():
        if not pending:return
        matrix=np.concatenate([entry[2] for entry in pending],axis=0)
        scores={name:predict_matrix(model,matrix,15 if name=='base15' else 18,config['threads']) for name,model in models.items()}
        offset=0
        for row,items,features,length in pending:
            rid,user,article,timestamp,target,target_time=row
            cold=first[target]>timestamp;available=metadata[target]['created']<=timestamp
            keys=['all','cold' if cold else 'warm','history<=5' if length<=5 else 'history>5']
            if available:keys.append('target-available')
            results={};hits=[]
            for name in models:
                ranking=sorted(zip(items,scores[name][offset:offset+len(items)]),key=lambda p:(-float(p[1]),p[0]))
                picked=[i for i,_ in ranking[:10]];results[name]=picked
                for key in keys:metrics[name][key].add(target,picked,user)
                hits.append(int(target in picked[:5]))
            paired_users[user]+=np.asarray([1,*hits])
            blocks[(timestamp-train_end)//(6*60*60*1000)]+=np.asarray([1,*hits])
            prediction.write(json.dumps(dict(request_id=rid,user_id=user,request_timestamp=timestamp,target_id=target,
                                              target_timestamp=target_time,target_available=available,zero_behavior_target=cold,
                                              history_length=length,article_ids=results))+'\n')
            offset+=len(items)
        pending.clear()

    try:
        rows=connection.execute('SELECT * FROM content_targets ORDER BY timestamp,ordinal')
        for timestamp,batch in groupby(rows,lambda r:r[3]):
            events=list(batch)
            for row in events:
                rid,user,article,_,target,target_time=row
                if target is None or target_time<=timestamp:continue
                valid=timestamp>train_end
                train=not valid and target_time<=train_end and rid%config['train_ordinal_modulus']==0
                if not(valid or train):continue
                if valid and not models:fit();state.trend.bucket=None
                history=(list(histories[user])+[(article,timestamp)])[-20:];length=lengths[user]+1
                _,items,base=state.query(history,timestamp,length)
                base=np.asarray(base,dtype=np.float32).reshape(-1,15)
                start=time.perf_counter();extra=vectors.features(items,history,timestamp)
                features_seconds+=time.perf_counter()-start
                x=np.concatenate((base,extra),axis=1)
                if not np.isfinite(x).all():raise ValueError('Nonfinite feature')
                candidate_digest.update(json.dumps([rid,items]).encode())
                if train:
                    selection['sampled_queries']+=1
                    if metadata[target]['created']>timestamp:selection['unavailable']+=1
                    elif target not in items:selection['not_retrieved']+=1
                    elif len(items)<2:selection['no_negative']+=1
                    else:
                        x.astype('<f4').tofile(stream)
                        np.asarray([int(i==target) for i in items],dtype='i1').tofile(labels)
                        base_digest.update(base.astype('<f4').tobytes());groups.append(len(items))
                else:
                    stats['requests']+=1;stats['candidate_hits']+=int(target in items);stats['candidates']+=len(items)
                    stats['unavailable_targets']+=int(metadata[target]['created']>timestamp)
                    if first[target]>timestamp:
                        stats['cold_requests']+=1;stats['cold_candidate_hits']+=int(target in items)
                    pending.append((row,items,x,length))
                    if len(pending)>=512:flush()
            for _,user,article,_,_,_ in events:
                histories[user].append((article,timestamp));lengths[user]+=1;state.observe(timestamp,article)
        flush()
        if not models:raise ValueError('No dev requests')
    finally:stream.close();labels.close();prediction.close()
    values=np.asarray(list(paired_users.values()));rng=np.random.default_rng(2026);bootstrap=[]
    for _ in range(400):
        sample=values[rng.integers(0,len(values),len(values))].sum(axis=0)
        bootstrap.append((sample[2]-sample[1])/sample[0])
    result=dict(models={name:{key:m.result() for key,m in mm.items()} for name,mm in metrics.items()},
                training_selection=dict(selection),training_groups=len(groups),training_rows=sum(groups),
                diagnostics=dict(stats),candidate_coverage=stats['candidate_hits']/stats['requests'],
                candidate_stream_sha256=candidate_digest.hexdigest(),shared_base_feature_sha256=base_digest.hexdigest(),
                fit_seconds=fit_times,content_feature_seconds=features_seconds,
                user_cluster_hr5_delta_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                six_hour_blocks={str(k):dict(requests=int(v[0]),base_hits=int(v[1]),content_hits=int(v[2]),delta=(v[2]-v[1])/v[0]) for k,v in blocks.items()},
                feature_names=FEATURES+CONTENT_FEATURES,
                training_matrix_sha256=file_hash(output/'train_features.f32'),labels_sha256=file_hash(output/'train_labels.i8'),
                paired_input_contract='same source matrix, labels, groups, candidates; base consumes first15 columns only')
    write_json(output/'metrics.json',result)
    return result
