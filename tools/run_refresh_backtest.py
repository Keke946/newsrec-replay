"""CPU training-only expanding-origin refit backtest with explicit label delay."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from collections import Counter,defaultdict,deque
from itertools import groupby
import argparse,json,time,sqlite3,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.time_ranker import CausalCandidates,CONFIG,FEATURES
from newsrec.evaluation import RankingMetrics
from newsrec.data import file_hash,source_hash,write_json
ROOT=Path(__file__).resolve().parents[1];LIMIT=1507757054762;SIX=21600000;DELAY=60000

def active_cutoff(t,start,end):
    # initial model uses labels <=start; new models at schedule+delay
    n=(t-start-DELAY)//SIX
    return start if n<1 else start+n*SIX-1

def training_signature(groups):return tuple(g["request_id"] for g in groups)

def eligible_groups(groups,cutoff):
    return [g for g in groups if g['target_time']<=cutoff]

def load_protocol(confirm=False,final=False):
    if confirm and final:raise ValueError('Conflicting evaluation modes')
    name='v1_final_v1.json' if final else 'refresh_confirmation_v1.json' if confirm else 'training_backtest_v1.json'
    protocol=json.loads((ROOT/'protocols'/name).read_text())
    folds=protocol['folds']
    if final:
        if protocol.get('mode')!='final_evaluation' or len(folds)!=1 or folds[0]['fit_end_ms']!=1507980021008 or folds[0]['target_end_ms']<=1507980021008:raise ValueError('Final boundary differs')
    elif confirm:
        if [(f['fit_end_ms'],f['target_end_ms']) for f in folds]!=[(1507757054762,1507980021008)]:raise ValueError('Confirmation boundary differs')
    elif max(f['target_end_ms'] for f in folds)!=LIMIT:raise ValueError('Development crosses protected boundary')
    return protocol

def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group();mode.add_argument('--confirm-validation',action='store_true');mode.add_argument('--final-once',action='store_true');args=parser.parse_args()
    protocol=load_protocol(args.confirm_validation,args.final_once);folds=protocol['folds'];limit=max(f['target_end_ms'] for f in folds)
    keys=[0]+[f['fold'] for f in folds]
    out=ROOT/('runs/v1-final-evaluation' if args.final_once else 'runs/refresh-confirmation-v0' if args.confirm_validation else 'runs/refresh-backtest-v0');out.mkdir(exist_ok=False);(out/'models').mkdir()
    db=ROOT/'runs/recovered-audit/audit.sqlite';before=file_hash(db);clock=time.perf_counter()
    manifest=dict(status='running',protocol=protocol,source_sha256=source_hash(),database_sha256=before,config=CONFIG,delay_ms=DELAY,refresh_ms=SIX,query_clock='all observed source click proxies, before label eligibility')
    write_json(out/'manifest.json',manifest)
    conn=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
    if args.final_once:
        actual_end=conn.execute('SELECT max(timestamp) FROM clicks WHERE role="train"').fetchone()[0]
        if actual_end!=limit:raise ValueError('Natural final end differs from frozen protocol')
    metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
    conn.execute('CREATE TEMP TABLE targets AS SELECT ordinal,user,article,timestamp,lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time FROM clicks WHERE role="train" AND timestamp<=?',(limit,))
    state=CausalCandidates(metadata,CONFIG);hist=defaultdict(lambda:deque(maxlen=20));lengths=Counter();groups=[];offset=evaloffset=0;events_seen=0
    trainstream=(out/'train.f32').open('wb');labelstream=(out/'labels.i8').open('wb');evalstream=(out/'eval.f32').open('wb');metastream=(out/'requests.jsonl').open('w')
    for t,batch in groupby(conn.execute('SELECT * FROM targets ORDER BY timestamp,ordinal'),lambda r:r[3]):
        batch=list(batch)
        for rid,user,article,_,target,tt in batch:
            history=(list(hist[user])+[(article,t)])[-20:];length=lengths[user]+1
            _,items,features=state.query(history,t,length)
            x=np.asarray(features,dtype='<f4').reshape(-1,15);events_seen+=1
            if target is None or tt<=t:continue
            if rid%13==0 and metadata[target]['created']<=t and target in items and len(items)>=2:
                x.tofile(trainstream);np.asarray([int(i==target) for i in items],dtype='i1').tofile(labelstream)
                groups.append(dict(request_id=rid,request_time=t,target_time=tt,offset=offset,size=len(items)));offset+=len(items)
            for f in folds:
                if f['fit_end_ms']<t<=f['target_end_ms'] and tt<=f['target_end_ms']:
                    x.tofile(evalstream);metastream.write(json.dumps(dict(fold=f['fold'],request_id=rid,user_id=user,request_timestamp=t,target_id=target,target_timestamp=tt,target_available=metadata[target]['created']<=t,history_length=length,items=items,offset=evaloffset))+'\n');evaloffset+=len(items)
        for _,user,article,_,_,_ in batch:hist[user].append((article,t));lengths[user]+=1;state.observe(t,article)
        if events_seen%50000<len(batch):print('feature events',events_seen,flush=True)
    for f in [trainstream,labelstream,evalstream,metastream]:f.close()
    expected_source=conn.execute('SELECT count(*) FROM clicks WHERE role="train" AND timestamp<=?',(limit,)).fetchone()[0]
    conn.close();assert events_seen==expected_source
    if not args.confirm_validation and not args.final_once:assert events_seen==778836
    if args.final_once:assert events_seen==1112623
    generation=time.perf_counter()-clock;write_json(out/'groups.json',groups)
    tx=np.memmap(out/'train.f32',dtype='<f4',mode='r',shape=(offset,15));ty=np.memmap(out/'labels.i8',dtype='i1',mode='r',shape=(offset,));ex=np.memmap(out/'eval.f32',dtype='<f4',mode='r',shape=(evaloffset,15))
    models={};fits={};trained_signatures={}
    def model_at(cutoff):
        if cutoff in models:return models[cutoff]
        tick=time.perf_counter();chosen=eligible_groups(groups,cutoff);assert chosen
        assert all(g['request_time']<g['target_time']<=cutoff for g in chosen)
        signature=training_signature(chosen)
        if args.final_once and signature in trained_signatures:
            origin=trained_signatures[signature];b=models[origin];path=out/'models'/f'{cutoff}.txt';b.save_model(str(path));models[cutoff]=b
            fits[cutoff]=dict(cutoff=cutoff,groups=len(chosen),rows=sum(g['size'] for g in chosen),max_label_time=max(g['target_time'] for g in chosen),extract_and_fit_seconds=time.perf_counter()-tick,model_sha256=file_hash(path),reuse_of_cutoff=origin)
            write_json(out/'fits.json',fits);print('no-new-label reuse',cutoff,origin,flush=True);return b
        rows=np.concatenate([np.arange(g['offset'],g['offset']+g['size']) for g in chosen]);sizes=[g['size'] for g in chosen]
        c=CONFIG;model=lgb.LGBMRanker(objective='lambdarank',n_estimators=c['n_estimators'],num_leaves=c['num_leaves'],learning_rate=c['learning_rate'],min_child_samples=c['min_child_samples'],reg_lambda=c['reg_lambda'],lambdarank_truncation_level=c['lambdarank_truncation_level'],random_state=c['seed'],n_jobs=c['threads'],deterministic=True,force_col_wise=True,verbosity=-1)
        model.fit(tx[rows],ty[rows],group=sizes,feature_name=FEATURES);b=model.booster_;path=out/'models'/f'{cutoff}.txt';b.save_model(str(path));models[cutoff]=b
        fits[cutoff]=dict(cutoff=cutoff,groups=len(chosen),rows=len(rows),max_label_time=max(g['target_time'] for g in chosen),extract_and_fit_seconds=time.perf_counter()-tick,model_sha256=file_hash(path))
        trained_signatures[signature]=cutoff
        write_json(out/'fits.json',fits);print('fit',cutoff,len(chosen),flush=True);return b
    # All scheduled refits are executed, even in request-sparse periods.
    for f in folds:
        start,end=f['fit_end_ms'],f['target_end_ms'];model_at(start)
        for s in range(start+SIX,end-DELAY+1,SIX):model_at(s-1)
    metrics={k:{n:RankingMetrics() for n in ['fixed','refresh']} for k in keys};paired={k:defaultdict(lambda:np.zeros(3)) for k in keys};stats=defaultdict(Counter);seen=set()
    pred=(out/'predictions.jsonl').open('w');pending=[]
    def flush():
        if not pending:return
        for fold,rows in groupby(pending,lambda r:r['fold']):
            rows=list(rows);f=folds[fold-1];starts=[];parts=[];at=0
            for r in rows:parts.append(np.asarray(ex[r['offset']:r['offset']+len(r['items'])]));starts.append((at,at+len(r['items'])));at+=len(r['items'])
            x=np.concatenate(parts);fixed=model_at(f['fit_end_ms']).predict(x,num_threads=4);refresh=np.empty(len(x))
            cutoffs=[active_cutoff(r['request_timestamp'],f['fit_end_ms'],f['target_end_ms']) for r in rows]
            for cut in set(cutoffs):
                ids=np.concatenate([np.arange(a,b) for (a,b),c in zip(starts,cutoffs) if c==cut]);refresh[ids]=model_at(cut).predict(x[ids],num_threads=4) if len(ids) else []
            for r,(a,b),cut in zip(rows,starts,cutoffs):
                rid=r['request_id'];assert rid not in seen;seen.add(rid);items=r['items'];target=r['target_id'];picks={};hits=[]
                for name,scores in [('fixed',fixed[a:b]),('refresh',refresh[a:b])]:
                    picked=[i for i,_ in sorted(zip(items,scores),key=lambda z:(-float(z[1]),z[0]))[:10]];picks[name]=picked;hits.append(int(target in picked[:5]))
                    for k in [0,fold]:metrics[k][name].add(target,picked,r['user_id'])
                for k in [0,fold]:paired[k][r['user_id']]+=np.asarray([1,*hits]);stats[k].update(requests=1,candidate_hits=int(target in items),unavailable=int(not r['target_available']),gains=int(hits[1] and not hits[0]),losses=int(hits[0] and not hits[1]))
                record={k:v for k,v in r.items() if k not in ['items','offset']};record.update(article_ids=picks,refresh_cutoff=cut);pred.write(json.dumps(record)+'\n')
        pending.clear()
    for r in map(json.loads,(out/'requests.jsonl').open()):
        pending.append(r)
        if len(pending)>=512:flush()
    flush();pred.close();result={}
    for k in keys:
        values=np.array(list(paired[k].values()));rng=np.random.default_rng(2026);boots=[]
        for _ in range(400):
            s=values[rng.integers(0,len(values),len(values))].sum(0);boots.append((s[2]-s[1])/s[0])
        result[k]=dict(metrics={n:m.result() for n,m in metrics[k].items()},counts=dict(stats[k]),user_cluster_ci95=np.quantile(boots,[.025,.975]).tolist())
    refreshfits=[v for cut,v in fits.items() if cut not in {f['fit_end_ms'] for f in folds}];gate=all(result[k]['metrics']['refresh']['hit_rate@5']>result[k]['metrics']['fixed']['hit_rate@5'] for k in keys[1:]) and result[0]['user_cluster_ci95'][0]>0 and all(v['extract_and_fit_seconds']<=60 for v in refreshfits)
    summary=dict(folds=result,gate_passed=bool(gate),total_models=len(fits),refresh_models=len(refreshfits),cumulative_extract_fit_seconds=sum(v['extract_and_fit_seconds'] for v in fits.values()),feature_generation_seconds=generation,source_events=events_seen,shared_candidate_features=True,final_scored=args.final_once,old_validation_scored=args.confirm_validation)
    if args.final_once:
        summary.pop('gate_passed',None);summary['selected_policy']='refresh fixed before scoring; no policy reselection from final results'
        summary['actual_fits']=sum('reuse_of_cutoff' not in f for f in fits.values());summary['no_new_label_reuses']=sum('reuse_of_cutoff' in f for f in fits.values())
        summary['final_protocol_sha256']=file_hash(ROOT/'protocols/v1_final_v1.json')
    assert file_hash(db)==before;write_json(out/'summary.json',summary);manifest.update(status='succeeded',wall_seconds=time.perf_counter()-clock);write_json(out/'manifest.json',manifest);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
