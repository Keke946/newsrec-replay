"""Reconstruct full causal candidate pools; reuse existing predictions, no fit."""
import argparse
from collections import Counter,defaultdict
from itertools import groupby
import json
from pathlib import Path
import random
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,source_hash
from newsrec.fusion import fixed_count_fusion
from newsrec.serving import ReplayEngine,prime
from newsrec.time_ranker import FEATURES


KEYS=('request_id','user_id','request_timestamp','target_id','target_timestamp','target_available','history_length')


def read_predictions(path,model):
    result={}
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['model']!=model:continue
            if row['request_id'] in result:raise ValueError('Duplicate request')
            result[row['request_id']]=row
    return result


def interval(values):
    values=sorted(values)
    def q(p):
        position=(len(values)-1)*p
        lo=int(position);fraction=position-lo
        return values[lo]*(1-fraction)+values[min(lo+1,len(values)-1)]*fraction
    return [q(.025),q(.975)]


def cluster_ci(users):
    pairs=list(users.values());rng=random.Random(2026);samples=[]
    for _ in range(400):
        chosen=rng.choices(pairs,k=len(pairs))
        samples.append(sum(row[1] for row in chosen)/sum(row[0] for row in chosen))
    return interval(samples)


class NoPredictionModel:
    def feature_name(self):return FEATURES
    def predict(self,*args,**kwargs):raise AssertionError('No model inference authorized in this diagnostic')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--ranker-run',type=Path,required=True)
    parser.add_argument('--tower-run',type=Path,required=True)
    parser.add_argument('--cosine-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists')
    args.output.mkdir(parents=True)
    start=time.perf_counter()
    manifest=json.loads((args.ranker_run/'manifest.json').read_text())
    protocol=manifest['protocol'];train_end=protocol['train_end_ms'];valid_end=protocol['validation_end_ms']
    if file_hash(args.database)!=manifest['database_sha256']:raise ValueError('Source DB differs')
    reference=read_predictions(args.ranker_run/'predictions.jsonl','lambdarank_time')
    trend_reference=read_predictions(args.ranker_run/'predictions.jsonl','rolling_trend')
    retrievals={name:read_predictions(run/'predictions.jsonl',model) for name,run,model in
                [('tower',args.tower_run,'two_tower_content_v0'),('cosine',args.cosine_run,'content_cosine')]}
    for name,predictions in retrievals.items():
        if predictions.keys()!=reference.keys():raise ValueError('Cohort differs: '+name)
        for rid,row in predictions.items():
            ref=reference[rid]
            if any(row[k]!=ref[k] for k in KEYS):raise ValueError('Request fields differ: '+name)
            if not train_end<row['request_timestamp']<row['target_timestamp']<=valid_end:
                raise ValueError('Outside validation protocol')
    connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in connection.execute('SELECT * FROM articles')}
    engine=ReplayEngine(metadata,NoPredictionModel(),manifest['config'])
    warmup=prime(engine,connection,train_end)
    groups={name:defaultdict(Counter) for name in retrievals}
    users={name:defaultdict(lambda:[0,0]) for name in retrievals}
    blocks={name:defaultdict(lambda:[0,0]) for name in retrievals}
    paired=Counter();seen=set()
    details=(args.output/'request_diagnostics.jsonl').open('w')
    try:
        rows=connection.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" '
                                'AND timestamp>? AND timestamp<=? ORDER BY timestamp,ordinal',(train_end,valid_end))
        for timestamp,batch in groupby(rows,lambda row:row[0]):
            events=list(batch)
            for _,rid,user,article in events:
                if rid not in reference:continue
                if rid in seen:raise ValueError('Repeated request')
                seen.add(rid);ref=reference[rid];target=ref['target_id']
                history=(list(engine.histories.get(user,()))+[(article,timestamp)])[-20:]
                length=engine.lengths[user]+1
                if ref['user_id']!=user or ref['request_timestamp']!=timestamp or ref['history_length']!=length:
                    raise ValueError('Raw click/history alignment differs')
                trend,base,_=engine.state.query(history,timestamp,length)
                if trend[:10]!=trend_reference[rid]['article_ids']:raise ValueError('Trend replay differs')
                if ref['target_available']!=(metadata[target]['created']<=timestamp):raise ValueError('Availability differs')
                base_set=set(base);base_hit=target in base_set
                actual_top5_hit=target in ref['article_ids'][:5]
                keys=['all','cold' if ref['zero_behavior_target'] else 'warm',
                      'history<=5' if length<=5 else 'history>5',
                      'target-available' if ref['target_available'] else 'target-unavailable',
                      'validation-first-half' if timestamp<=(train_end+valid_end)//2 else 'validation-second-half']
                result={}
                for name,predictions in retrievals.items():
                    items=predictions[rid]['article_ids']
                    if len(items)!=len(set(items)):raise ValueError('Duplicate retrieval')
                    if any(i not in metadata or metadata[i]['created']>timestamp for i in items):
                        raise ValueError('Unknown/future retrieved article')
                    fused=fixed_count_fusion(base,trend,history,items)
                    hit=target in fused;retrieved_hit=target in items
                    novel_hit=retrieved_hit and not base_hit
                    delta=int(hit)-int(base_hit)
                    record=dict(requests=1,base_candidate_hits=int(base_hit),fused_candidate_hits=int(hit),
                                gains=int(hit and not base_hit),losses=int(base_hit and not hit),
                                retrieval_top10_hits=int(retrieved_hit),retrieval_hits_outside_base=int(novel_hit),
                                oracle_union_hits=int(base_hit or retrieved_hit),
                                actual_ranker_top5_hits=int(actual_top5_hit),
                                actual_ranker_top5_hits_removed=int(actual_top5_hit and not hit),
                                base_candidate_count=len(base),fused_candidate_count=len(fused),
                                added_candidates=len(set(fused)-base_set))
                    for key in keys:groups[name][key].update(record)
                    users[name][user][0]+=1;users[name][user][1]+=delta
                    block=(timestamp-train_end)//(6*60*60*1000)
                    blocks[name][block][0]+=1;blocks[name][block][1]+=delta
                    result[name]=dict(base_hit=base_hit,fused_hit=hit,novel_hit=novel_hit,
                                      fused_candidates=fused,retrieved=items)
                paired['both_novel']+=int(result['tower']['novel_hit'] and result['cosine']['novel_hit'])
                paired['tower_only_novel']+=int(result['tower']['novel_hit'] and not result['cosine']['novel_hit'])
                paired['cosine_only_novel']+=int(result['cosine']['novel_hit'] and not result['tower']['novel_hit'])
                details.write(json.dumps(dict(request_id=rid,user_id=user,timestamp=timestamp,target=target,
                                             cold=ref['zero_behavior_target'],base=base,methods=result))+'\n')
            engine.ingest(timestamp,[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in events])
        if seen!=reference.keys():raise ValueError('Missing replay requests')
        expected=json.loads((args.ranker_run/'training.json').read_text())['validation_candidate_hits']
        if groups['tower']['all']['base_candidate_hits']!=expected:raise ValueError('Base coverage differs')
    finally:
        details.close();connection.close()
    output=dict(status='succeeded',scope='exploratory validation candidate coverage, not ranked HR',
                requests=len(seen),warmup_events=warmup,exact_request_alignment=True,
                exact_per_request_candidate_count=True,base_coverage_verified=True,
                pairwise_novel=dict(paired),methods={},final_region_scored=False,
                source_sha256=source_hash(),elapsed_seconds=time.perf_counter()-start)
    for name in groups:
        summary={}
        for key,c in groups[name].items():
            d=dict(c);n=c['requests']
            d.update(base_coverage=c['base_candidate_hits']/n,fused_coverage=c['fused_candidate_hits']/n,
                     coverage_delta=(c['gains']-c['losses'])/n,
                     oracle_union_coverage=c['oracle_union_hits']/n,
                     mean_candidates=c['base_candidate_count']/n)
            summary[key]=d
        output['methods'][name]=dict(groups=summary,user_cluster_delta_ci95=cluster_ci(users[name]),
                                     user_macro_coverage_delta=sum(d/n for n,d in users[name].values())/len(users[name]),
                                     six_hour_blocks={str(k):dict(requests=n,coverage_delta=d/n) for k,(n,d) in blocks[name].items()})
    output['inputs']={str(path):file_hash(path) for path in [args.database,args.ranker_run/'predictions.jsonl',
                     args.tower_run/'predictions.jsonl',args.cosine_run/'predictions.jsonl']}
    output['diagnostics_sha256']=file_hash(args.output/'request_diagnostics.jsonl')
    (args.output/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({name:value['groups']['all'] for name,value in output['methods'].items()},indent=2))


if __name__=='__main__':main()
