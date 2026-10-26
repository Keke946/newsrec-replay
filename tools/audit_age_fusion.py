"""Post-hoc candidate-coverage diagnostic; cannot overturn primary HR5 gate."""
from collections import Counter,defaultdict
from itertools import groupby
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.serving import ReplayEngine,prime
from newsrec.time_ranker import FEATURES,CONFIG
from newsrec.fusion import fixed_count_fusion
from newsrec.data import write_json


class NoModel:
    def feature_name(self):return FEATURES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    m=json.loads((args.run/'manifest.json').read_text());p=m['protocol']
    with (args.reference/'predictions.jsonl').open() as f:ref={r['request_id']:r for r in map(json.loads,f)}
    preds={}
    for name in ['cosine','inbatch','mixed']:
        with (args.run/name/'predictions.jsonl').open() as f:preds[name]={r['request_id']:r for r in map(json.loads,f)}
        if preds[name].keys()!=ref.keys():raise ValueError('Cohort differs')
    conn=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
    engine=ReplayEngine(metadata,NoModel(),CONFIG);prime(engine,conn,p['train_end_ms'])
    counts={name:defaultdict(Counter) for name in preds};seen=set();digest=hashlib.sha256()
    rows=conn.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" '
                      'AND timestamp>? AND timestamp<=? ORDER BY timestamp,ordinal',(p['train_end_ms'],p['validation_end_ms']))
    try:
        for t,batch in groupby(rows,lambda r:r[0]):
            events=list(batch)
            for _,rid,user,article in events:
                if rid not in ref:continue
                r=ref[rid];seen.add(rid);history=(list(engine.histories[user])+[(article,t)])[-20:]
                if engine.lengths[user]+1!=r['history_length']:raise ValueError('History differs')
                trend,base,_=engine.state.query(history,t,r['history_length']);target=r['target_id'];hit=target in base
                digest.update(json.dumps([rid,base]).encode())
                for name,items in preds.items():
                    retrieved=items[rid]['article_ids']
                    if any(metadata[i]['created']>t for i in retrieved):raise ValueError('Future retrieval')
                    fused=fixed_count_fusion(base,trend,history,retrieved)
                    if len(fused)!=len(base):raise ValueError('Count differs')
                    f=target in fused
                    d=dict(requests=1,base_hits=int(hit),retrieval_hits_outside_base=int(target in retrieved and not hit),
                           fused_hits=int(f),gains=int(f and not hit),losses=int(hit and not f),
                           removed_ranker_top5_hits=int(target in r['article_ids']['base15'][:5] and not f))
                    for key in ['all','cold' if r['zero_behavior_target'] else 'warm']:counts[name][key].update(d)
            engine.ingest(t,[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in events])
        expected=json.loads((args.reference/'metrics.json').read_text())['diagnostics']['candidate_hits']
        if seen!=ref.keys() or counts['mixed']['all']['base_hits']!=expected:raise ValueError('Incomplete/base mismatch')
    finally:conn.close()
    out=dict(scope='preregistered age-retrieval candidate-budget gate; internal development only',
             cannot_claim_confirmation=True,per_request_candidate_count_matched=True,
             candidate_stream_sha256=digest.hexdigest(),methods={n:{k:dict(v) for k,v in groups.items()} for n,groups in counts.items()},
             limitations=['candidate coverage is not ranked HR','does not establish ranked HR gains','same internal development cohort'])
    write_json(args.run/'fixed_budget_fusion.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
