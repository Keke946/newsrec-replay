"""Independent source/cohort, label availability and paired prediction checks."""
import argparse,json,sqlite3,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,write_json
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();modes=parser.add_mutually_exclusive_group();modes.add_argument('--confirm-validation',action='store_true');modes.add_argument('--final-once',action='store_true');args=parser.parse_args()
run=root/('runs/v1-final-evaluation' if args.final_once else 'runs/refresh-confirmation-v0' if args.confirm_validation else 'runs/refresh-backtest-v0');m=json.loads((run/'manifest.json').read_text());p=m['protocol'];folds=p['folds']
assert m['status']=='succeeded'
conn=sqlite3.connect(f'file:{root}/runs/recovered-audit/audit.sqlite?mode=ro',uri=True)
created=dict(conn.execute('SELECT article,created FROM articles'))
limit=max(f['target_end_ms'] for f in folds)
if args.final_once:
 assert len(folds)==1 and folds[0]['fit_end_ms']==1507980021008
 assert conn.execute('SELECT max(timestamp) FROM clicks WHERE role="train"').fetchone()[0]==limit
conn.execute('CREATE TEMP TABLE independent AS SELECT ordinal,user,timestamp,lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time,rank() OVER(PARTITION BY user ORDER BY timestamp) history_length FROM clicks WHERE role="train" AND timestamp<=?',(limit,))
expected={}
for f in folds:
 for row in conn.execute('SELECT * FROM independent WHERE timestamp>? AND timestamp<=? AND target_time>timestamp AND target_time<=?',(f['fit_end_ms'],f['target_end_ms'],f['target_end_ms'])):expected[row[0]]=(f['fold'],row)
conn.close();seen=set();counts=Counter();hits=Counter();initial_matches=0;fits=json.loads((run/'fits.json').read_text());groups=json.loads((run/'groups.json').read_text())
for cutoff,fit in fits.items():
 eligible=[g for g in groups if g['target_time']<=int(cutoff)]
 assert len(eligible)==fit['groups'] and sum(g['size'] for g in eligible)==fit['rows']
 assert max(g['target_time'] for g in eligible)==fit['max_label_time']<=int(cutoff)
 assert file_hash(run/'models'/f'{cutoff}.txt')==fit['model_sha256']
 if 'reuse_of_cutoff' in fit:
  origin=fit['reuse_of_cutoff'];prior=[g for g in groups if g['target_time']<=origin]
  assert [g['request_id'] for g in prior]==[g['request_id'] for g in eligible]
  assert fits[str(origin)]['model_sha256']==fit['model_sha256']
for r in map(json.loads,(run/'predictions.jsonl').open()):
 rid=r['request_id'];assert rid not in seen;seen.add(rid);fold,row=expected[rid];_,u,t,target,tt,length=row
 assert (r['fold'],r['user_id'],r['request_timestamp'],r['target_id'],r['target_timestamp'],r['history_length'])==(fold,u,t,target,tt,length)
 assert r['target_available']==(created[target]<=t)
 f=folds[fold-1];cut=r['refresh_cutoff'];assert cut==f['fit_end_ms'] or (cut+1-f['fit_end_ms'])%21600000==0 and cut+1+60000<=t
 n=(t-f['fit_end_ms']-60000)//21600000;latest=f['fit_end_ms'] if n<1 else f['fit_end_ms']+n*21600000-1
 assert cut==latest and fits[str(cut)]['max_label_time']<=cut
 if cut==f['fit_end_ms']:assert r['article_ids']['fixed']==r['article_ids']['refresh'];initial_matches+=1
 for n,ids in r['article_ids'].items():
  assert len(ids)==len(set(ids))<=10 and all(created[i]<=t for i in ids)
  hits[(fold,n)]+=int(target in ids[:5])
 counts[fold]+=1
assert seen==expected.keys()
if args.confirm_validation:assert sum(counts.values())==114855
elif not args.final_once:assert counts[3]==107904
summary=json.loads((run/'summary.json').read_text())
if args.final_once:
 assert summary['final_scored'] and summary['final_protocol_sha256']==file_hash(root/'protocols/v1_final_v1.json')
 assert 'gate_passed' not in summary
for f in [f['fold'] for f in folds]:
 for n in ['fixed','refresh']:assert abs(hits[(f,n)]/counts[f]-summary['folds'][str(f)]['metrics'][n]['hit_rate@5'])<1e-12
assert file_hash(root/'runs/recovered-audit/audit.sqlite')==m['database_sha256']
result=dict(source_cohort_complete=True,requests=len(seen),fold_counts=dict(counts),all_request_fields_and_history_verified=True,label_arrival_and_model_identity_verified=True,initial_shared_model_top10_requests=initial_matches,source_database_unchanged=True)
result['evidence_sha256']={name:file_hash(run/name) for name in ['train.f32','labels.i8','groups.json','predictions.jsonl']}
write_json(run/'verification.json',result);print(json.dumps(result,indent=2))
