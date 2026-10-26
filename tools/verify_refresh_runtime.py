"""Create bound immutable registry and replay all source click proxies."""
from itertools import groupby
import argparse,json,sqlite3,sys,time,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,write_json
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine,metadata_hash,encoder_hashes
from newsrec.time_ranker import FEATURES,CONFIG
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/refresh-runtime-v0'));args=parser.parse_args()
root=Path(__file__).resolve().parents[1];source=root/'runs/refresh-confirmation-v0';out=root/args.output;out.mkdir(exist_ok=False)
m=json.loads((source/'manifest.json').read_text());assert m['status']=='succeeded'
assert encoder_hashes()==json.loads((root/'protocols/v1_encoder_sha256.json').read_text())
start,end=1507757054762,1507980021008;db=root/'runs/recovered-audit/audit.sqlite';assert file_hash(db)==m['database_sha256']
conn=sqlite3.connect(f'file:{db}?mode=ro',uri=True);metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
fits=json.loads((source/'fits.json').read_text());entries=[]
# Registry references own model files inside its root; copied, never legacy pickle.
import shutil
(out/'models').mkdir()
for cut,fit in sorted(fits.items(),key=lambda kv:int(kv[0])):
    cut=int(cut);activate=start if cut==start else cut+1+60000
    path=out/'models'/f'{cut}.txt';shutil.copyfile(source/'models'/path.name,path)
    entries.append(dict(cutoff=cut,activate_ms=activate,max_label_time=fit['max_label_time'],file='models/'+path.name,sha256=fit['model_sha256']))
write_json(out/'registry.json',dict(schema='precomputed-refresh-replay-v1',start_ms=start,database_sha256=m['database_sha256'],metadata_sha256=metadata_hash(metadata),encoder_sha256=encoder_hashes(),feature_names=FEATURES,config=CONFIG,models=entries))
registry=ModelRegistry(out/'registry.json',m['database_sha256']);engine=RefreshReplayEngine(metadata,registry)
ref={r['request_id']:r for r in map(json.loads,(source/'predictions.jsonl').open())};seen=set();count=0;clock=time.perf_counter();output=(out/'responses.jsonl').open('w')
for t,batch in groupby(conn.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" AND timestamp<=? ORDER BY timestamp,ordinal',(end,)),lambda r:r[0]):
    batch=list(batch);events=[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in batch];wanted=[o for _,o,_,_ in batch if o in ref]
    response=engine.process_batch(t,events,wanted)
    for p in response['predictions']:
        rid=p['event_id'];r=ref[rid];assert rid not in seen;seen.add(rid)
        assert (p['user_id'],t,p['history_length'])==(r['user_id'],r['request_timestamp'],r['history_length'])
        assert p['article_ids']==r['article_ids']['refresh'] and response['model_cutoff']==r['refresh_cutoff'],f'Prediction or cutoff differs at {rid}'
        output.write(json.dumps(dict(response,request_id=rid,predictions=[p]))+'\n')
    count+=len(batch)
    if count%100000<len(batch):print('replay source events',count,flush=True)
output.close();conn.close();assert seen==ref.keys() and len(seen)==114855;assert file_hash(db)==m['database_sha256']
result=dict(status='succeeded',source_events=count,requests=len(seen),all_top10_history_cutoff_exact=True,registry_sha256=registry.version,source_database_unchanged=True,wall_seconds=time.perf_counter()-clock,switches=engine.switches,mode='precomputed replay; not live training, journal or HTTP load test')
write_json(out/'parity.json',result);print(json.dumps(result,indent=2))
