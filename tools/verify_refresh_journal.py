"""Real seven-hour source prefix, two restarts across first refresh activation."""
from itertools import groupby
import argparse,json,sqlite3,sys,time
from contextlib import closing
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,write_json
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine,prime_refresh
from newsrec.refresh_journal import identity_for
from newsrec.append_refresh_journal import AppendRefreshJournal as RefreshJournal
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('runs/refresh-journal-v0'));args=parser.parse_args()
root=Path(__file__).resolve().parents[1];out=root/args.output;out.mkdir(exist_ok=False)
db=root/'runs/recovered-audit/audit.sqlite';registry_path=root/'runs/refresh-runtime-v0/registry.json';digest=file_hash(db)
registry=ModelRegistry(registry_path,digest);start=registry.start;activation=start+21600000+60000;end=start+7*3600000
conn=sqlite3.connect(f'file:{db}?mode=ro',uri=True);metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
through=conn.execute('SELECT max(timestamp) FROM clicks WHERE role="train" AND timestamp<=?',(start,)).fetchone()[0]
def factory():
    if file_hash(db)!=digest:raise ValueError('Source DB changed')
    e=RefreshReplayEngine(metadata,ModelRegistry(registry_path,digest))
    with closing(sqlite3.connect(f'file:{db}?mode=ro',uri=True)) as c:prime_refresh(e,c,start)
    return e
identity=identity_for(registry,through);write_json(out/'identity.json',identity)
control=factory();journal=RefreshJournal(out/'batches.jsonl',factory,identity)
ref={r['request_id']:r for r in map(json.loads,(root/'runs/refresh-confirmation-v0/predictions.jsonl').open())};seen=set();events_count=batches=0;restarts=[];switched=False;first=None
clock=time.perf_counter()
for t,group in groupby(conn.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" AND timestamp>? AND timestamp<=? ORDER BY timestamp,ordinal',(start,end)),lambda r:r[0]):
    rows=list(group);events=[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in rows];score=[o for _,o,_,_ in rows if o in ref]
    if not switched and t>=activation:
        journal.close();tick=time.perf_counter();journal=RefreshJournal(out/'batches.jsonl',factory,identity);restarts.append(dict(stage='before_first_activation',seconds=time.perf_counter()-tick))
    expected=control.process_batch(t,events,score);actual=journal.process_batch(t,events,score);assert actual==expected
    if first is None:first=(t,events,score,actual)
    for p in actual['predictions']:
        rid=p['event_id'];r=ref[rid];assert p['article_ids']==r['article_ids']['refresh'] and p['history_length']==r['history_length'] and actual['model_cutoff']==r['refresh_cutoff'];seen.add(rid)
    if not switched and t>=activation:
        assert actual['model_cutoff']!=start
        journal.close();tick=time.perf_counter();journal=RefreshJournal(out/'batches.jsonl',factory,identity);restarts.append(dict(stage='after_first_activation',seconds=time.perf_counter()-tick))
        assert journal.process_batch(t,events,score)==expected;switched=True
    batches+=1;events_count+=len(rows)
# This old retry must not rewind current state or create another log row.
previous=journal.observed_through;assert journal.process_batch(first[0],first[1],first[2])==first[3];assert journal.observed_through==previous
expected_ids={rid for rid,r in ref.items() if start<r['request_timestamp']<=end};assert seen==expected_ids and switched
recovered=journal.replayed_batches;last=journal.observed_through;switches=journal.engine.switches;journal.close();conn.close()
count=len(journal.records);assert count==batches
transitions=[json.loads(row[4]) for row in journal.records];actual_switches=[s for s in transitions if s['from_cutoff']!=s['to_cutoff']]
assert file_hash(db)==digest
result=dict(status='succeeded',scope='fixed real seven-hour validation prefix; no training/new accuracy experiment',source_events=events_count,requests=len(seen),committed_batches=count,complete_response_parity=True,offline_top10_history_cutoff_exact=True,restarts=restarts,initial_and_refresh_switches=actual_switches,integrity_check='full record chain and replay outputs verified',last_timestamp=last,wall_seconds_excluding_initial_factories=time.perf_counter()-clock,source_database_unchanged=True,abrupt_exit='separate synthetic child-process test, not real-prefix crash')
write_json(out/'verification.json',result);print(json.dumps(result,indent=2))
