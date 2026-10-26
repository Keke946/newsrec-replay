"""One fixed real-log full-vs-snapshot restore and next-five-minute parity."""
from itertools import groupby
from contextlib import closing
from pathlib import Path
import hashlib,json,shutil,sqlite3,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.state_snapshot import SnapshotRefreshJournal,capture
from newsrec.refresh_journal import identity_for,canonical
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine,prime_refresh
from newsrec.data import file_hash,write_json
root=Path(__file__).resolve().parents[1];out=root/'runs/v1-snapshot-benchmark';out.mkdir(exist_ok=False)
original=root/'runs/refresh-journal-append-v0/batches.jsonl';before=file_hash(original);shutil.copyfile(original,out/'batches.jsonl')
db=root/'runs/recovered-audit/audit.sqlite';digest=file_hash(db);rp=root/'runs/refresh-runtime-v0/registry.json';registry=ModelRegistry(rp,digest)
with closing(sqlite3.connect(f'file:{db}?mode=ro',uri=True)) as conn:
 metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
 through=conn.execute('SELECT max(timestamp) FROM clicks WHERE role="train" AND timestamp<=?',(registry.start,)).fetchone()[0]
def empty_factory():return RefreshReplayEngine(metadata,ModelRegistry(rp,digest))
def factory():
 e=empty_factory()
 with closing(sqlite3.connect(f'file:{db}?mode=ro',uri=True)) as conn:prime_refresh(e,conn,registry.start)
 return e
identity=identity_for(registry,through);snapshot=out/'state.json.gz'
def construct():return SnapshotRefreshJournal(out/'batches.jsonl',factory,identity,empty_factory=empty_factory,snapshot_path=snapshot)
def fingerprint(e):return hashlib.sha256(canonical(json.loads(canonical(capture(e)))).encode()).hexdigest()
tick=time.perf_counter();journal=construct();full=time.perf_counter()-tick;assert journal.snapshot_status=='none';state=fingerprint(journal.engine);control=journal.engine;prefix=len(journal.records)
tick=time.perf_counter();checkpoint=journal.checkpoint();write_seconds=time.perf_counter()-tick;last=journal.observed_through;first=json.loads(journal.records[0][2]);first_response=json.loads(journal.records[0][3]);journal.close()
tick=time.perf_counter();journal=construct();fast=time.perf_counter()-tick;assert journal.snapshot_status=='loaded' and journal.replayed_batches==0 and fingerprint(journal.engine)==state
requests=events=batches=0
with closing(sqlite3.connect(f'file:{db}?mode=ro',uri=True)) as conn:
 for t,rows in groupby(conn.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" AND timestamp>? AND timestamp<=? ORDER BY timestamp,ordinal',(last,last+300000)),lambda r:r[0]):
  rows=list(rows);batch=[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in rows];a=control.process_batch(t,batch);b=journal.process_batch(t,batch);assert a==b;requests+=len(b['predictions']);events+=len(rows);batches+=1
assert requests>0 and fingerprint(control)==fingerprint(journal.engine)
journal.checkpoint();current=journal.observed_through;journal.close()
tick=time.perf_counter();journal=construct();again=time.perf_counter()-tick;assert journal.snapshot_status=='loaded' and journal.process_batch(**first)==first_response and journal.observed_through==current;journal.close()
assert file_hash(db)==digest and file_hash(original)==before
result=dict(status='succeeded',prefix_batches=prefix,snapshot=checkpoint,checkpoint_seconds=write_seconds,full_constructor_seconds=full,snapshot_constructor_seconds=fast,second_snapshot_constructor_seconds=again,all_mutable_state_exact=True,next_requests_exact=requests,next_source_events=events,next_batches=batches,old_retry_does_not_rewind=True,source_db_and_original_journal_unchanged=True,scope='one paired real-prefix timing observation, not production latency or repeated benchmark; logs retained, no compaction')
write_json(out/'verification.json',result);print(json.dumps(result,indent=2))
