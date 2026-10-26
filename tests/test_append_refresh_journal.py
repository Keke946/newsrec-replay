import json,os,subprocess,sys,unittest
from pathlib import Path
import test_refresh_replay as fixtures
from newsrec.append_refresh_journal import AppendRefreshJournal
from newsrec.refresh_journal import identity_for,RefreshJournalError,canonical
from newsrec.serving import ContractError

class AppendTests(unittest.TestCase):
    setUp=fixtures.RefreshReplayTests.setUp;tearDown=fixtures.RefreshReplayTests.tearDown
    write=fixtures.RefreshReplayTests.write;registry=fixtures.RefreshReplayTests.registry;engine=fixtures.RefreshReplayTests.engine;event=fixtures.RefreshReplayTests.event
    def factory(self):
        e=self.engine();e.process_batch(99,[self.event()],[]);return e
    def journal(self,identity=None):return AppendRefreshJournal(self.root/'log.jsonl',self.factory,identity or identity_for(self.registry(),99))
    def test_restart_switch_exact_retry(self):
        j=self.journal();a=j.process_batch(101,[self.event(2)]);b=j.process_batch(self.switch,[self.event(3)]);j.close()
        k=self.journal();self.assertEqual(k.process_batch(101,[self.event(2)]),a);self.assertEqual(k.process_batch(self.switch,[self.event(3)]),b);self.assertEqual(len(k.records),2)
        with self.assertRaises(ContractError):k.process_batch(101,[self.event(2,a=2)])
        k.close()
    def test_historical_fallback_stays_pinned(self):
        p=self.root/'next.txt';raw=p.read_bytes();p.unlink();j=self.journal();a=j.process_batch(self.switch,[self.event(2)]);j.close();p.write_bytes(raw)
        k=self.journal();self.assertEqual(k.process_batch(self.switch,[self.event(2)]),a);self.assertEqual(a['model_cutoff'],100);self.assertNotEqual(k.process_batch(self.switch+1,[self.event(3)])['model_cutoff'],100);k.close()
    def test_before_append_failure(self):
        j=self.journal();original=j._persist
        def fail(*a):raise OSError('before append')
        j._persist=fail
        with self.assertRaises(RefreshJournalError):j.process_batch(101,[self.event(2)])
        self.assertEqual(j.observed_through,99);j._persist=original;j.process_batch(101,[self.event(2)]);j.close()
    def test_complete_append_ack_loss(self):
        j=self.journal();original=j._persist
        def fail(*a):original(*a);raise OSError('ack lost')
        j._persist=fail
        with self.assertRaises(RefreshJournalError):j.process_batch(101,[self.event(2)])
        j._persist=original;self.assertEqual(j.process_batch(101,[self.event(2)])['model_cutoff'],100);self.assertEqual(len(j.records),1);j.close()
    def test_partial_unacknowledged_tail_preserved(self):
        j=self.journal();a=j.process_batch(101,[self.event(2)]);j.close()
        with (self.root/'log.jsonl').open('ab') as f:f.write(b'{"partial":');f.flush();os.fsync(f.fileno())
        k=self.journal();self.assertEqual(k.process_batch(101,[self.event(2)]),a);self.assertEqual(len(list(self.root.glob('*.uncommitted-*'))),1);k.close()
    def test_complete_corrupt_line_is_rejected(self):
        j=self.journal();j.process_batch(101,[self.event(2)]);j.close();p=self.root/'log.jsonl';lines=p.read_bytes().splitlines(True);r=json.loads(lines[1]);r['body'][3]='{}';lines[1]=canonical(r).encode()+b'\n';p.write_bytes(b''.join(lines))
        with self.assertRaises(RefreshJournalError):self.journal()
    def test_owner_identity_and_no_sqlite_migration(self):
        j=self.journal()
        with self.assertRaises(RefreshJournalError):self.journal()
        j.close();identity=identity_for(self.registry(),99);identity['database_sha256']='other'
        with self.assertRaises(RefreshJournalError):self.journal(identity)
        p=self.root/'foreign';p.write_bytes(b'SQLite format 3\x00');before=p.read_bytes()
        with self.assertRaises(RefreshJournalError):AppendRefreshJournal(p,self.factory,identity_for(self.registry(),99))
        self.assertEqual(p.read_bytes(),before)
    def test_missing_committed_refresh_fails(self):
        j=self.journal();j.process_batch(self.switch,[self.event(2)]);j.close();(self.root/'next.txt').unlink()
        with self.assertRaises(ContractError):self.journal()
    def test_real_abrupt_child_exit(self):
        code='''import os,sys
from pathlib import Path
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine
from newsrec.refresh_journal import identity_for
from newsrec.append_refresh_journal import AppendRefreshJournal
p=Path(sys.argv[1]);meta={i:dict(category=i,created=0,words=20) for i in [1,2,3]}
def factory():
 e=RefreshReplayEngine(meta,ModelRegistry(p/'registry.json','fixture'));e.process_batch(99,[dict(event_id=1,user_id=1,article_id=1)],[]);return e
j=AppendRefreshJournal(p/'log.jsonl',factory,identity_for(ModelRegistry(p/'registry.json','fixture'),99));j.process_batch(101,[dict(event_id=2,user_id=1,article_id=1)]);os._exit(23)
'''
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'));r=subprocess.run([sys.executable,'-c',code,str(self.root)],env=env,capture_output=True,timeout=30);self.assertEqual(r.returncode,23,r.stderr.decode())
        j=self.journal();self.assertEqual(j.replayed_batches,1);self.assertEqual(j.process_batch(101,[self.event(2)])['model_cutoff'],100);j.close()
