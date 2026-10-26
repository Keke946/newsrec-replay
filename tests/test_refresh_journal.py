import json,os,sqlite3,subprocess,sys,unittest
from pathlib import Path
import test_refresh_replay as fixtures
from newsrec.refresh_journal import RefreshJournal,RefreshJournalError,identity_for,canonical
from newsrec.serving import ContractError

class RefreshJournalTests(unittest.TestCase):
    setUp=fixtures.RefreshReplayTests.setUp
    tearDown=fixtures.RefreshReplayTests.tearDown
    write=fixtures.RefreshReplayTests.write
    registry=fixtures.RefreshReplayTests.registry
    engine=fixtures.RefreshReplayTests.engine
    event=fixtures.RefreshReplayTests.event
    def factory(self):
        e=self.engine();e.process_batch(99,[self.event()],[]);return e
    def journal(self,factory=None,identity=None):
        return RefreshJournal(self.root/'log.sqlite',factory or self.factory,identity or identity_for(self.registry(),99))
    def test_restart_switch_and_retry(self):
        j=self.journal();a=j.process_batch(101,[self.event(2)]);b=j.process_batch(self.switch,[self.event(3)])
        j.close();k=self.journal()
        self.assertEqual(k.process_batch(101,[self.event(2)]),a);self.assertEqual(k.process_batch(self.switch,[self.event(3)]),b)
        self.assertEqual(k.replayed_batches,2)
        with self.assertRaises(ContractError):k.process_batch(101,[self.event(2,a=2)])
        k.close()
    def test_before_write_failure(self):
        j=self.journal();original=j._persist
        def fail(*args):raise sqlite3.OperationalError('injected')
        j._persist=fail
        with self.assertRaises(RefreshJournalError):j.process_batch(101,[self.event(2)])
        self.assertEqual(j.observed_through,99);j._persist=original;j.process_batch(101,[self.event(2)]);j.close()
    def test_insert_before_commit_failure(self):
        j=self.journal();original=j._persist
        def fail(*args):
            j.connection.execute('INSERT INTO refresh_batches(timestamp,payload,result,switch) VALUES(?,?,?,?)',args)
            raise sqlite3.OperationalError('before commit')
        j._persist=fail
        with self.assertRaises(RefreshJournalError):j.process_batch(101,[self.event(2)])
        self.assertEqual(j.connection.execute('SELECT count(*) FROM refresh_batches').fetchone()[0],0)
        self.assertEqual(j.observed_through,99);j._persist=original;j.process_batch(101,[self.event(2)]);j.close()
    def test_commit_then_ack_loss(self):
        j=self.journal();original=j._persist
        def fail(*args):original(*args);raise sqlite3.OperationalError('commit response lost')
        j._persist=fail
        with self.assertRaises(RefreshJournalError):j.process_batch(101,[self.event(2)])
        self.assertEqual(j.observed_through,101);j._persist=original;r=j.process_batch(101,[self.event(2)])
        self.assertEqual(r['model_cutoff'],100);self.assertEqual(j.connection.execute('SELECT count(*) FROM refresh_batches').fetchone()[0],1);j.close()
    def test_fallback_is_pinned_after_artifact_arrives(self):
        path=self.root/'next.txt';raw=path.read_bytes();path.unlink()
        j=self.journal();old=j.process_batch(self.switch,[self.event(2)]);self.assertEqual(old['model_cutoff'],100);j.close()
        path.write_bytes(raw);k=self.journal();self.assertEqual(k.process_batch(self.switch,[self.event(2)]),old)
        new=k.process_batch(self.switch+1,[self.event(3)]);self.assertEqual(new['model_cutoff'],self.start+21600000-1);k.close()
    def test_result_corruption(self):
        j=self.journal();j.process_batch(101,[self.event(2)]);j.close()
        with sqlite3.connect(self.root/'log.sqlite') as c:
            data=json.loads(c.execute('SELECT result FROM refresh_batches').fetchone()[0]);data['accepted']=99
            c.execute('UPDATE refresh_batches SET result=?',(canonical(data),))
        with self.assertRaises(RefreshJournalError):self.journal()
    def test_switch_corruption(self):
        j=self.journal();j.process_batch(101,[self.event(2)]);j.close()
        with sqlite3.connect(self.root/'log.sqlite') as c:c.execute('UPDATE refresh_batches SET switch=?',(canonical(dict(from_cutoff=99,to_cutoff=100)),))
        with self.assertRaises(RefreshJournalError):self.journal()
    def test_missing_committed_model_rejects_recovery(self):
        j=self.journal();j.process_batch(self.switch,[self.event(2)]);j.close();(self.root/'next.txt').unlink()
        with self.assertRaises(ContractError):self.journal()
    def test_owner_identity_and_source_database_protection(self):
        j=self.journal()
        with self.assertRaises(RefreshJournalError):self.journal()
        j.close();identity=identity_for(self.registry(),99);identity['database_sha256']='different'
        with self.assertRaises(RefreshJournalError):self.journal(identity=identity)
        unrelated=self.root/'business.sqlite'
        with sqlite3.connect(unrelated) as c:c.execute('CREATE TABLE business(value)');c.execute('INSERT INTO business VALUES(1)')
        with self.assertRaises(RefreshJournalError):RefreshJournal(unrelated,self.factory,identity_for(self.registry(),99))
        with sqlite3.connect(unrelated) as c:self.assertEqual(c.execute('SELECT value FROM business').fetchall(),[(1,)])
    def test_pinned_future_and_rollback_rejected(self):
        r=self.registry()
        with self.assertRaises(ContractError):r.select_pinned(101,self.start+21600000-1)
        r.select_pinned(self.switch,self.start+21600000-1)
        with self.assertRaises(ContractError):r.select_pinned(self.switch+1,100)
    def test_recovery_failure_closes(self):
        j=self.journal()
        def badfactory():raise RuntimeError('cannot rebuild')
        def badwrite(*args):raise sqlite3.OperationalError('cannot write')
        j.factory=badfactory;j._persist=badwrite
        with self.assertRaises(RefreshJournalError):j.process_batch(101,[self.event(2)])
        self.assertTrue(j.closed)
    def test_api_write_failure_is_503(self):
        from fastapi.testclient import TestClient
        from newsrec.refresh_api import create_refresh_app
        j=self.journal();client=TestClient(create_refresh_app(j));original=j._persist
        def fail(*args):raise sqlite3.OperationalError('fail')
        j._persist=fail
        self.assertEqual(client.post('/replay/batch',json=dict(timestamp=101,events=[self.event(2)])).status_code,503)
        j._persist=original;self.assertEqual(client.post('/replay/batch',json=dict(timestamp=101,events=[self.event(2)])).status_code,200);j.close()
    def test_child_abrupt_exit_after_commit(self):
        code='''import os,sys
from pathlib import Path
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine
from newsrec.refresh_journal import RefreshJournal,identity_for
p=Path(sys.argv[1]);meta={i:dict(category=i,created=0,words=20) for i in [1,2,3]}
def factory():
 e=RefreshReplayEngine(meta,ModelRegistry(p/'registry.json','fixture'));e.process_batch(99,[dict(event_id=1,user_id=1,article_id=1)],[]);return e
j=RefreshJournal(p/'log.sqlite',factory,identity_for(ModelRegistry(p/'registry.json','fixture'),99))
j.process_batch(101,[dict(event_id=2,user_id=1,article_id=1)])
os._exit(23)
'''
        env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
        result=subprocess.run([sys.executable,'-c',code,str(self.root)],env=env,capture_output=True,timeout=30)
        self.assertEqual(result.returncode,23,result.stderr.decode());j=self.journal()
        self.assertEqual(j.process_batch(101,[self.event(2)])['model_cutoff'],100);self.assertEqual(j.replayed_batches,1);j.close()
