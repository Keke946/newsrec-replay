import gzip,json,unittest
from unittest.mock import patch
import test_refresh_replay as fixtures
from newsrec.state_snapshot import SnapshotRefreshJournal,capture,restore_state
from newsrec.refresh_journal import identity_for,RefreshJournalError,canonical

class SnapshotTests(unittest.TestCase):
    setUp=fixtures.RefreshReplayTests.setUp;tearDown=fixtures.RefreshReplayTests.tearDown
    write=fixtures.RefreshReplayTests.write;registry=fixtures.RefreshReplayTests.registry;engine=fixtures.RefreshReplayTests.engine;event=fixtures.RefreshReplayTests.event
    def factory(self):
        e=self.engine();e.process_batch(99,[self.event()],[]);return e
    def journal(self):return SnapshotRefreshJournal(self.root/'log.jsonl',self.factory,identity_for(self.registry(),99),empty_factory=self.engine,snapshot_path=self.root/'state.gz')
    def test_state_roundtrip_and_next_response(self):
        a=self.factory();a.process_batch(101,[self.event(2)])
        b=restore_state(self.engine(),json.loads(canonical(capture(a))))
        self.assertEqual(capture(a),capture(b));self.assertEqual(a.process_batch(102,[self.event(3)]),b.process_batch(102,[self.event(3)]))
    def test_snapshot_tail_and_old_retry(self):
        j=self.journal();a=j.process_batch(101,[self.event(2)]);j.checkpoint();b=j.process_batch(102,[self.event(3)]);j.close()
        k=self.journal();self.assertEqual(k.snapshot_status,'loaded');self.assertEqual(k.replayed_batches,1);self.assertEqual(k.process_batch(101,[self.event(2)]),a);self.assertEqual(k.process_batch(102,[self.event(3)]),b);k.close()
    def test_no_warmup_on_valid_snapshot(self):
        j=self.journal();j.process_batch(101,[self.event(2)]);j.checkpoint();j.close()
        def forbidden():raise AssertionError('Full warmup should not run')
        k=SnapshotRefreshJournal(self.root/'log.jsonl',forbidden,identity_for(self.registry(),99),empty_factory=self.engine,snapshot_path=self.root/'state.gz');self.assertEqual(k.snapshot_status,'loaded');k.close()
    def test_corrupt_snapshot_falls_back(self):
        j=self.journal();a=j.process_batch(101,[self.event(2)]);j.checkpoint();j.close();(self.root/'state.gz').write_bytes(b'bad')
        k=self.journal();self.assertTrue(k.snapshot_status.startswith('rejected'));self.assertEqual(k.process_batch(101,[self.event(2)]),a);k.close()
    def test_wrong_prefix_falls_back(self):
        import hashlib
        j=self.journal();j.process_batch(101,[self.event(2)]);j.checkpoint();j.close();p=self.root/'state.gz';w=json.loads(gzip.decompress(p.read_bytes()));w['body']['prefix_count']=999;w['sha256']=hashlib.sha256(canonical(w['body']).encode()).hexdigest();p.write_bytes(gzip.compress(canonical(w).encode()))
        k=self.journal();self.assertTrue(k.snapshot_status.startswith('rejected'));k.close()
    def test_atomic_replace_failure_preserves_old_snapshot(self):
        j=self.journal();j.process_batch(101,[self.event(2)]);j.checkpoint();old=(self.root/'state.gz').read_bytes();j.process_batch(102,[self.event(3)])
        with patch('newsrec.state_snapshot.os.replace',side_effect=OSError('injected')):
            with self.assertRaises(OSError):j.checkpoint()
        self.assertEqual((self.root/'state.gz').read_bytes(),old);self.assertEqual(j.observed_through,102);j.close()
    def test_historical_fallback_snapshot(self):
        p=self.root/'next.txt';raw=p.read_bytes();p.unlink();j=self.journal();a=j.process_batch(self.switch,[self.event(2)]);j.checkpoint();j.close();p.write_bytes(raw)
        k=self.journal();self.assertEqual(k.snapshot_status,'loaded');self.assertEqual(k.process_batch(self.switch,[self.event(2)]),a);self.assertNotEqual(k.process_batch(self.switch+1,[self.event(3)])['model_cutoff'],100);k.close()
    def test_inconsistent_state_rejected(self):
        a=self.factory();d=json.loads(canonical(capture(a)));d['state']['trend']['counts']={}
        with self.assertRaises(RefreshJournalError):restore_state(self.engine(),d)
    def test_multidigit_numeric_keys_have_stable_checksum(self):
        j=self.journal();j.process_batch(101,[self.event(2,20,2),self.event(3,3,3)]);j.checkpoint();j.close()
        k=self.journal();self.assertEqual(k.snapshot_status,'loaded');self.assertEqual(k.engine.lengths[20],1);k.close()
    def test_checkpoint_api_is_explicit(self):
        from fastapi.testclient import TestClient
        from newsrec.refresh_api import create_refresh_app
        j=self.journal();client=TestClient(create_refresh_app(j));self.assertEqual(client.post('/replay/checkpoint').status_code,200);j.close()
        self.assertEqual(TestClient(create_refresh_app(self.factory())).post('/replay/checkpoint').status_code,409)
