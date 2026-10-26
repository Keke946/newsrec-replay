import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from newsrec.journal import JournalEngine, JournalError
from newsrec.serving import ReplayEngine, ContractError
from newsrec.time_ranker import FEATURES


class Model:
    def feature_name(self): return FEATURES
    def predict(self, features, **kwargs): return features[:, 12]


META = {10: dict(category=1, created=0, words=100),
        20: dict(category=2, created=0, words=200)}


def factory(): return ReplayEngine(META, Model())


def events(article, n=1):
    return [dict(event_id=i, user_id=i+1, article_id=article) for i in range(n)]


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'actions.sqlite'
        self.engine = JournalEngine(self.path, factory, {'test': 1})

    def tearDown(self):
        self.engine.close()
        self.temp.cleanup()

    def reopen(self):
        self.engine.close()
        self.engine = JournalEngine(self.path, factory, {'test': 1})

    def test_queries_restore_cached_ranking_not_only_events(self):
        self.engine.ingest(10, events(10))
        self.engine.recommend(999, 20)
        self.engine.ingest(30, events(20, 3))
        expected = self.engine.recommend(999, 40)
        self.assertEqual(expected['article_ids'], [10])
        self.reopen()
        self.assertEqual(self.engine.replayed_actions, 4)
        self.assertEqual(self.engine.recommend(999, 40), expected)
        # Demonstrates why event-only recovery is a regression.
        wrong = factory()
        wrong.ingest(10, events(10)); wrong.ingest(30, events(20, 3))
        self.assertEqual(wrong.recommend(999, 40)['article_ids'], [20, 10])

    def test_failed_write_restores_cache_clocks_and_allows_retry(self):
        self.engine.ingest(10, events(10))
        with patch.object(self.engine, '_persist', side_effect=sqlite3.OperationalError('injected disk full')):
            with self.assertRaises(JournalError): self.engine.recommend(1, 40, 10)
        self.assertIsNone(self.engine.engine.query_clock)
        self.assertIsNone(self.engine.engine.state.trend.bucket)
        self.engine.ingest(20, events(20))
        self.assertEqual(self.engine.recommend(1, 30)['history_length'], 2)

    def test_commit_then_error_preserves_committed_event_once(self):
        persist = self.engine._persist
        def ambiguous(*args):
            persist(*args)
            raise sqlite3.OperationalError('commit ack lost')
        with patch.object(self.engine, '_persist', side_effect=ambiguous):
            with self.assertRaises(JournalError): self.engine.ingest(10, events(10))
        self.assertTrue(self.engine.ingest(10, events(10))['duplicate_batch'])
        self.assertEqual(self.engine.engine.lengths[1], 1)

    def test_transaction_failure_after_insert_leaves_no_committed_action(self):
        def fail(*args):
            with self.engine.connection:
                self.engine.connection.execute('INSERT INTO actions(kind,payload,result) VALUES (?,?,?)', args)
                raise sqlite3.OperationalError('injected before commit')
        with patch.object(self.engine, '_persist', side_effect=fail):
            with self.assertRaises(JournalError): self.engine.ingest(10, events(10))
        self.assertIsNone(self.engine.observed_through)
        self.assertEqual(self.engine.connection.execute('SELECT count(*) FROM actions').fetchone()[0], 0)
        self.assertEqual(self.engine.ingest(10, events(10))['accepted'], 1)

    def test_unrelated_database_is_not_modified(self):
        path=Path(self.temp.name)/'source.sqlite'
        with sqlite3.connect(path) as connection:
            connection.execute('CREATE TABLE articles (id INTEGER)')
        before=path.read_bytes()
        with self.assertRaises(JournalError): JournalEngine(path, factory, {'test': 1})
        self.assertEqual(before,path.read_bytes())

    def test_concurrent_retries_and_restart_observe_batch_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs=list(pool.map(lambda _:self.engine.ingest(10, events(10)),range(8)))
        self.assertEqual(sum(row['accepted'] for row in outputs), 1)
        self.reopen()
        self.assertEqual(self.engine.engine.lengths[1], 1)

    def test_model_error_rolls_back_query_mutation(self):
        self.engine.ingest(10, events(10))
        with patch.object(self.engine.engine.model, 'predict', side_effect=RuntimeError('injected model failure')):
            with self.assertRaises(RuntimeError): self.engine.recommend(1, 40, 10)
        self.assertIsNone(self.engine.engine.query_clock)
        self.engine.ingest(20, events(20))

    def test_identity_corruption_and_second_owner_fail_closed(self):
        with self.assertRaises(JournalError): JournalEngine(self.path, factory, {'test': 1})
        self.engine.ingest(10, events(10))
        self.engine.close()
        with self.assertRaises(JournalError): JournalEngine(self.path, factory, {'test': 2})
        with sqlite3.connect(self.path) as connection:
            connection.execute('UPDATE actions SET result=?', (json.dumps({'wrong': True}),))
        with self.assertRaises(JournalError): JournalEngine(self.path, factory, {'test': 1})

    def test_invalid_action_not_logged_and_recovery_failure_closes(self):
        with self.assertRaises(ContractError): self.engine.ingest(10, events(999))
        self.assertEqual(self.engine.connection.execute('SELECT count(*) FROM actions').fetchone()[0], 0)
        with patch.object(self.engine, '_restore', side_effect=RuntimeError('broken factory')):
            with self.assertRaises(JournalError): self.engine.ingest(10, events(999))
        with self.assertRaises(JournalError): self.engine.recommend(1, 20)
        from fastapi.testclient import TestClient
        from newsrec.api import create_app
        with TestClient(create_app(self.engine)) as client:
            self.assertEqual(client.get('/health').status_code,503)

    def test_process_exit_after_commit_before_response_and_restart(self):
        self.engine.close()
        code = '''
import os
from test_journal import factory, events
from newsrec.journal import JournalEngine
import sys
e = JournalEngine(sys.argv[1], factory, {'test': 1})
persist = e._persist
def crash(*args):
    persist(*args)
    os._exit(23)
e._persist = crash
e.ingest(10, events(10))
'''
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent)+os.pathsep+str(Path(__file__).resolve().parents[1]/'src'))
        process = subprocess.run([sys.executable, '-c', code, str(self.path)], env=env, capture_output=True)
        self.assertEqual(process.returncode, 23, process.stderr)
        self.reopen()
        self.assertTrue(self.engine.ingest(10, events(10))['duplicate_batch'])
        self.assertEqual(self.engine.engine.lengths[1], 1)

    def test_api_adapter_reports_durability_and_write_errors(self):
        from fastapi.testclient import TestClient
        from newsrec.api import create_app
        with TestClient(create_app(self.engine)) as client:
            self.assertIn('SQLite', client.get('/health').json()['state'])
            with patch.object(self.engine, '_persist', side_effect=sqlite3.OperationalError('disk full')):
                self.assertEqual(client.post('/events', json=dict(timestamp=10,events=events(10))).status_code, 503)
            self.assertEqual(client.post('/events', json=dict(timestamp=10,events=events(10))).json()['accepted'], 1)
