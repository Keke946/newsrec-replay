import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.serving import ReplayEngine,ContractError
from newsrec.time_ranker import FEATURES


class FakeModel:
    def feature_name(self):return FEATURES
    def predict(self,features,**kwargs):return features[:,0]


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.metadata={10:dict(category=1,created=0,words=100),20:dict(category=2,created=0,words=200),
                       30:dict(category=1,created=100,words=300)}
        self.engine=ReplayEngine(self.metadata,FakeModel())
        self.batch=[dict(event_id=1,user_id=1,article_id=10),dict(event_id=2,user_id=2,article_id=20)]

    def test_batch_replay_idempotency_and_closed_timestamp_requests(self):
        first=self.engine.ingest(10,self.batch)
        second=self.engine.ingest(10,list(reversed(self.batch)))
        self.assertEqual(first['accepted'],2)
        self.assertEqual(second['accepted'],0)
        self.assertEqual(self.engine.lengths[1],1)
        with self.assertRaises(ContractError):self.engine.recommend(1,10,10)
        changed=[dict(event_id=1,user_id=1,article_id=20)]
        with self.assertRaises(ContractError):self.engine.ingest(10,changed)

    def test_atomic_schema_rejection_does_not_partially_observe(self):
        bad=self.batch+[dict(event_id=3,user_id=1,article_id=999)]
        with self.assertRaises(ContractError):self.engine.ingest(10,bad)
        self.assertIsNone(self.engine.observed_through)
        self.assertEqual(len(self.engine.histories),0)
        self.engine.ingest(10,self.batch)
        self.assertEqual(self.engine.lengths[1],1)

    def test_query_retries_late_events_future_items_and_empty_user(self):
        self.engine.ingest(10,self.batch)
        one=self.engine.recommend(1,20,10,k=5)
        two=self.engine.recommend(1,20,10,k=5)
        self.assertEqual(one,two)
        self.assertNotIn(30,one['article_ids'])
        with self.assertRaises(ContractError):self.engine.recommend(1,19,10)
        with self.assertRaises(ContractError):self.engine.ingest(19,self.batch)
        cold=self.engine.recommend(999,20)
        self.assertEqual(cold['mode'],'cold_user_trend')
        self.assertEqual(cold['history_length'],0)
        empty=ReplayEngine(self.metadata,FakeModel()).recommend(1,1)
        self.assertEqual(empty['article_ids'],[])

    def test_feature_schema_mismatch_is_rejected(self):
        class BadModel(FakeModel):
            def feature_name(self):return ['wrong']
        with self.assertRaises(ContractError):ReplayEngine(self.metadata,BadModel())

    def test_concurrent_batch_retries_are_observed_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs=list(pool.map(lambda _:self.engine.ingest(10,self.batch),range(8)))
        self.assertEqual(sum(row['accepted'] for row in outputs),2)
        self.assertEqual(self.engine.lengths[1],1)
        self.assertEqual(sum(self.engine.state.trend.counts.values()),2)

    def test_fastapi_validation_idempotency_and_openapi(self):
        try:
            from fastapi.testclient import TestClient
            from newsrec.api import create_app
        except ImportError:self.skipTest('optional FastAPI/httpx dependencies required')
        with TestClient(create_app(self.engine)) as client:
            self.assertEqual(client.get('/health').status_code,200)
            self.assertEqual(client.get('/openapi.json').status_code,200)
            self.assertEqual(client.post('/recommend',json=dict(user_id=True,timestamp=20)).status_code,422)
            self.assertEqual(client.post('/recommend',json=dict(user_id=1,timestamp=20,extra=1)).status_code,422)
            payload=dict(timestamp=10,events=self.batch)
            self.assertEqual(client.post('/events',json=payload).json()['accepted'],2)
            self.assertEqual(client.post('/events',json=payload).json()['accepted'],0)
            response=client.post('/recommend',json=dict(user_id=1,timestamp=20,current_article=10,k=5))
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['mode'],'ranker')
            self.assertEqual(client.post('/recommend',json=dict(user_id=1,timestamp=19)).status_code,409)
