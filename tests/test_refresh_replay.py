import json,tempfile,unittest
from pathlib import Path
import numpy as np
import lightgbm as lgb
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine,metadata_hash,encoder_hashes
from newsrec.serving import ContractError
from newsrec.time_ranker import FEATURES,CONFIG

class RefreshReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.start=100;self.switch=self.start+21600000+60000
        self.meta={i:dict(category=i,created=0,words=20) for i in [1,2,3]}
        x=np.arange(120,dtype=np.float32).reshape(8,15);y=np.arange(8)%2
        model=lgb.LGBMRanker(n_estimators=1,min_child_samples=1,n_jobs=1,verbosity=-1)
        model.fit(x,y,group=[4,4],feature_name=FEATURES)
        model.booster_.save_model(str(self.root/'initial.txt'));model.booster_.save_model(str(self.root/'next.txt'))
        from newsrec.data import file_hash
        self.info=dict(schema='precomputed-refresh-replay-v1',start_ms=self.start,database_sha256='fixture',metadata_sha256=metadata_hash(self.meta),encoder_sha256=encoder_hashes(),feature_names=FEATURES,config=CONFIG,
          models=[dict(cutoff=self.start,activate_ms=self.start,max_label_time=99,file='initial.txt',sha256=file_hash(self.root/'initial.txt')),dict(cutoff=self.start+21600000-1,activate_ms=self.switch,max_label_time=self.start+21600000-2,file='next.txt',sha256=file_hash(self.root/'next.txt'))])
        self.write()
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        (self.root/'registry.json').write_text(json.dumps(self.info))
    def registry(self):return ModelRegistry(self.root/'registry.json','fixture')
    def engine(self):return RefreshReplayEngine(self.meta,self.registry())
    def event(self,i=1,u=1,a=1):return dict(event_id=i,user_id=u,article_id=a)
    def test_activation_boundary(self):
        r=self.registry();self.assertEqual(r.select(self.switch-1)[1]['cutoff'],100)
        self.assertEqual(r.select(self.switch)[1]['cutoff'],self.start+21600000-1)
        with self.assertRaises(ContractError):r.select(101)
    def test_missing_then_late_refresh(self):
        raw=(self.root/'next.txt').read_bytes();(self.root/'next.txt').unlink();r=self.registry()
        self.assertEqual(r.select(self.switch)[1]['cutoff'],100)
        (self.root/'next.txt').write_bytes(raw)
        self.assertEqual(r.select(self.switch+1)[1]['cutoff'],self.start+21600000-1)
    def test_missing_initial(self):
        (self.root/'initial.txt').unlink()
        with self.assertRaises(ContractError):self.registry()
    def test_checksum_failure_closes_engine(self):
        e=self.engine();e.process_batch(99,[self.event()],[])
        (self.root/'next.txt').write_text('bad')
        with self.assertRaises(ContractError):e.process_batch(self.switch,[self.event(2)],None)
        self.assertTrue(e.closed);self.assertEqual(e.observed_through,99)
    def test_identity_future_and_path_rejected(self):
        for field,value in [('database_sha256','other'),('feature_names',['bad']),('encoder_sha256',{})]:
            old=self.info[field];self.info[field]=value;self.write()
            with self.assertRaises(ContractError):self.registry()
            self.info[field]=old
        self.info['models'][1]['max_label_time']=self.switch;self.write()
        with self.assertRaises(ContractError):self.registry()
        self.info['models'][1]['max_label_time']=99;self.info['models'][1]['file']='../escape.txt';self.write()
        with self.assertRaises(ContractError):self.registry()
    def test_metadata_rejected(self):
        wrong={**self.meta,4:dict(category=4,created=0,words=1)}
        with self.assertRaises(ContractError):RefreshReplayEngine(wrong,self.registry())
    def test_tied_own_click_and_batch_contract(self):
        e=self.engine();e.process_batch(99,[self.event()],[])
        out=e.process_batch(101,[self.event(2,1,2),self.event(3,1,3)])
        self.assertEqual([r['history_length'] for r in out['predictions']],[2,2]);self.assertEqual(e.lengths[1],3)
        with self.assertRaises(ContractError):e.process_batch(101,[self.event(4)])
        with self.assertRaises(ContractError):e.process_batch(102,[self.event(4)],score_event_ids=[99])
        self.assertEqual(e.observed_through,101)
    def test_unscored_queries_preserve_cache(self):
        a,b=self.engine(),self.engine()
        a.process_batch(299999,[self.event()],[]);b.process_batch(299999,[self.event()],[])
        a.process_batch(300000,[self.event(2,1,2)],[]);b.process_batch(300000,[self.event(2,1,2)])
        self.assertEqual(a.state.trend.cached,[1]);self.assertEqual(a.state.trend.bucket,1)
        self.assertEqual(a.process_batch(300001,[self.event(3,1,3)])['predictions'],b.process_batch(300001,[self.event(3,1,3)])['predictions'])
    def test_opt_in_api(self):
        from fastapi.testclient import TestClient
        from newsrec.refresh_api import create_refresh_app
        e=self.engine();client=TestClient(create_refresh_app(e))
        self.assertEqual(client.post('/replay/batch',json=dict(timestamp=99,events=[self.event()],score_event_ids=[])).status_code,200)
        r=client.post('/replay/batch',json=dict(timestamp=101,events=[self.event(2)]));self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['model_cutoff'],100)
        self.assertEqual(client.post('/replay/batch',json=dict(timestamp=101,events=[self.event(2)])).status_code,409)
        self.assertEqual(client.post('/replay/batch',json=dict(timestamp=True,events=[self.event(2)])).status_code,422)
        self.assertEqual(client.get('/health').json()['mode'],'precomputed_replay')
    def test_api_closed_is_unavailable(self):
        from fastapi.testclient import TestClient
        from newsrec.refresh_api import create_refresh_app
        e=self.engine();client=TestClient(create_refresh_app(e));e.closed=True
        self.assertEqual(client.get('/health').status_code,503)
        self.assertEqual(client.post('/replay/batch',json=dict(timestamp=101,events=[self.event()])).status_code,503)
    def test_all_click_warmup_boundary(self):
        import sqlite3
        from newsrec.refresh_replay import prime_refresh
        e=self.engine();conn=sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE clicks(timestamp,ordinal,user,article,role)')
        conn.executemany('INSERT INTO clicks VALUES(?,?,?,?,?)',[(99,1,1,1,'train'),(101,2,1,2,'train')])
        self.assertEqual(prime_refresh(e,conn,100),1);self.assertEqual(e.observed_through,99)
        self.assertEqual(e.process_batch(101,[self.event(2,1,2)])['predictions'][0]['history_length'],2)
        with self.assertRaises(ContractError):prime_refresh(self.engine(),conn,101)
        conn.close()
