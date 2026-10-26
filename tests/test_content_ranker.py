import sys
from pathlib import Path
import tempfile
import sqlite3
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.content_ranker import ContentVectors,run_pair,predict_matrix,validate_protocol


class ContentTests(unittest.TestCase):
    def setUp(self):
        self.store=ContentVectors([10,20,30],np.asarray([[1.,0.],[0.,1.],[-1.,0.]]),{10:0,20:0,30:100})

    def test_known_cosines_last_max_and_weighted_mean(self):
        result=self.store.features([10,20],[(10,10),(20,20)],30)
        np.testing.assert_allclose(result[:,0],[0,1])
        np.testing.assert_allclose(result[:,1],[1,1])
        np.testing.assert_allclose(result[:,2],[.7/1.7,1/1.7],rtol=1e-6)

    def test_only_last_three_and_causal_history(self):
        history=[(10,1),(20,2),(10,3),(20,4)]
        np.testing.assert_array_equal(self.store.features([10],history,4),self.store.features([10],history[-3:],4))
        with self.assertRaisesRegex(ValueError,'Future history'):self.store.features([10],[(10,5)],4)
        with self.assertRaisesRegex(ValueError,'candidate'):self.store.features([30],[(10,1)],4)

    def test_empty_and_zero_nonfinite_contracts(self):
        self.assertEqual(self.store.features([],[(10,1)],1).shape,(0,3))
        np.testing.assert_array_equal(self.store.features([10],[],1),np.zeros((1,3)))
        for array in [np.asarray([[0.,0.]]),np.asarray([[np.nan,1.]])]:
            with self.assertRaises(ValueError):ContentVectors([10],array,{10:0})

    def test_empty_candidate_batch_does_not_call_booster(self):
        class NeverModel:
            def predict(self,*args,**kwargs):raise AssertionError('Empty batch must stay in denominator')
        self.assertEqual(len(predict_matrix(NeverModel(),np.empty((0,18)),18,1)),0)

    def test_future_catalog_vector_does_not_change_past_features(self):
        other=ContentVectors([10,20,30],np.asarray([[1.,0.],[0.,1.],[100.,100.]]),{10:0,20:0,30:100})
        np.testing.assert_array_equal(self.store.features([10,20],[(10,1)],2),other.features([10,20],[(10,1)],2))

    def test_protected_boundary_rejected_before_source_read(self):
        with self.assertRaisesRegex(ValueError,'protected boundary'):
            run_pair(None,None,None,Path('.'),dict(train_end_ms=1,validation_end_ms=3,old_validation_start_ms=2))

    def test_confirmation_requires_explicit_flag_and_exact_window(self):
        p=dict(mode='validation_confirmation',train_end_ms=2,validation_end_ms=3,
               old_validation_start_ms=2,old_validation_end_ms=3)
        with self.assertRaisesRegex(ValueError,'authorization'):validate_protocol(p)
        validate_protocol(p,True)
        for key,value in [('train_end_ms',1),('validation_end_ms',4)]:
            with self.assertRaisesRegex(ValueError,'frozen validation'):validate_protocol(dict(p,**{key:value}),True)

    def test_pair_baseline_matches_original_and_future_rows_do_not_change_results(self):
        from test_contracts import create_fixture
        from newsrec.data import discover,load_database,file_hash
        from newsrec.time_ranker import run,CONFIG
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);results=[]
            config=dict(CONFIG,n_estimators=3,min_child_samples=1,train_ordinal_modulus=1,threads=1)
            protocol=dict(train_end_ms=10000000,validation_end_ms=20000000,old_validation_start_ms=20000000)
            for index,extra in enumerate([(),[(1,30,999)]]):
                data=root/f'data{index}';data.mkdir();create_fixture(data,extra_train=extra)
                conn=load_database(discover(data),root/f'db{index}.sqlite')
                conn.execute('UPDATE clicks SET timestamp=timestamp*100000')
                conn.execute('UPDATE articles SET created=created*100000')
                metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
                store=ContentVectors(list(metadata),np.eye(len(metadata),dtype=np.float32),{i:d['created'] for i,d in metadata.items()})
                paired=root/f'paired{index}';paired.mkdir()
                actual=run_pair(conn,store,metadata,paired,protocol,config)
                rows=[json.loads(line) for line in (paired/'predictions.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['target_timestamp']<=protocol['validation_end_ms'] for r in rows))
                if index==0:
                    old=root/'old';old.mkdir();run(conn,old,protocol,config)
                    self.assertEqual(file_hash(old/'train_features.f32'),actual['shared_base_feature_sha256'])
                    self.assertEqual((old/'ranker.txt').read_text(),(paired/'base15.txt').read_text())
                    reference={r['request_id']:r['article_ids'] for r in map(json.loads,(old/'predictions.jsonl').read_text().splitlines()) if r['model']=='lambdarank_time'}
                    self.assertEqual(reference,{r['request_id']:r['article_ids']['base15'] for r in rows})
                results.append(((paired/'predictions.jsonl').read_text(),file_hash(paired/'base15.txt'),file_hash(paired/'content18.txt')))
                conn.close()
            self.assertEqual(results[0],results[1])
