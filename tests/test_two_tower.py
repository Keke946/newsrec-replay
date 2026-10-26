import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

try:
    import numpy as np
    import torch
    from newsrec.two_tower import VectorStore, TwoTower, negative_mask, build_examples, evaluate
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'optional CPU PyTorch environment required')
class TwoTowerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = sqlite3.connect(':memory:')
        self.db.execute('CREATE TABLE articles(article INTEGER,created INTEGER)')
        self.db.executemany('INSERT INTO articles VALUES(?,?)', [(10,0),(11,0),(12,25),(13,0)])
        self.db.execute('CREATE TABLE clicks(role TEXT,user INTEGER,article INTEGER,timestamp INTEGER,ordinal INTEGER)')
        self.db.executemany('INSERT INTO clicks VALUES(?,?,?,?,?)', [
            ('train',1,10,10,0),('train',1,11,20,1),('train',1,13,20,2),
            ('train',1,12,30,3),('train',1,10,40,4),('testA',1,11,15,5)])
        (self.root/'VECTOR_MANIFEST.json').write_text(json.dumps({
            'rows':4,'dimensions':250,'vectors_dtype':'<f4','article_ids_dtype':'<i8'}))
        np.asarray([10,11,12,13],dtype='<i8').tofile(self.root/'article_ids.i64')
        vectors=np.zeros((4,250),dtype='<f4')
        vectors[np.arange(4),np.arange(4)]=1
        vectors.tofile(self.root/'article_vectors.f32')
        self.store=VectorStore(self.root,self.db)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def examples(self,name,**kwargs):
        directory=self.root/name
        directory.mkdir()
        return build_examples(self.db,self.store,directory,**kwargs)

    def test_mask_duplicate_targets_and_future_items(self):
        mask=negative_mask(torch.tensor([1,1,2]),torch.tensor([0,0,15]),torch.tensor([10,20,20]))
        self.assertEqual(mask.tolist(),[[True,False,False],[False,True,True],[True,True,True]])
        with self.assertRaises(ValueError):
            negative_mask(torch.tensor([1]),torch.tensor([20]),torch.tensor([10]))

    def test_examples_cutoff_ties_role_and_unavailable_denominator(self):
        queries,records,info=self.examples('valid',end=30,request_after=10)
        self.assertEqual(records.tolist(),[[1,2,20,2,30,2]])
        # At t20, only t10 and the request's own article13 are known.
        self.assertEqual(float(queries[0,1]),0)
        self.assertGreater(float(queries[0,0]),0)
        self.assertGreater(float(queries[0,3]),0)
        _,trained,info=self.examples('train',end=30,training=True)
        self.assertEqual(len(trained),1)
        self.assertEqual(info['unavailable_training_positives_excluded'],1)

    def test_source_truncation_rejected(self):
        (self.root/'article_vectors.f32').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'Truncated'):
            VectorStore(self.root,self.db)

    def test_nonlearned_control_reuses_legal_catalog_and_recent_history(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
        from evaluate_content_control import ContentCosine
        queries,records,_=self.examples('cosine',end=30,request_after=10)
        result=evaluate(ContentCosine(),self.store,queries,records,self.root,
                        dimension=250,model_name='content_cosine')
        row=json.loads((self.root/'predictions.jsonl').read_text())
        self.assertEqual(row['model'],'content_cosine')
        self.assertEqual(row['article_ids'][0],13)
        self.assertNotIn(12,row['article_ids'])
        self.assertEqual(result['overall']['requests'],1)

    def test_finite_gradients_and_available_candidates(self):
        model=TwoTower()
        x=torch.randn(4,250)
        logits=model.users(x)@model.items(x).T/.1
        loss=torch.nn.functional.cross_entropy(logits,torch.arange(4))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        queries,records,_=self.examples('evaluate',end=30,request_after=10)
        result=evaluate(model,self.store,queries,records,self.root)
        self.assertEqual(result['overall']['requests'],1)
        self.assertEqual(result['overall']['hit_rate@5'],0)
        prediction=json.loads((self.root/'predictions.jsonl').read_text())
        self.assertNotIn(12,prediction['article_ids'])
