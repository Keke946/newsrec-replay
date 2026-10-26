import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.mixed_negatives import mixed_mask,state_hash,train_mixed
from newsrec.two_tower import negative_mask,TwoTower


class MixedTests(unittest.TestCase):
    def setUp(self):
        self.targets=np.asarray([0,1,2,0]);self.times=np.asarray([1,2,6,3])
        self.created=np.asarray([0,0,5,100,0,0]);self.catalog=np.arange(6)

    def test_same_actual_budget_no_future_or_own_positive_negatives(self):
        pool,mask,counts=mixed_mask(self.targets,self.times,self.created,self.catalog,np.random.default_rng(2027))
        original=negative_mask(torch.tensor(self.targets),torch.tensor(self.created[self.targets]),torch.tensor(self.times)).numpy()
        np.testing.assert_array_equal(counts,original.sum(axis=1)-1)
        np.testing.assert_array_equal(mask.sum(axis=1)-1,counts)
        self.assertTrue(mask[:,:4].diagonal().all())
        self.assertFalse((mask[:,:4]&~original).any())
        extra=mask[:,4:]
        self.assertFalse((extra&(self.created[pool][None,:]>self.times[:,None])).any())
        self.assertFalse((extra&(pool[None,:]==self.targets[:,None])).any())
        np.testing.assert_array_equal(extra.sum(axis=1),counts//2)

    def test_seed_repeatability_and_initial_state_identity(self):
        a=mixed_mask(self.targets,self.times,self.created,self.catalog,np.random.default_rng(2027))
        b=mixed_mask(self.targets,self.times,self.created,self.catalog,np.random.default_rng(2027))
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        torch.manual_seed(2026);one=TwoTower()
        torch.manual_seed(2026);two=TwoTower()
        self.assertEqual(state_hash(one),state_hash(two))

    def test_finite_gradients_and_duplicate_positive_case(self):
        pool,mask,_=mixed_mask(self.targets,self.times,self.created,self.catalog,np.random.default_rng(2027))
        x=torch.randn(4,4+len(pool),requires_grad=True)
        loss=torch.nn.functional.cross_entropy(x.masked_fill(~torch.from_numpy(mask),-torch.inf),torch.arange(4))
        loss.backward();self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[0,3]),0.)

    def test_no_usable_negative_preserves_diagonal_and_masks_pool(self):
        pool,mask,counts=mixed_mask(np.asarray([0,0]),np.asarray([1,1]),self.created,self.catalog,np.random.default_rng(2027))
        np.testing.assert_array_equal(counts,[0,0]);self.assertEqual(int(mask.sum()),2)
        self.assertFalse(mask[:,2:].any())

    def test_all_zero_negative_batch_skips_update_like_original(self):
        class Store:
            created=np.asarray([0])
            def vectors(self,*args):raise AssertionError('Zero-negative batch must be skipped')
        model=TwoTower();before=state_hash(model)
        rows=np.asarray([[1,1,1,0,2,1],[2,2,1,0,2,1]],dtype=np.int64)
        result=train_mixed(model,Store(),np.zeros((2,250),dtype=np.float32),rows,1)
        self.assertEqual(result['updates'],0);self.assertIsNone(result['mean_loss'])
        self.assertEqual(state_hash(model),before)
