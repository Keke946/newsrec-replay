import sys
import tempfile
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    import numpy as np
    from newsrec.sequence import SequenceTower,SequenceProvider,build_sequences
except ImportError:torch=None


@unittest.skipIf(torch is None,'optional CPU PyTorch required')
class SequenceTests(unittest.TestCase):
    def test_shared_item_initialization_and_finite_gradients(self):
        hashes=[]
        for kind in ['pool','gru']:
            torch.manual_seed(2026);model=SequenceTower(kind)
            hashes.append(model.item_initialization_hash())
            values=torch.zeros(3,20,250);values[:,:3]=torch.randn(3,3,250)
            loss=(model.users(values)@model.items(values[:,0]).T).sum()
            loss.backward()
            self.assertTrue(all(torch.isfinite(value.grad).all() for value in model.parameters()))
        self.assertEqual(hashes[0],hashes[1])

    def test_padding_is_validated_and_gru_uses_order(self):
        torch.manual_seed(2026);model=SequenceTower('gru')
        values=torch.zeros(1,20,250);values[:,:3]=torch.randn(1,3,250)
        first=model.users(values)
        reversed_values=values.clone();reversed_values[:,:3]=values[:,:3].flip(1)
        self.assertFalse(torch.allclose(first,model.users(reversed_values)))
        values[:,1]=0
        with self.assertRaises(ValueError):model.users(values)

    def test_builder_matches_pooled_cohort_and_hides_tied_events(self):
        from test_two_tower import TwoTowerTests
        from newsrec.two_tower import build_examples
        fixture=TwoTowerTests();fixture.setUp()
        try:
            first=fixture.root/'sequence';first.mkdir()
            second=fixture.root/'pool';second.mkdir()
            indices,records,_=build_sequences(fixture.db,fixture.store,first,30,request_after=10)
            _,reference,_=build_examples(fixture.db,fixture.store,second,30,request_after=10)
            np.testing.assert_equal(records,reference)
            # At t20 request article13 is known; tied article11 remains hidden.
            self.assertEqual(indices[0,:2].tolist(),[0,3])
            self.assertTrue((indices[0,2:]==-1).all())
            values=SequenceProvider(indices,fixture.store)[:]
            self.assertEqual(values.shape,(1,20,250))
            self.assertEqual(float(values[0,2:].sum()),0)
        finally:fixture.tearDown()

    def test_future_event_injection_does_not_change_past_sequence_rows(self):
        from test_two_tower import TwoTowerTests
        fixture=TwoTowerTests();fixture.setUp()
        try:
            first=fixture.root/'before';second=fixture.root/'after';first.mkdir();second.mkdir()
            indices,records,_=build_sequences(fixture.db,fixture.store,first,30,request_after=10)
            before_indices=np.array(indices,copy=True);before_records=np.array(records,copy=True)
            fixture.db.execute('INSERT INTO clicks VALUES(?,?,?,?,?)',('train',1,11,999,99))
            indices,records,_=build_sequences(fixture.db,fixture.store,second,30,request_after=10)
            np.testing.assert_equal(indices,before_indices);np.testing.assert_equal(records,before_records)
        finally:fixture.tearDown()
