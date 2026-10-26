import unittest
import numpy as np
from newsrec.age_fusion import AgeCandidates,candidate_features
from newsrec.content_ranker import ContentVectors
from newsrec.time_ranker import CausalCandidates,CONFIG
class AgeFusionTests(unittest.TestCase):
    def setUp(self):
        self.metadata={i:dict(created=t,category=i%2,words=10) for i,t in enumerate([-1,0,21600000,21600001])}
        self.vectors=ContentVectors(np.arange(4),np.ones((4,250),dtype=np.float32),{i:d['created'] for i,d in self.metadata.items()})
    def test_original_feature_parity(self):
        state=CausalCandidates(self.metadata,CONFIG)
        state.observe(21600000,1)
        history=[(1,21600000)]
        trend,items,x=state.query(history,21600000,1)
        np.testing.assert_array_equal(x,candidate_features(state,items,trend,history,21600000,1))
    def test_retrieval_age_and_ties(self):
        state=AgeCandidates(self.metadata,CONFIG,self.vectors)
        self.assertEqual(state.retrieve([(1,21600000)],21600000),[1,2])
        with self.assertRaises(ValueError):state.retrieve([(1,21600001)],21600000)
    def test_budget(self):
        state=AgeCandidates(self.metadata,CONFIG,self.vectors);state.observe(21600000,1)
        _,items,x=state.query([(1,21600000)],21600000,1)
        self.assertEqual(items,[1]);self.assertEqual(len(x),1)
