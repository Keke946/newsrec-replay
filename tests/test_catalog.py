import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.catalog import FreshCatalog
from newsrec.time_ranker import CausalCandidates,CONFIG


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.metadata={i:dict(category=i%2,created=i*10,words=100) for i in range(1,21)}

    def test_only_published_recent_unobserved_and_no_current_history(self):
        index=FreshCatalog(self.metadata,max_age_ms=80)
        result=index.recommend([(12,125)],150,{11,13},k=5)
        self.assertEqual(len(result),5)
        self.assertTrue(all(70<=self.metadata[item]['created']<=150 for item in result))
        self.assertFalse({11,12,13}&set(result))
        self.assertEqual(len(set(result)),len(result))

    def test_future_catalog_append_does_not_change_past_candidates(self):
        first=FreshCatalog(self.metadata).recommend([(2,100)],100,set(),10)
        extra=dict(self.metadata);extra[999]=dict(category=0,created=101,words=100)
        second=FreshCatalog(extra).recommend([(2,100)],100,set(),10)
        self.assertEqual(first,second)

    def test_unseen_global_fill_and_candidate_budget(self):
        config=dict(CONFIG,fresh_slots=10,fresh_max_age_ms=100)
        state=CausalCandidates(self.metadata,config)
        state.observe(100,1)
        _,items,_=state.query([(2,150)],150,1)
        self.assertEqual(len(items),12)
        self.assertLessEqual(len(items),63)
        with self.assertRaisesRegex(ValueError,'backwards'):
            state.query([(2,140)],140,1)
        with self.assertRaisesRegex(ValueError,'Late'):
            state.observe(140,3)
