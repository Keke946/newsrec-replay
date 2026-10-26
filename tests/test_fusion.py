import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.fusion import fixed_count_fusion


class FusionTests(unittest.TestCase):
    def test_exact_count_history_protection_and_ten_novel_slots(self):
        trend=list(range(50));history=[(60,0),(61,0),(62,0)]
        base=trend+[62,61,60]
        result=fixed_count_fusion(base,trend,history,list(range(100,110)))
        self.assertEqual(len(result),53)
        self.assertEqual(set(base)-set(result),set(range(40,50)))
        self.assertTrue({60,61,62}.issubset(result))

    def test_overlap_refills_without_spending_extra_candidates(self):
        trend=list(range(50));history=[(49,0),(48,0),(47,0)]
        result=fixed_count_fusion(trend,trend,history,list(range(10)))
        self.assertEqual(set(result),set(trend))
        self.assertEqual(len(result),50)

    def test_empty_or_small_base_retains_exact_budget(self):
        self.assertEqual(fixed_count_fusion([],[],[],[100]),[])
        self.assertEqual(fixed_count_fusion([1],[1],[(1,0)],[100]),[1])

    def test_budget_contract_rejects_unregistered_inputs(self):
        with self.assertRaises(ValueError):fixed_count_fusion([1,1],[1],[],[])
        with self.assertRaises(ValueError):fixed_count_fusion([1],[1],[],list(range(11)))
        with self.assertRaises(ValueError):fixed_count_fusion([1],[1],[],[],slots=5)
