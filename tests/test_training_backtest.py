import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('plan',Path(__file__).resolve().parents[1]/'tools/plan_training_backtest.py');plan=importlib.util.module_from_spec(spec);spec.loader.exec_module(plan)
class BacktestTests(unittest.TestCase):
    def test_train_only_disjoint_intervals(self):
        folds=plan.expanding_folds(list(range(1,101)),100)
        self.assertEqual([(f['fit_end_ms'],f['target_end_ms']) for f in folds],[(40,60),(60,80),(80,100)])
    def test_future_unordered_and_tie_collapse(self):
        for times,limit in [(list(range(101)),99),([2,1]*10,100),([1]*100,2)]:
            with self.assertRaises(ValueError):plan.expanding_folds(times,limit)
