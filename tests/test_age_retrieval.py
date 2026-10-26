import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('age',Path(__file__).resolve().parents[1]/'tools/evaluate_age_retrieval.py')
age=importlib.util.module_from_spec(spec);spec.loader.exec_module(age)
class AgeTests(unittest.TestCase):
    def test_inclusive_boundaries(self):
        np.testing.assert_array_equal(age.eligible(np.array([-1,0,21600000,21600001]),21600000),[False,True,True,False])
    def test_stable_tie_and_empty(self):
        self.assertEqual(age.top(np.array([8,2,3]),np.array([1.,1.,2.])).tolist(),[2,1,0])
        self.assertEqual(age.top(np.array([],dtype=int),np.array([])).tolist(),[])
if __name__=='__main__':unittest.main()
