import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('refresh',Path(__file__).resolve().parents[1]/'tools/run_refresh_backtest.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
class RefreshTests(unittest.TestCase):
 def test_activation_delay(self):
  s=100;h=r.SIX;d=r.DELAY
  self.assertEqual(r.active_cutoff(s+h+d-1,s,s+3*h),s)
  self.assertEqual(r.active_cutoff(s+h+d,s,s+3*h),s+h-1)
  self.assertEqual(r.active_cutoff(s+2*h+d,s,s+3*h),s+2*h-1)
 def test_label_arrival_not_request_time(self):
  g=[dict(request_time=1,target_time=10),dict(request_time=2,target_time=100)]
  self.assertEqual(r.eligible_groups(g,10),g[:1]);self.assertEqual(r.eligible_groups(g,9),[])
 def test_protocol_modes_are_exact(self):
  dev=r.load_protocol(False);confirm=r.load_protocol(True)
  self.assertEqual(max(f['target_end_ms'] for f in dev['folds']),1507757054762)
  self.assertEqual([(f['fit_end_ms'],f['target_end_ms']) for f in confirm['folds']],[(1507757054762,1507980021008)])
 def test_final_boundary_is_explicit_and_exclusive(self):
  final=r.load_protocol(final=True)
  self.assertEqual(final['folds'][0]['fit_end_ms'],1507980021008)
  self.assertTrue(final['no_final_retuning'])
  with self.assertRaises(ValueError):r.load_protocol(True,True)
 def test_no_new_labels_signature_is_exact(self):
  a=[dict(request_id=1,request_time=1,target_time=2),dict(request_id=2,request_time=3,target_time=10)]
  self.assertEqual(r.training_signature(r.eligible_groups(a,2)),r.training_signature(r.eligible_groups(a,9)))
  self.assertNotEqual(r.training_signature(r.eligible_groups(a,9)),r.training_signature(r.eligible_groups(a,10)))
