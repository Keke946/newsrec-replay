import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).resolve().parents[1]/'tools/audit_temporal_targets.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class DiagnosticTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual([audit.gap_bin(x) for x in [1,60000,60001,1800000,1800001,21600000,21600001]],['le1m','le1m','1m_to30m','1m_to30m','30m_to6h','30m_to6h','over6h'])
        with self.assertRaises(ValueError):audit.gap_bin(0)
    def test_decomposition(self):
        a={'a':dict(requests=10,original_hits=2,fused18_hits=3),'b':dict(requests=10,original_hits=2,fused18_hits=2)}
        b={'a':dict(requests=30,original_hits=10,fused18_hits=13),'b':dict(requests=10,original_hits=2,fused18_hits=1)}
        r=audit.symmetric_decomposition(a,b)
        self.assertAlmostEqual(r['composition_term']+r['within_stratum_term'],(3-1)/40-1/20)
    def test_missing_stratum_is_flagged(self):
        r=audit.symmetric_decomposition({'a':dict(requests=1,original_hits=0,fused18_hits=1)},{'b':dict(requests=1,original_hits=0,fused18_hits=0)})
        self.assertFalse(r['exact_if_no_unmatched'])
