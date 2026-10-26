import importlib.util
from pathlib import Path
import unittest

try:
    import numpy
except ImportError:
    numpy = None


@unittest.skipIf(numpy is None, 'Optional analysis tests require NumPy; core remains stdlib-only')
class ClusterTests(unittest.TestCase):
    def load(self):
        path = Path(__file__).resolve().parents[1] / 'tools/compare_validation.py'
        spec = importlib.util.spec_from_file_location('validation_stats', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.cluster_bootstrap

    def test_weighting_and_pairing_have_known_values(self):
        result = self.load()([[3, 0, 0], [1, 0, 1]], replicas=40)
        self.assertEqual(result['request_weighted_delta'], .25)
        self.assertEqual(result['cluster_macro_delta'], .5)

    def test_seed_is_reproducible_and_identical_models_have_zero_delta(self):
        function = self.load()
        a = function([[3, 1, 1], [5, 2, 2]], replicas=40)
        self.assertEqual(a, function([[3, 1, 1], [5, 2, 2]], replicas=40))
        self.assertEqual(a['request_weighted_95pct_percentile_interval'], [0., 0.])

    def test_one_cluster_is_not_falsely_reported_as_inferential_interval(self):
        result = self.load()([[1, 0, 1]], replicas=40)
        self.assertNotIn('request_weighted_95pct_percentile_interval', result)


if __name__ == '__main__':
    unittest.main()
