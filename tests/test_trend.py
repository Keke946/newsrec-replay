import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from newsrec.baseline import RollingPopularity


class TrendTests(unittest.TestCase):
    def metadata(self):
        return {1: {'created': 0}, 2: {'created': 0}, 3: {'created': 200}}

    def test_empty_state_does_not_fabricate_recommendations(self):
        self.assertEqual(RollingPopularity().recommend(self.metadata(), 10, 5), [])

    def test_expired_items_are_not_served_from_cache(self):
        trend = RollingPopularity(window_ms=10, refresh_ms=100)
        trend.observe(1, 1)
        self.assertEqual(trend.recommend(self.metadata(), 2, 5), [1])
        self.assertEqual(trend.recommend(self.metadata(), 11, 5), [])

    def test_refresh_has_no_future_knowledge_and_respects_cadence(self):
        trend = RollingPopularity(window_ms=1000, refresh_ms=10)
        trend.observe(1, 1)
        self.assertEqual(trend.recommend(self.metadata(), 2, 5), [1])
        trend.observe(3, 2)
        trend.observe(4, 2)
        self.assertEqual(trend.recommend(self.metadata(), 5, 5), [1])
        self.assertEqual(trend.recommend(self.metadata(), 10, 5), [2, 1])

    def test_future_created_article_is_not_recommended(self):
        trend = RollingPopularity()
        trend.observe(1, 3)
        self.assertEqual(trend.recommend(self.metadata(), 10, 5), [])

    def test_unordered_observation_and_future_state_are_rejected(self):
        trend = RollingPopularity()
        trend.observe(10, 1)
        with self.assertRaises(ValueError):
            trend.observe(9, 2)
        with self.assertRaises(ValueError):
            trend.recommend(self.metadata(), 9, 5)


if __name__ == '__main__':
    unittest.main()
