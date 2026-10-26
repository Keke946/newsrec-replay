"""Behavioral tests for time boundaries, data integrity and metric semantics."""
import csv
import json
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from newsrec.baseline import WindowedItemCF, replay
from newsrec.cli import main
from newsrec.data import ARTICLE_COLUMNS, CLICK_COLUMNS, audit, discover, load_database
from newsrec.evaluation import RankingMetrics, minmax


def create_fixture(path, extra_train=(), extra_test=()):
    articles = [(10, 1, 0, 100), (20, 1, 0, 100), (30, 2, 140, 100),
                (40, 2, 0, 100), (50, 3, 0, 100)]
    # At t=100 user1 has seen 10->20; target30 is not created until140.
    train = [(1, 10, 10), (1, 20, 100), (1, 30, 150), (1, 10, 200),
             (2, 20, 20), (2, 10, 80), (2, 40, 120), (2, 20, 180),
             (3, 10, 25), (3, 20, 60), (3, 10, 130), (3, 20, 190)]
    for name, header, rows in (
            ('articles.csv', ARTICLE_COLUMNS, articles),
            ('train_click_log.csv', CLICK_COLUMNS, [(*row, 1, 1, 1, 1, 1, 1) for row in train + list(extra_train)]),
            ('testB_click_log.csv', CLICK_COLUMNS, [(*row, 1, 1, 1, 1, 1, 1) for row in extra_test])):
        with (path / name).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)


class MetricTests(unittest.TestCase):
    def test_user_balanced_score_is_not_active_user_weighted(self):
        metric = RankingMetrics((5,))
        for _ in range(3):
            metric.add(10, [], user_id=1)
        metric.add(10, [10], user_id=2)
        result = metric.result()
        self.assertEqual(result['hit_rate@5'], .25)
        self.assertEqual(result['user_balanced']['hit_rate@5'], .5)
        self.assertEqual(result['user_balanced']['users'], 2)

    def test_missing_requests_and_empty_candidates_remain_in_denominator(self):
        metric = RankingMetrics((5,))
        metric.add(10, [20, 10])
        metric.add(30, [])
        result = metric.result()
        self.assertEqual(result['hit_rate@5'], .5)
        self.assertEqual(result['mrr@5'], .25)
        self.assertAlmostEqual(result['ndcg@5'], .5 / math.log2(3))

    def test_duplicate_predictions_rejected(self):
        with self.assertRaises(ValueError):
            RankingMetrics().add(1, [1, 1])

    def test_empty_metric_has_no_fabricated_score(self):
        self.assertIsNone(RankingMetrics().result()['hit_rate@5'])

    def test_normalization_equal_single_empty_and_order(self):
        self.assertEqual(minmax({1: 2., 2: 2.}), {1: 1., 2: 1.})
        self.assertEqual(minmax({1: 8.}), {1: 1.})
        self.assertEqual(minmax({}), {})
        self.assertEqual(minmax({3: 4., 1: 1., 2: 2.}), {3: 1., 1: 0., 2: 1 / 3})
        with self.assertRaises(ValueError):
            minmax({1: float('nan')})


class ReplayTests(unittest.TestCase):
    def run_fixture(self, root, **kwargs):
        root.mkdir()
        create_fixture(root, **kwargs)
        db = load_database(discover(root), root.parent / (root.name + '.sqlite'))
        self.addCleanup(db.close)
        return replay(db, 90, 200, k=5), db

    def test_future_catalog_items_never_recommended_and_unavailable_target_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            (metrics, rows), _ = self.run_fixture(Path(temporary) / 'source')
            row = next(row for row in rows if row['user_id'] == 1 and row['request_timestamp'] == 100)
            self.assertEqual(row['target_id'], 30)
            self.assertFalse(row['target_available'])
            self.assertNotIn(30, row['article_ids'])
            self.assertGreater(metrics['models']['itemcf']['overall']['requests'], metrics['available_targets'])

    def test_events_after_eval_end_do_not_change_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            (before, rows_before), _ = self.run_fixture(Path(temporary) / 'before')
            (after, rows_after), _ = self.run_fixture(Path(temporary) / 'after',
                                                      extra_train=[(4, 50, 300), (4, 50, 400)])
            self.assertEqual(rows_before, rows_after)
            self.assertEqual(before, after)

    def test_other_role_events_do_not_leak_into_training_or_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            (before, rows_before), _ = self.run_fixture(Path(temporary) / 'before')
            (after, rows_after), _ = self.run_fixture(Path(temporary) / 'after',
                                                      extra_test=[(1, 50, 5), (1, 50, 90)])
            self.assertEqual(rows_before, rows_after)
            self.assertEqual(before, after)

    def test_repeated_click_target_is_eligible(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, rows), _ = self.run_fixture(Path(temporary) / 'source')
            row = next(r for r in rows if r['user_id'] == 1 and r['request_timestamp'] == 150)
            self.assertEqual(row['target_id'], 10)
            self.assertIn(10, row['article_ids'])

    def test_replay_scores_reachable_targets_as_hits(self):
        with tempfile.TemporaryDirectory() as temporary:
            (metrics, _), _ = self.run_fixture(Path(temporary) / 'source')
            score = metrics['models']['popularity']['overall']['hit_rate@5']
            self.assertGreater(score, 0.)
            self.assertLess(score, 1.)

    def test_current_target_is_not_used_as_current_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, rows), _ = self.run_fixture(Path(temporary) / 'source')
            row = next(r for r in rows if r['user_id'] == 1 and r['request_timestamp'] == 100)
            self.assertEqual(row['history_length'], 2)
            self.assertTrue(row['zero_behavior_target'])

    def test_tied_targets_excluded_and_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            (metrics, _), _ = self.run_fixture(Path(temporary) / 'source', extra_train=[(2, 50, 120)])
            self.assertEqual(metrics['tie_transitions_excluded'], 1)

    def test_empty_training_and_invalid_cutoffs_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, _), db = self.run_fixture(Path(temporary) / 'source')
            with self.assertRaises(ValueError):
                replay(db, 0, 200)
            with self.assertRaises(ValueError):
                replay(db, 200, 100)

    def test_streamed_predictions_equal_collected_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            (before, rows), db = self.run_fixture(Path(temporary) / 'source')
            streamed = []
            after, empty = replay(db, 90, 200, k=5, on_prediction=streamed.append)
            self.assertEqual(before, after)
            self.assertEqual(rows, streamed)
            self.assertEqual(empty, [])

    def test_unknown_history_uses_deterministic_eligible_fallback(self):
        model = WindowedItemCF().fit([(1, 10, 1), (1, 20, 2)])
        metadata = {10: {'created': 0}, 20: {'created': 100}}
        self.assertEqual(model.recommend([999], metadata, 50, 5), [10])

    def test_popularity_ties_are_resolved_by_article_id(self):
        model = WindowedItemCF().fit([(1, 20, 1), (1, 10, 2)])
        self.assertEqual(model.popular_items, [10, 20])

    def test_training_boundary_is_not_also_an_evaluation_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            (_, _), db = self.run_fixture(Path(temporary) / 'source')
            _, rows = replay(db, 100, 200, k=5)
            self.assertTrue(all(row['request_timestamp'] > 100 for row in rows))


class DataTests(unittest.TestCase):
    def test_audit_records_duplicates_missing_metadata_and_unsorted_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fixture(root, extra_train=[(1, 10, 10), (5, 999, 50)])
            files = discover(root)
            db = load_database(files, root / 'audit.sqlite')
            self.addCleanup(db.close)
            result = audit(db, files)
            train = result['roles']['train']
            self.assertEqual(train['identical_event_extra_rows'], 1)
            self.assertEqual(train['click_rows_without_metadata'], 1)
            self.assertEqual(train['users_unsorted_in_source'], 1)

    def test_fail_on_corrupt_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_fixture(root)
            (root / 'articles.csv').write_text('article_id,wrong\n1,2\n')
            with self.assertRaises(ValueError):
                load_database(discover(root), root / 'audit.sqlite')

    def test_cli_manifest_failure_and_no_input_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            create_fixture(data)
            snapshot = {p.name: p.read_bytes() for p in data.glob('*.csv')}
            output = root / 'result'
            main(['baseline', '--data-dir', str(data), '--output', str(output),
                  '--scope', 'local-unverified', '--train-end', '100', '--eval-end', '200'])
            self.assertEqual(snapshot, {p.name: p.read_bytes() for p in data.glob('*.csv')})
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'succeeded')
            with self.assertRaises(FileExistsError):
                main(['audit', '--data-dir', str(data), '--output', str(output), '--scope', 'local-unverified'])
            broken = root / 'failed'
            with self.assertRaises(ValueError):
                main(['baseline', '--data-dir', str(data), '--output', str(broken),
                      '--scope', 'local-unverified', '--train-end', '0', '--eval-end', '200'])
            self.assertEqual(json.loads((broken / 'manifest.json').read_text())['status'], 'failed')

    def test_full_run_refuses_automatic_sample_cutoffs(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / 'data'
            data.mkdir()
            create_fixture(data)
            with self.assertRaises(SystemExit):
                main(['baseline', '--data-dir', str(data), '--output', str(Path(temporary) / 'out'),
                      '--scope', 'local-unverified', '--sample-smoke-cutoffs'])


if __name__ == '__main__':
    unittest.main()
