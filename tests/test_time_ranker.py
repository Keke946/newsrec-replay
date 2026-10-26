import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.time_ranker import CausalCandidates,CONFIG,FEATURES,EventWindow


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.metadata={10:dict(category=1,created=0,words=100),
                       20:dict(category=2,created=0,words=200),
                       30:dict(category=1,created=500,words=300)}

    def test_window_boundary_and_candidate_availability(self):
        window=EventWindow(100)
        window.observe(10,10);window.observe(50,20);window.expire(110)
        self.assertNotIn(10,window.counts)
        state=CausalCandidates(self.metadata)
        state.observe(100,10);state.observe(100,30)
        trend,items,features=state.query([(20,200)],200,1)
        self.assertEqual(trend,[10])
        self.assertEqual(items,[10,20])
        self.assertTrue(all(len(row)==len(FEATURES) for row in features))
        self.assertEqual(features[1][FEATURES.index('item_match_last')],1)

    def test_future_state_is_rejected(self):
        state=CausalCandidates(self.metadata)
        state.observe(500,10)
        with self.assertRaisesRegex(ValueError,'future'):
            state.query([(20,100)],100,1)

    def test_own_event_only_enters_history_until_batch_reveal(self):
        state=CausalCandidates(self.metadata)
        state.observe(10,10)
        _,items,features=state.query([(20,20)],20,1)
        row=features[items.index(20)]
        self.assertEqual(row[FEATURES.index('log_count_6h')],0)
        self.assertEqual(row[FEATURES.index('category_match_last')],1)
        state.observe(20,20)
        _,items,features=state.query([(20,21)],21,2)
        self.assertGreater(features[items.index(20)][0],0)


class RankerIntegration(unittest.TestCase):
    def test_invalid_active_features_rejected_before_any_data_read(self):
        try:
            import lightgbm
        except ImportError:
            self.skipTest('optional LightGBM environment required')
        from newsrec.time_ranker import run
        for columns in [[],[0,0],[-1],[len(FEATURES)]]:
            with self.assertRaisesRegex(ValueError,'active feature'):
                run(None,Path('.'),dict(train_end_ms=1,validation_end_ms=2),dict(CONFIG,feature_columns=columns))

    def test_future_events_do_not_change_trained_model_or_past_predictions(self):
        try:
            import lightgbm
        except ImportError:
            self.skipTest('optional LightGBM environment required')
        from test_contracts import create_fixture
        from newsrec.data import discover,load_database
        from newsrec.time_ranker import run
        from newsrec.baseline import replay
        outputs=[]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for index,extra in enumerate([(),[(1,30,999)]]):
                data=root/f'data{index}';data.mkdir()
                create_fixture(data,extra_train=extra)
                connection=load_database(discover(data),root/f'audit{index}.sqlite')
                # Span multiple five-minute cache buckets, unlike toy ms times.
                connection.execute('UPDATE clicks SET timestamp=timestamp*100000')
                connection.execute('UPDATE articles SET created=created*100000')
                directory=root/f'run{index}';directory.mkdir()
                config=dict(CONFIG,n_estimators=3,min_child_samples=1,train_ordinal_modulus=1,threads=1)
                result=run(connection,directory,dict(train_end_ms=10000000,validation_end_ms=20000000),config)
                baseline,_=replay(connection,10000000,20000000,include_trend=True)
                self.assertEqual(result['models']['rolling_trend']['overall'],baseline['models']['rolling_trend']['overall'])
                rows=[json.loads(line) for line in (directory/'predictions.jsonl').read_text().splitlines()]
                self.assertEqual({row['request_id'] for row in rows},{8,12,4})
                self.assertTrue(all(row['target_timestamp']<=20000000 for row in rows))
                self.assertEqual(result['diagnostics']['validation_requests'],3)
                outputs.append((directory/'predictions.jsonl').read_text())
                connection.close()
            self.assertEqual(outputs[0],outputs[1])
