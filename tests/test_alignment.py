import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
try:
    from analyze_retrieval import align
except ImportError:
    align=None


@unittest.skipIf(align is None,'optional NumPy analysis dependency required')
class AlignmentTests(unittest.TestCase):
    def test_matching_cohort_and_label_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            row={'request_id':2,'user_id':1,'request_timestamp':20,'target_id':30,
                 'target_timestamp':40,'target_available':True,'history_length':1,
                 'zero_behavior_target':False,'model':'itemcf','article_ids':[30]}
            (root/'baseline').write_text(json.dumps(row)+'\n')
            extra=dict(row,model='content_cosine',article_ids=[])
            (root/'extra').write_text(json.dumps(extra)+'\n')
            result=align(root/'baseline',[root/'extra'],root/'joined')
            self.assertEqual(result['aligned_requests'],1)
            self.assertEqual(result['metrics']['content_cosine']['all']['hit_rate@5'],0)
            extra['target_id']=99
            (root/'extra').write_text(json.dumps(extra)+'\n')
            with self.assertRaisesRegex(ValueError,'target_id'):
                align(root/'baseline',[root/'extra'],root/'mismatch')

    def test_missing_and_extra_requests_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            row={'request_id':2,'user_id':1,'request_timestamp':20,'target_id':30,
                 'target_timestamp':40,'target_available':True,'history_length':1,
                 'zero_behavior_target':False,'model':'itemcf','article_ids':[30]}
            (root/'baseline').write_text(json.dumps(row)+'\n')
            (root/'extra').write_text('')
            with self.assertRaises(KeyError):
                align(root/'baseline',[root/'extra'],root/'missing')
            (root/'baseline').write_text('')
            (root/'extra').write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'Extra requests'):
                align(root/'baseline',[root/'extra'],root/'extra-result')
