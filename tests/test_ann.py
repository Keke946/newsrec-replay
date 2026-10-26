import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import faiss
    import numpy as np
    from newsrec.ann import normalized,SnapshotIndex,fidelity
except ImportError:
    faiss=None


@unittest.skipIf(faiss is None,'optional Faiss/NumPy dependencies required')
class ANNTests(unittest.TestCase):
    def test_exact_cosine_normalization_and_temporal_contract(self):
        index=SnapshotIndex([1,2,3],[[4,0],[0,2],[-2,0]],[0,0,5],5)
        score,ids=index.search([[8,0]],[5],k=2)
        self.assertEqual(ids.tolist(),[[1,2]])
        self.assertAlmostEqual(float(score[0,0]),1)
        with self.assertRaisesRegex(ValueError,'newer'):
            index.search([[1,0]],[4],k=2)
        with self.assertRaisesRegex(ValueError,'Future'):
            SnapshotIndex([1],[[1,0]],[6],5)

    def test_invalid_vectors_ids_and_shapes_fail(self):
        for values in [[[0,0]],[[float('nan'),0]]]:
            with self.assertRaises(ValueError):normalized(values)
        with self.assertRaises(ValueError):SnapshotIndex([1,1],[[1,0],[0,1]],[0,0],5)
        with self.assertRaises(ValueError):fidelity([[1,2]],[[1,1]],[[1,.5]],[[1,1]])

    def test_tie_aware_fidelity_is_distinct_from_id_overlap(self):
        values=fidelity([[1,2]],[[1,3]],[[1,.5]],[[1,.5]])
        self.assertEqual(values['mean_strict_neighbor_recall'],.5)
        self.assertEqual(values['mean_score_threshold_recall'],1)

    def test_hnsw_and_serialization_roundtrip(self):
        faiss.omp_set_num_threads(1)
        vectors=np.random.default_rng(2026).normal(size=(100,8)).astype('float32')
        exact=SnapshotIndex(np.arange(100),vectors,np.zeros(100,dtype='int64'),0)
        approximate=SnapshotIndex(np.arange(100),vectors,np.zeros(100,dtype='int64'),0,kind='hnsw')
        x,y=exact.search(vectors[:5],np.zeros(5),k=10)
        a,b=approximate.search(vectors[:5],np.zeros(5),k=10,ef_search=64)
        self.assertEqual(fidelity(y,b,x,a)['mean_strict_neighbor_recall'],1)
        restored=faiss.deserialize_index(faiss.serialize_index(approximate.index))
        np.testing.assert_equal(restored.search(normalized(vectors[:5]),10)[1],
                                approximate.index.search(normalized(vectors[:5]),10)[1])
