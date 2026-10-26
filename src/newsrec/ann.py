"""Time-bounded Faiss retrieval: fidelity to exact neighbors, not click quality."""
import numpy as np


def normalized(vectors):
    vectors=np.ascontiguousarray(vectors,dtype=np.float32)
    if vectors.ndim!=2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('Need a nonempty finite vector matrix')
    norms=np.linalg.norm(vectors,axis=1)
    if (norms<=1e-12).any():raise ValueError('Zero vector cannot define cosine similarity')
    return np.ascontiguousarray(vectors/norms[:,None],dtype=np.float32)


class SnapshotIndex:
    def __init__(self,ids,vectors,created,snapshot_ms,kind='flat',m=16,ef_construction=80):
        import faiss
        ids=np.asarray(ids,dtype=np.int64)
        created=np.asarray(created,dtype=np.int64)
        vectors=normalized(vectors)
        if len(ids)!=len(vectors) or len(created)!=len(ids) or len(np.unique(ids))!=len(ids):
            raise ValueError('IDs, vectors and metadata must align uniquely')
        if (created>snapshot_ms).any():raise ValueError('Future item cannot enter an earlier snapshot')
        if kind not in ('flat','hnsw'):raise ValueError('Unknown index kind')
        self.ids=ids.copy()
        self.snapshot_ms=snapshot_ms
        self.dimension=vectors.shape[1]
        if kind=='flat':self.index=faiss.IndexFlatIP(self.dimension)
        else:
            self.index=faiss.IndexHNSWFlat(self.dimension,m,faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction=ef_construction
        self.index.add(vectors)

    def search(self,queries,timestamps,k=10,ef_search=None):
        if k<1 or k>len(self.ids):raise ValueError('k outside catalog bounds')
        queries=normalized(queries)
        timestamps=np.asarray(timestamps)
        if queries.shape[1]!=self.dimension or timestamps.shape!=(len(queries),):
            raise ValueError('Query shape or timestamp count differs')
        if (timestamps<self.snapshot_ms).any():raise ValueError('Index snapshot is newer than request')
        if ef_search is not None:
            if not hasattr(self.index,'hnsw') or ef_search<k:raise ValueError('Invalid HNSW search width')
            self.index.hnsw.efSearch=ef_search
        distances,positions=self.index.search(queries,k)
        if (positions<0).any():raise ValueError('Index returned incomplete results')
        return distances,self.ids[positions]


def fidelity(exact_ids,approximate_ids,exact_scores,approximate_scores,epsilon=1e-6):
    exact_ids=np.asarray(exact_ids);approximate_ids=np.asarray(approximate_ids)
    exact_scores=np.asarray(exact_scores);approximate_scores=np.asarray(approximate_scores)
    if exact_ids.shape!=approximate_ids.shape or exact_ids.ndim!=2 or not exact_ids.size:
        raise ValueError('Nonempty aligned neighbor matrices required')
    if exact_scores.shape!=exact_ids.shape or approximate_scores.shape!=exact_ids.shape:
        raise ValueError('Score and neighbor shapes differ')
    if not np.isfinite(exact_scores).all() or not np.isfinite(approximate_scores).all():
        raise ValueError('Nonfinite retrieval score')
    for row in approximate_ids:
        if len(set(row))!=len(row):raise ValueError('Duplicate approximate neighbors')
    strict=np.asarray([len(set(first)&set(second))/len(first) for first,second in zip(exact_ids,approximate_ids)])
    tied=(approximate_scores>=exact_scores[:,-1,None]-epsilon).mean(axis=1)
    return {'mean_strict_neighbor_recall':float(strict.mean()),
            'p05_strict_neighbor_recall':float(np.quantile(strict,.05)),
            'mean_score_threshold_recall':float(tied.mean()),
            'epsilon':epsilon,'queries':len(strict),'k':exact_ids.shape[1]}
