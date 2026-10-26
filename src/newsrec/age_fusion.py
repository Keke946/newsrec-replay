"""Fixed, label-free age-constrained cosine fusion for causal ranker experiments."""
from collections import Counter
import math
import numpy as np
from .time_ranker import CausalCandidates
from .fusion import fixed_count_fusion


def candidate_features(state,items,trend,history,timestamp,length):
    rank={i:1/(j+1) for j,i in enumerate(trend)}
    categories=Counter(state.metadata[i]['category'] for i,_ in history)
    short=Counter(state.metadata[i]['category'] for i,_ in history[-3:])
    gap=max(0,timestamp-history[-2][1])/1000 if len(history)>1 else 0
    out=[]
    for i in items:
        d=state.metadata[i];count=state.trend.counts.get(i,0);hour=state.hour.counts.get(i,0);quarter=state.quarter.counts.get(i,0)
        repeat=next((1/j for j,(a,_) in enumerate(reversed(history),1) if a==i),0.)
        out.append([math.log1p(count),math.log1p(hour),math.log1p(quarter),hour/max(count,1),quarter/max(count,1),
                    math.log1p(max(0,timestamp-d['created'])/1000),math.log1p(d['words']),categories[d['category']]/max(len(history),1),
                    float(bool(history) and d['category']==state.metadata[history[-1][0]]['category']),float(bool(history) and i==history[-1][0]),
                    repeat,math.log1p(length),rank.get(i,0.),short[d['category']]/max(min(3,len(history)),1),math.log1p(gap)])
    return out


class AgeCandidates(CausalCandidates):
    def __init__(self,metadata,config,vectors):
        super().__init__(metadata,config);self.vectors=vectors
        self.ids=np.array(list(vectors.lookup),dtype=np.int64)
        self.rows=np.array([vectors.lookup[int(i)] for i in self.ids])
        self.dates=np.array([metadata[int(i)]['created'] for i in self.ids]);self.order=np.argsort(self.dates,kind='stable')
        self.sorted_dates=self.dates[self.order]

    def retrieve(self,history,timestamp):
        if any(t>timestamp for _,t in history):raise ValueError('Future history')
        lo=np.searchsorted(self.sorted_dates,timestamp-21600000,side='left');hi=np.searchsorted(self.sorted_dates,timestamp,side='right')
        selected=self.order[lo:hi]
        if not history or not len(selected):return []
        history=history[-3:];weights=np.array([.7**d for d in range(len(history)-1,-1,-1)],dtype=np.float32)
        query=(self.vectors.normalized[[self.vectors.lookup[i] for i,_ in history]]*weights[:,None]).sum(axis=0)
        query/=max(float(np.linalg.norm(query)),1e-12)
        scores=self.vectors.normalized[self.rows[selected]]@query
        order=np.lexsort((self.ids[selected],-scores))[:10]
        return list(map(int,self.ids[selected[order]]))

    def query(self,history,timestamp,history_length):
        trend,base,_=super().query(history,timestamp,history_length)
        items=fixed_count_fusion(base,trend,history,self.retrieve(history,timestamp))
        return trend,items,candidate_features(self,items,trend,history,timestamp,history_length)
