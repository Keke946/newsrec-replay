"""Fixed 6h publication-age internal retrieval; no model training or window search."""
import json,sqlite3,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.two_tower import VectorStore,TwoTower
from newsrec.evaluation import RankingMetrics
from newsrec.data import write_json,file_hash
ROOT=Path(__file__).resolve().parents[1]

def eligible(created,t):return (created>=t-21600000)&(created<=t)

def top(ids,scores,k=10):return np.lexsort((ids,-scores))[:k]

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    out=ROOT/'runs/age-retrieval-dev-v0';out.mkdir(exist_ok=False)
    source=ROOT/'runs/mixed-negative-dev-v0'
    manifest=json.loads((source/'manifest.json').read_text())
    write_json(out/'manifest.json',dict(status='running',protocol=manifest['protocol'],age_ms=21600000,training=False))
    conn=sqlite3.connect(f'file:{ROOT}/runs/recovered-audit/audit.sqlite?mode=ro',uri=True)
    store=VectorStore(ROOT/'data/runtime',conn)
    assert file_hash(ROOT/'data/runtime/article_vectors.f32')==manifest['protocol']['vector_sha256']
    n=json.loads((source/'development/examples.json').read_text())['examples'];assert n==107904
    records=np.load(source/'development/records.npy',mmap_mode='r')[:n]
    queries=np.load(source/'development/queries.npy',mmap_mode='r')[:n]
    raw=store.vectors(np.arange(len(store.ids))); methods={'cosine':(raw,np.asarray(queries))}
    for name in ('inbatch','mixed'):
        model=TwoTower();model.load_state_dict(torch.load(source/name/'model_state.pt',weights_only=True));model.eval()
        gallery=np.concatenate([model.items(torch.from_numpy(raw[s:s+4096])).numpy() for s in range(0,len(raw),4096)])
        users=np.concatenate([model.users(torch.from_numpy(np.array(queries[s:s+4096],copy=True))).numpy() for s in range(0,n,4096)])
        methods[name]=(gallery,users)
    order=np.argsort(store.created,kind='stable');dates=store.created[order]
    metrics={name:RankingMetrics() for name in methods};streams={}
    for name in methods:
        (out/name).mkdir();streams[name]=(out/name/'predictions.jsonl').open('w')
    start=time.perf_counter();eligible_targets=0;sizes=[]
    try:
        for j,row in enumerate(records):
            u,rid,t,target,tt,length=map(int,row)
            lo,hi=np.searchsorted(dates,[t-21600000,t],side='left');hi=np.searchsorted(dates,t,side='right')
            selected=order[lo:hi];sizes.append(len(selected));eligible_targets+=int(t-21600000<=store.created[target]<=t)
            for name,(gallery,users) in methods.items():
                scores=gallery[selected]@users[j]
                picked=selected[top(store.ids[selected],scores)]
                ids=list(map(int,store.ids[picked]));target_id=int(store.ids[target]);metrics[name].add(target_id,ids,user_id=u)
                streams[name].write(json.dumps(dict(request_id=rid,user_id=u,request_timestamp=t,target_id=target_id,target_timestamp=tt,target_available=bool(store.created[target]<=t),history_length=length,article_ids=ids))+'\n')
            if j%10000==0:print(j,flush=True)
    finally:
        for stream in streams.values():stream.close()
        conn.close()
    result=dict(requests=n,eligible_targets=eligible_targets,gallery_mean=float(np.mean(sizes)),gallery_min=min(sizes),gallery_max=max(sizes),seconds=time.perf_counter()-start,methods={k:v.result() for k,v in metrics.items()})
    write_json(out/'metrics.json',result);write_json(out/'manifest.json',dict(status='succeeded',protocol=manifest['protocol'],age_ms=21600000,training=False,source_weights={name:file_hash(source/name/'model_state.pt') for name in ('inbatch','mixed')}));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
