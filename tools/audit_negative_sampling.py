"""Reconstruct training targets and audit the exact seed2026 shuffled batches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--end',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    conn=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    created=dict(conn.execute('SELECT article,created FROM articles'))
    rows=conn.execute('WITH p AS (SELECT user,ordinal,timestamp,'
                      'lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,'
                      'lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time '
                      'FROM clicks WHERE role="train" AND timestamp<=?) '
                      'SELECT user,ordinal,timestamp,target,target_time FROM p WHERE target_time>timestamp '
                      'ORDER BY user,timestamp,ordinal',(args.end,))
    records=[];unavailable=0
    for user,rid,t,target,tt in rows:
        if created[target]>t:unavailable+=1
        else:records.append((user,rid,t,target,tt))
    conn.close();records=np.asarray(records,dtype=np.int64)
    np.save(args.output/'records.npy',records,allow_pickle=False)
    counts=Counter(map(int,records[:,3]));n=len(records);order=np.random.default_rng(2026).permutation(n)
    totals=Counter();valid_counts=[];unique_counts=[]
    for start in range(0,n,256):
        batch=records[order[start:start+256]];items=batch[:,3];times=batch[:,2]
        ages=np.asarray([created[int(i)] for i in items])
        distinct=items[:,None]!=items[None,:];legal=ages[None,:]<=times[:,None]
        diagonal=np.eye(len(batch),dtype=bool)
        usable=distinct&legal
        ids=np.unique(items);uc=np.asarray([created[int(i)] for i in ids])
        unique=((uc[None,:]<=times[:,None])&(ids[None,:]!=items[:,None])).sum(axis=1)
        valid_counts.extend(usable.sum(axis=1).tolist());unique_counts.extend(unique.tolist())
        totals.update(batches=1,attempted_negative_logits=len(batch)*(len(batch)-1),
                      future_removed=int(((~legal)&(~diagonal)).sum()),
                      duplicate_positive_removed=int(((~distinct)&legal&(~diagonal)).sum()),
                      usable_negative_logits=int(usable.sum()))
    head=sorted(counts.values(),reverse=True)
    report=dict(status='succeeded',training_examples=n,unavailable_training_positives_excluded=unavailable,
                published_catalog=sum(t<=args.end for t in created.values()),positive_target_articles=len(counts),
                top_target_share={str(k):sum(head[:k])/n for k in [1,10,100,1000]},batch_totals=dict(totals),
                mean_usable_negatives=float(np.mean(valid_counts)),mean_unique_usable_negatives=float(np.mean(unique_counts)),
                usable_negative_quantiles=np.quantile(valid_counts,[0,.05,.5,.95,1]).tolist(),
                sampling='exact original target ordering, np seed2026 permutation, batch256; no neural inference',
                database_sha256=file_hash(args.database),end=args.end,
                limitations=['not an exposure audit; sampled negatives are unlabeled','describes mechanism, does not prove failure cause',
                             'same non-positive item may occupy multiple usable negative columns; not equivalent to duplicate-positive bug'])
    write_json(args.output/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
