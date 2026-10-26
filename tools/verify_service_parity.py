"""Replay real prefix events through the service engine; compare exact top10."""
import argparse
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash
from serve_replay import load_engine


def main():
    import numpy as np
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--model-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,default=1000)
    args=parser.parse_args()
    if args.requests<1:parser.error('requests must be positive')
    if args.output.exists():parser.error('output exists')
    expected={}
    with (args.model_run/'predictions.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['model']=='lambdarank_time':
                expected[row['request_id']]=row
                if len(expected)>=args.requests:break
    manifest=json.loads((args.model_run/'manifest.json').read_text())
    expected_total=json.loads((args.model_run/'metrics.json').read_text())['models']['lambdarank_time']['overall']['requests']
    engine=load_engine(args.database,args.model_run)
    stop=max(row['request_timestamp'] for row in expected.values())
    connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    count=0;durations=[]
    try:
        rows=connection.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" '
                                'AND timestamp>? AND timestamp<=? ORDER BY timestamp,ordinal',
                                (manifest['protocol']['train_end_ms'],stop))
        for timestamp,batch in groupby(rows,lambda row:row[0]):
            events=list(batch)
            for _,ordinal,user,article in events:
                if ordinal not in expected:continue
                start=time.perf_counter()
                result=engine.recommend(user,timestamp,current_article=article,k=10)
                durations.append((time.perf_counter()-start)*1000)
                reference=expected[ordinal]
                if result['article_ids']!=reference['article_ids'] or result['history_length']!=reference['history_length']:
                    raise ValueError(f'Service/offline mismatch: {ordinal}')
                count+=1
            engine.ingest(timestamp,[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in events])
        if count!=len(expected):raise ValueError('Missing parity requests')
    finally:connection.close()
    complete=count==expected_total
    output=dict(requests=count,reference_expected_requests=expected_total,full_validation_cohort=complete,
                exact_top10_and_history_length=True,
                model_sha256=file_hash(args.model_run/'ranker.txt'),
                reference_predictions_sha256=file_hash(args.model_run/'predictions.jsonl'),
                latency_ms=dict(p50=float(np.quantile(durations,.5)),p95=float(np.quantile(durations,.95)),
                                mean=float(np.mean(durations))),
                scope=('complete validation cohort' if complete else 'chronological validation prefix')+', real history, internal engine call',
                limitations=['internal engine timing excludes HTTP','one process and prediction thread, warmed model',
                             'not production load/SLA or durable state validation'])
    args.output.write_text(json.dumps(output,indent=2)+'\n');print(output)


if __name__=='__main__':main()
