"""Bounded real-validation prefix replay, restart mid-batch, full-output parity.

No fit or final evaluation. The earlier full offline/service check is separate.
"""
import argparse
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash
from newsrec.journal import JournalEngine
from serve_replay import load_engine,journal_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--model-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,default=1000)
    args=parser.parse_args()
    if args.requests<2:parser.error('At least two requests required')
    if args.output.exists():parser.error('Output exists')
    args.output.mkdir(parents=True)
    expected={}
    with (args.model_run/'predictions.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['model']=='lambdarank_time':
                expected[row['request_id']]=row
                if len(expected)==args.requests:break
    if len(expected)!=args.requests:raise ValueError('Requested prefix unavailable')
    manifest=json.loads((args.model_run/'manifest.json').read_text())
    identity=journal_identity(args.database,args.model_run)
    factory=lambda:load_engine(args.database,args.model_run)
    control=factory()
    journal_path=args.output/'actions.sqlite'
    durable=JournalEngine(journal_path,factory,identity)
    connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    count=0;event_count=0;replay_seconds=None;latencies=[];restart_actions=None
    stop=max(row['request_timestamp'] for row in expected.values())
    start=time.perf_counter()
    try:
        rows=connection.execute('SELECT timestamp,ordinal,user,article FROM clicks WHERE role="train" '
                                'AND timestamp>? AND timestamp<=? ORDER BY timestamp,ordinal',
                                (manifest['protocol']['train_end_ms'],stop))
        for timestamp,batch in groupby(rows,lambda row:row[0]):
            events=list(batch)
            for _,ordinal,user,article in events:
                if ordinal not in expected:continue
                reference=control.recommend(user,timestamp,current_article=article)
                before=time.perf_counter()
                actual=durable.recommend(user,timestamp,current_article=article)
                latencies.append((time.perf_counter()-before)*1000)
                if actual!=reference:raise ValueError(f'Full durable/control response differs: {ordinal}')
                if actual['article_ids']!=expected[ordinal]['article_ids'] or actual['history_length']!=expected[ordinal]['history_length']:
                    raise ValueError(f'Offline prediction differs: {ordinal}')
                count+=1
                if count==args.requests//2:
                    # Restart after a query, before revealing its timestamp batch.
                    # Re-request the same query after replay and compare scores too.
                    durable.close()
                    before=time.perf_counter()
                    durable=JournalEngine(journal_path,factory,identity)
                    replay_seconds=time.perf_counter()-before
                    restart_actions=durable.replayed_actions
                    if durable.recommend(user,timestamp,current_article=article)!=reference:
                        raise ValueError('Restart query differs')
            payload=[dict(event_id=o,user_id=u,article_id=a) for _,o,u,a in events]
            if durable.ingest(timestamp,payload)!=control.ingest(timestamp,payload):
                raise ValueError('Ingest response differs')
            event_count+=len(events)
        action_count=durable.connection.execute('SELECT count(*) FROM actions').fetchone()[0]
        durable.close()
        # Second restart also checks the final ingest, plus journal integrity.
        durable=JournalEngine(journal_path,factory,identity)
        integrity=durable.connection.execute('PRAGMA integrity_check').fetchone()[0]
        if integrity!='ok' or count!=args.requests:raise ValueError('Integrity/count differs')
        output=dict(status='succeeded',scope='chronological validation prefix; not full-cohort durability',
                    requests=count,observed_events_after_cutoff=event_count,
                    exact_full_responses_vs_uninterrupted=True,exact_top10_and_history_vs_offline=True,
                    restart_after_request=args.requests//2,restarts=2,actions_at_first_restart=restart_actions,
                    committed_actions=action_count,integrity_check=integrity,
                    first_restart_seconds=replay_seconds,elapsed_seconds=time.perf_counter()-start,
                    identity=identity,final_region_scored=False,
                    limitations=['internal calls, no HTTP load test','one owner; Linux advisory file lock',
                                 'crash/disk error injection covered separately by synthetic tests',
                                 'no distributed exactly-once or hardware power-loss guarantee',
                                 'full log replay is linear and includes priming; no compact checkpoints'])
        import numpy as np
        output['durable_recommend_internal_ms']=dict(p50=float(np.quantile(latencies,.5)),p95=float(np.quantile(latencies,.95)))
    finally:
        durable.close();connection.close()
    output['journal_sha256']=file_hash(journal_path)
    output['journal_bytes']=journal_path.stat().st_size
    (args.output/'report.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
