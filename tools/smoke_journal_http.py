"""Local real-HTTP restart smoke; synthetic events only in a new private journal."""
import argparse
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--model-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output exists')
    args.output.mkdir(parents=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
    base=f'http://127.0.0.1:{port}'
    def request(route,payload=None):
        encoded=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(base+route,data=encoded,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)
    def start(log):
        process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('serve_replay.py')),
                                  '--database',str(args.database),'--model-run',str(args.model_run),
                                  '--journal',str(args.output/'actions.sqlite'),'--port',str(port)],
                                 stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Server exited; inspect server.log')
            try:
                health=request('/health')
                if 'SQLite' not in health['state']:raise RuntimeError('Journal not enabled')
                return process
            except (urllib.error.URLError,TimeoutError):time.sleep(.1)
        process.terminate();process.wait(timeout=10)
        raise RuntimeError('Health deadline exceeded')
    def stop(process):
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    manifest=json.loads((args.model_run/'manifest.json').read_text())
    cutoff=manifest['protocol']['train_end_ms']
    connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    item=connection.execute('SELECT article FROM articles WHERE created<=? ORDER BY article LIMIT 1',(cutoff,)).fetchone()[0]
    connection.close()
    batch=dict(timestamp=cutoff+10,events=[dict(event_id=2000000000,user_id=2000000000,article_id=item)])
    query=dict(user_id=2000000000,timestamp=cutoff+20,current_article=item)
    with (args.output/'server.log').open('w') as log:
        process=start(log)
        try:
            assert request('/events',batch)['accepted']==1
            assert request('/events',batch)['duplicate_batch']
            expected=request('/recommend',query)
        finally:stop(process)
        process=start(log)
        try:
            assert request('/events',batch)['duplicate_batch']
            assert request('/recommend',query)==expected
            assert request('/health')['observed_through']==cutoff+10
        finally:stop(process)
    result=dict(status='succeeded',transport='real loopback HTTP',journal_enabled=True,
                server_starts=2,duplicate_batch_after_restart=True,full_recommend_response_after_restart=True,
                source='one synthetic timestamp batch and user; original DB opened read-only',
                limitations=['smoke only, not throughput/load or full-cohort HTTP verification',
                             'server shutdown here was graceful; abrupt-exit test is separate'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
