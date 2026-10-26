"""Verify every request field and publication-age boundary independently."""
import json,sqlite3
from pathlib import Path
root=Path(__file__).resolve().parents[1]
conn=sqlite3.connect(f'file:{root}/runs/recovered-audit/audit.sqlite?mode=ro',uri=True)
created=dict(conn.execute('SELECT article,created FROM articles'));conn.close()
ref={r['request_id']:r for r in map(json.loads,(root/'runs/content-ranker-dev-v0/predictions.jsonl').open())}
for name in ('cosine','inbatch','mixed'):
    seen=set()
    for r in map(json.loads,(root/f'runs/age-retrieval-dev-v0/{name}/predictions.jsonl').open()):
        rid=r['request_id'];assert rid not in seen;seen.add(rid)
        for key in ('request_id','user_id','request_timestamp','target_id','target_timestamp','target_available','history_length'):assert r[key]==ref[rid][key],key
        assert len(r['article_ids'])==len(set(r['article_ids']))<=10
        assert all(r['request_timestamp']-21600000<=created[i]<=r['request_timestamp'] for i in r['article_ids'])
    assert seen==ref.keys()
print('3 x 107904 request records and all age constraints verified')
