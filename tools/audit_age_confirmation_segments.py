"""Post-hoc descriptive error segments; no tuning or adoption-gate changes."""
import json,sqlite3
from collections import defaultdict,Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1];run=root/'runs/age-fusion-confirmation-v0'
conn=sqlite3.connect(f'file:{root}/runs/recovered-audit/audit.sqlite?mode=ro',uri=True);created=dict(conn.execute('SELECT article,created FROM articles'));conn.close()
old={r['request_id']:r for r in map(json.loads,(root/'runs/content-ranker-confirmation-v0/predictions.jsonl').open())};groups=defaultdict(Counter)
for r in map(json.loads,(run/'predictions.jsonl').open()):
    age=r['request_timestamp']-created[r['target_id']];a=int(r['target_id'] in old[r['request_id']]['article_ids']['base15'][:5]);b=int(r['target_id'] in r['article_ids']['content18'][:5])
    keys=['target_unavailable' if age<0 else 'target_age_0_6h' if age<=21600000 else 'target_age_over6h','history_le5' if r['history_length']<=5 else 'history_gt5','cold' if r['zero_behavior_target'] else 'warm']
    for k in keys:groups[k].update(requests=1,original_hits=a,fused18_hits=b,gains=int(b and not a),losses=int(a and not b))
result=dict(scope='explicit post-hoc descriptive labels only; no new fit or gate change; not serving features',groups={k:dict(v) for k,v in groups.items()})
(run/'posthoc_segments.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
