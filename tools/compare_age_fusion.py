"""Paired internal development gate; original reference remains unchanged."""
from collections import defaultdict
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import write_json
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--confirm-validation',action='store_true');args=parser.parse_args()
run=root/('runs/age-fusion-confirmation-v0' if args.confirm_validation else 'runs/age-fusion-ranker-dev-v0')
reference_dir=root/('runs/content-ranker-confirmation-v0' if args.confirm_validation else 'runs/content-ranker-dev-v0')
if args.confirm_validation:
    from newsrec.data import file_hash
    assert file_hash(reference_dir/'base15.txt')=='5dd61654fc253657a5cb9598d4dfaad78af36dc777c00dcde2afc4251daeb82e'
    manifest=json.loads((run/'manifest.json').read_text())
    assert (manifest['protocol']['train_end_ms'],manifest['protocol']['validation_end_ms'])==(1507757054762,1507980021008)
reference={r['request_id']:r for r in map(json.loads,(reference_dir/'predictions.jsonl').open())}
users=defaultdict(lambda:np.zeros(4));blocks=defaultdict(lambda:np.zeros(4));counts=defaultdict(int);seen=set()
start=json.loads((run/'manifest.json').read_text())['protocol']['train_end_ms']
for r in map(json.loads,(run/'predictions.jsonl').open()):
    rid=r['request_id'];assert rid not in seen;seen.add(rid);old=reference[rid]
    for key in ('user_id','request_timestamp','target_id','target_timestamp','target_available','zero_behavior_target','history_length'):assert r[key]==old[key],key
    target=r['target_id'];hits=[int(target in old['article_ids']['base15'][:5]),int(target in r['article_ids']['base15'][:5]),int(target in r['article_ids']['content18'][:5])]
    users[r['user_id']]+=np.array([1,*hits]);blocks[(r['request_timestamp']-start)//21600000]+=np.array([1,*hits])
    counts['gains']+=int(hits[2] and not hits[0]);counts['losses']+=int(hits[0] and not hits[2])
assert seen==reference.keys() and len(seen)==(114855 if args.confirm_validation else 107904)
values=np.array(list(users.values()));rng=np.random.default_rng(2026);boot=[]
for _ in range(400):
    s=values[rng.integers(0,len(values),len(values))].sum(axis=0);boot.append((s[3]-s[1])/s[0])
totals=values.sum(axis=0);ci=np.quantile(boot,[.025,.975]);passed=bool(totals[3]>max(totals[1:3]) and ci[0]>0)
result=dict(requests=len(seen),users=len(users),original_base15_hr5=totals[1]/totals[0],fused15_hr5=totals[2]/totals[0],fused18_hr5=totals[3]/totals[0],fused18_vs_original_delta=(totals[3]-totals[1])/totals[0],user_cluster_ci95=ci.tolist(),gate_passed=passed,**counts,six_hour_blocks={str(k):v.tolist() for k,v in blocks.items()},scope=('fixed original-validation confirmation; previously explored; not independent final' if args.confirm_validation else 'internal development; no original validation or final scoring'),names='base15/content18 in run files mean fused15/fused18; original control is reference_dir base15',reference_dir=str(reference_dir))
write_json(run/'comparison.json',result);print(json.dumps(result,indent=2))
