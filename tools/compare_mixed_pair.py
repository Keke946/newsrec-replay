"""Compare fixed internal retrieval pair; no re-fitting or parameter selection."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import write_json
from newsrec.evaluation import RankingMetrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    args=parser.parse_args()
    manifest=json.loads((args.run/'manifest.json').read_text())
    if manifest['status']!='succeeded':raise ValueError('Run not successful')
    audit=json.loads((args.audit/'report.json').read_text())
    if manifest['models']['mixed']['training_log']['usable_negative_logits']!=audit['batch_totals']['usable_negative_logits']:
        raise ValueError('Actual negative budget differs')
    a=manifest['models']['inbatch'];b=manifest['models']['mixed']
    if a['initial_state_sha256']!=b['initial_state_sha256'] or a['training_log'][0]['updates']!=b['training_log']['updates']:
        raise ValueError('Initial state or optimizer budget differs')
    with (args.reference/'predictions.jsonl').open() as f:reference={r['request_id']:r for r in map(json.loads,f)}
    predictions={}
    keys=('user_id','request_timestamp','target_id','target_timestamp','target_available','history_length')
    metrics={};users=defaultdict(lambda:np.zeros(3));blocks=defaultdict(lambda:np.zeros(3))
    for name in ['inbatch','mixed']:
        with (args.run/name/'predictions.jsonl').open() as f:predictions[name]={r['request_id']:r for r in map(json.loads,f)}
        if predictions[name].keys()!=reference.keys():raise ValueError('Request cohort differs')
        metrics[name]=defaultdict(lambda:RankingMetrics((5,10)))
        for rid,r in predictions[name].items():
            ref=reference[rid]
            if any(r[k]!=ref[k] for k in keys):raise ValueError('Request fields differ')
            for key in ['all','cold' if ref['zero_behavior_target'] else 'warm','history<=5' if r['history_length']<=5 else 'history>5']:
                metrics[name][key].add(r['target_id'],r['article_ids'],r['user_id'])
    for rid,ref in reference.items():
        hits=[int(ref['target_id'] in predictions[name][rid]['article_ids'][:5]) for name in ['inbatch','mixed']]
        users[ref['user_id']]+=np.asarray([1,*hits])
        block=(ref['request_timestamp']-manifest['protocol']['train_end_ms'])//(6*60*60*1000)
        blocks[block]+=np.asarray([1,*hits])
    pairs=np.asarray(list(users.values()));rng=np.random.default_rng(2026);samples=[]
    for _ in range(400):
        s=pairs[rng.integers(len(pairs),size=len(pairs))].sum(axis=0);samples.append((s[2]-s[1])/s[0])
    result=dict(exact_shared_cohort=True,exact_actual_negative_budget=True,exact_initial_state=True,
                actual_negative_logits=audit['batch_totals']['usable_negative_logits'],
                requests=len(reference),models={name:{k:v.result() for k,v in mm.items()} for name,mm in metrics.items()},
                user_cluster_hr5_delta_ci95=np.quantile(samples,[.025,.975]).tolist(),
                six_hour_blocks={str(k):dict(requests=int(v[0]),base_hits=int(v[1]),mixed_hits=int(v[2]),delta=(v[2]-v[1])/v[0]) for k,v in blocks.items()},
                warning='internal development only; no learned-ranker full-pool fusion or unbiased sampling claim')
    write_json(args.run/'comparison.json',result)
    print(json.dumps({name:m['all'] for name,m in result['models'].items()},indent=2))
    print('paired HR5 delta CI:',result['user_cluster_hr5_delta_ci95'])


if __name__=='__main__':main()
