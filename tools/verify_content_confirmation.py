"""Verify exact old-baseline recovery and paired cohort before interpreting gains."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args()
    m=json.loads((args.run/'metrics.json').read_text())
    old=json.loads((args.reference/'metrics.json').read_text())
    checks={
        'exact_baseline_model':file_hash(args.run/'base15.txt')==file_hash(args.reference/'ranker.txt'),
        'exact_training_base_features':m['shared_base_feature_sha256']==file_hash(args.reference/'train_features.f32'),
        'exact_training_labels':m['labels_sha256']==file_hash(args.reference/'train_labels.i8'),
        'exact_baseline_aggregate_metrics':m['models']['base15']['all']==old['models']['lambdarank_time']['overall'],
        'exact_candidate_hits':m['diagnostics']['candidate_hits']==old['diagnostics']['validation_candidate_hits'],
    }
    if not all(checks.values()):raise ValueError(f'Control differs: {checks}')
    count=gains=losses=0
    keys=('request_id','user_id','request_timestamp','target_id','target_timestamp','target_available','zero_behavior_target','history_length')
    with (args.reference/'predictions.jsonl').open() as reference,(args.run/'predictions.jsonl').open() as actual:
        references=(r for r in map(json.loads,reference) if r['model']=='lambdarank_time')
        for r in map(json.loads,actual):
            ref=next(references)
            if any(r[k]!=ref[k] for k in keys):raise ValueError('Paired request identity differs')
            if r['article_ids']['base15']!=ref['article_ids']:raise ValueError('Baseline top10 differs')
            a=r['target_id'] in r['article_ids']['base15'][:5]
            b=r['target_id'] in r['article_ids']['content18'][:5]
            gains+=int(b and not a);losses+=int(a and not b);count+=1
        if next(references,None) is not None:raise ValueError('Missing predictions')
    checks.update(exact_request_fields_and_baseline_top10=True,requests=count,
                  hr5_gained_requests=gains,hr5_lost_requests=losses,hr5_net_hits=gains-losses,
                  hr5_delta=(gains-losses)/count,
                  reference_predictions_sha256=file_hash(args.reference/'predictions.jsonl'),
                  predictions_sha256=file_hash(args.run/'predictions.jsonl'))
    write_json(args.run/'baseline_parity.json',checks);print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
