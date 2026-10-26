"""Align separate retrieval outputs to the existing complete temporal baseline.

Fail on missing, extra, duplicate or differently labelled requests. Keep the
joined per-request predictions private; write aggregate evidence separately.
"""
import argparse
from itertools import groupby
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from newsrec.evaluation import RankingMetrics
from compare_validation import compare


def align(baseline, extras, output):
    indexed = []
    for path in extras:
        rows = {}
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['request_id'] in rows:
                    raise ValueError('Duplicate extra request')
                rows[row['request_id']] = row
        indexed.append(rows)
    seen = set()
    groups = {}
    with Path(baseline).open() as stream, Path(output).open('x') as destination:
        for request_id, batch in groupby((json.loads(line) for line in stream), lambda row: row['request_id']):
            if request_id in seen:
                raise ValueError('Repeated baseline request')
            seen.add(request_id)
            rows = list(batch)
            reference = rows[0]
            for lookup in indexed:
                row = lookup.pop(request_id)  # missing requests must fail
                for key in ('user_id', 'request_timestamp', 'target_id', 'target_timestamp',
                            'target_available', 'history_length'):
                    if row[key] != reference[key]:
                        raise ValueError(f'Request {request_id} differs: {key}')
                row['zero_behavior_target'] = reference['zero_behavior_target']
                rows.append(row)
            if len({row['model'] for row in rows}) != len(rows):
                raise ValueError('Duplicate method for request')
            for row in rows:
                model = row['model']
                if model not in groups:
                    groups[model] = {key:RankingMetrics((5,10)) for key in ('all','zero-behavior','warm','history<=5','history>5')}
                keys = ['all', 'zero-behavior' if reference['zero_behavior_target'] else 'warm',
                        'history<=5' if reference['history_length'] <= 5 else 'history>5']
                for key in keys:
                    groups[model][key].add(row['target_id'], row['article_ids'], user_id=row['user_id'])
                destination.write(json.dumps(row)+'\n')
    if any(lookup for lookup in indexed):
        raise ValueError('Extra requests absent from baseline')
    return {'aligned_requests':len(seen), 'metrics':{model:{key:value.result() for key,value in values.items()}
                                                   for model,values in groups.items()},
            'scope':'same validation cohort; zero-behavior diagnostic inherited from frozen baseline',
            'zero_behavior_definition':'first train-role click time strictly later than request; ties are not cold here',
            'warning':'methods differ in representations and statistics updates; final evaluation untouched'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--extra',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True)
    joined=args.output/'joined_predictions.jsonl'
    result=align(args.baseline,args.extra,joined)
    (args.output/'cohort_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    for reference,challenger in [('content_cosine','two_tower_content_v0'),('rolling_trend','two_tower_content_v0')]:
        result=compare(joined,reference=reference,challenger=challenger)
        (args.output/f'{challenger}_vs_{reference}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Complete cohort aligned; aggregate and paired reports written.')


if __name__=='__main__':
    main()
