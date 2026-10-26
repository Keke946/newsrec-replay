"""Require exact old/new rolling control parity, then paired validation analysis."""
import argparse
import json
from pathlib import Path
from compare_validation import compare


def verify(original,current):
    old={}
    with Path(original).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['model']=='rolling_trend':
                if row['request_id'] in old:raise ValueError('Duplicate original request')
                old[row['request_id']]=row
    count=0
    with Path(current).open() as stream:
        for line in stream:
            row=json.loads(line)
            if row['model']!='rolling_trend':continue
            reference=old.pop(row['request_id'])
            if row!=reference:
                differences=[key for key in set(row)|set(reference) if row.get(key)!=reference.get(key)]
                raise ValueError(f'Control changed at request {row["request_id"]}: {differences}')
            count+=1
    if old:raise ValueError('Current run omits original requests')
    return {'requests':count,'exact_prediction_record_equality':True,
            'scope':'unchanged rolling trend control, including ordered candidates and labels'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    parity=verify(args.original,args.run/'predictions.jsonl')
    with (args.run/'baseline_parity.json').open('x') as stream:json.dump(parity,stream,indent=2)
    for reference in ['rolling_trend','same_pool_6h']:
        result=compare(args.run/'predictions.jsonl',reference=reference,challenger='lambdarank_time')
        result['limitations'][0]='same-pool control for same_pool_6h; models also differ in supplied dynamic/user features'
        with (args.run/f'paired_vs_{reference}.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(parity)


if __name__=='__main__':main()
