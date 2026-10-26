"""Descriptive oracle candidate coverage, never a realizable ranker score."""
import argparse
from collections import defaultdict
from itertools import groupby
import json
from pathlib import Path


def summarize(path):
    counts=defaultdict(lambda:defaultdict(int))
    with Path(path).open() as stream:
        for _,rows in groupby((json.loads(line) for line in stream),lambda row:row['request_id']):
            rows=list(rows)
            by_model={row['model']:row for row in rows}
            reference=by_model['rolling_trend']
            groups=['all']
            if reference['zero_behavior_target']:
                groups.append('zero-behavior')
                if reference['target_available']:
                    groups.append('zero-behavior-and-target-available')
            target=reference['target_id']
            for name,row in by_model.items():
                candidates=set(reference['article_ids'][:10])|set(row['article_ids'][:10])
                own=target in row['article_ids'][:10]
                trend=target in reference['article_ids'][:10]
                for group in groups:
                    key=f'{name}/{group}'
                    counts[key]['requests']+=1
                    counts[key]['method_top10_hits']+=int(own)
                    counts[key]['rolling_top10_hits']+=int(trend)
                    counts[key]['both_top10_hits']+=int(own and trend)
                    counts[key]['method_only_top10_hits']+=int(own and not trend)
                    counts[key]['union_hits']+=int(target in candidates)
                    counts[key]['total_union_candidates']+=len(candidates)
    result={}
    for key,values in counts.items():
        result[key]=dict(values)
        result[key]['oracle_union_coverage']=values['union_hits']/values['requests']
        result[key]['mean_union_candidates']=values['total_union_candidates']/values['requests']
    return {'scope':'validation-only candidate complementarity',
            'warning':'union has up to 20 candidates vs each method top10; oracle coverage is not HR@5 or a realizable ranker result',
            'zero_behavior_definition':'first train-role click strictly after request; same-time first clicks excluded',
            'groups':result}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.output.open('x') as stream:
        json.dump(summarize(args.predictions),stream,indent=2)
        stream.write('\n')


if __name__=='__main__':
    main()
