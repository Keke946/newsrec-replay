"""Post-hoc algebraic temporal/target audit; no fit, tuning, final reads."""
from collections import Counter,defaultdict,deque
from itertools import groupby
import json,sqlite3,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import write_json,file_hash
ROOT=Path(__file__).resolve().parents[1]

def gap_bin(ms):
    if ms<=0:raise ValueError('Nonpositive prediction horizon')
    return 'le1m' if ms<=60000 else '1m_to30m' if ms<=1800000 else '30m_to6h' if ms<=21600000 else 'over6h'

def symmetric_decomposition(a,b):
    # delta validation minus development = composition + within-stratum terms
    na=sum(v['requests'] for v in a.values());nb=sum(v['requests'] for v in b.values())
    comp=within=0.;missing=[]
    for k in a.keys()|b.keys():
        x=a.get(k,{});y=b.get(k,{})
        if not x.get('requests') or not y.get('requests'):missing.append(k);continue
        pa=x['requests']/na;pb=y['requests']/nb;da=(x['fused18_hits']-x['original_hits'])/x['requests'];db=(y['fused18_hits']-y['original_hits'])/y['requests']
        comp+=(pb-pa)*(da+db)/2;within+=(db-da)*(pa+pb)/2
    return dict(composition_term=comp,within_stratum_term=within,unmatched_strata=missing,exact_if_no_unmatched=not missing)

def main():
    out=ROOT/'runs/temporal-target-audit-v0';out.mkdir(exist_ok=False)
    conn=sqlite3.connect(f'file:{ROOT}/runs/recovered-audit/audit.sqlite?mode=ro',uri=True);before=file_hash(ROOT/'runs/recovered-audit/audit.sqlite')
    created=dict(conn.execute('SELECT article,created FROM articles'));refs={};counts={};input_hashes={}
    for label,old,new,n in [('development','content-ranker-dev-v0','age-fusion-ranker-dev-v0',107904),('validation','content-ranker-confirmation-v0','age-fusion-confirmation-v0',114855)]:
        paths=[ROOT/'runs'/old/'predictions.jsonl',ROOT/'runs'/new/'predictions.jsonl'];input_hashes[label]=[file_hash(p) for p in paths]
        original={r['request_id']:r for r in map(json.loads,paths[0].open())};actual={r['request_id']:r for r in map(json.loads,paths[1].open())}
        assert original.keys()==actual.keys() and len(original)==n
        counts[label]=defaultdict(lambda:defaultdict(Counter))
        for rid,r in actual.items():
            o=original[rid]
            for k in ('request_id','user_id','request_timestamp','target_id','target_timestamp','target_available','history_length'):assert o[k]==r[k]
            assert rid not in refs
            refs[rid]=(label,r,o)
    seen=set();rows=conn.execute('SELECT user,timestamp,ordinal,article FROM clicks WHERE role="train" AND timestamp<=1507980021008 ORDER BY user,timestamp,ordinal')
    for user,events in groupby(rows,lambda r:r[0]):
        history=deque(maxlen=20);length=0
        for t,batch in groupby(events,lambda r:r[1]):
            batch=list(batch)
            for _,_,rid,article in batch:
                if rid not in refs:continue
                label,r,o=refs[rid];seen.add(rid)
                assert (r['user_id'],r['request_timestamp'],r['history_length'])==(user,t,length+1)
                h=(list(history)+[article])[-20:];target=r['target_id'];age=t-created[target]
                axes=dict(all='all',gap=gap_bin(r['target_timestamp']-t),age='unavailable' if age<0 else 'le6h' if age<=21600000 else 'over6h',
                          repeat='last3' if target in h[-3:] else 'last20_only' if target in h else 'outside20',history='le5' if length+1<=5 else 'gt5')
                axes['gap_x_repeat']=axes['gap']+'__'+axes['repeat']
                a=int(target in o['article_ids']['base15'][:5]);b=int(target in r['article_ids']['base15'][:5]);c=int(target in r['article_ids']['content18'][:5])
                for axis,value in axes.items():counts[label][axis][value].update(requests=1,original_hits=a,fused15_hits=b,fused18_hits=c,gains=int(c and not a),losses=int(a and not c))
            history.extend(a for _,_,_,a in batch);length+=len(batch)
    assert seen==refs.keys()
    conn.close();assert file_hash(ROOT/'runs/recovered-audit/audit.sqlite')==before
    result=dict(scope='explicit post-hoc description of already-explored windows; no training, final scoring or causal claims',requests_verified=len(seen),source_database_sha256=before,input_predictions_sha256=input_hashes,
                windows={w:{a:{k:dict(c) for k,c in vals.items()} for a,vals in axes.items()} for w,axes in counts.items()},
                decomposition={axis:symmetric_decomposition(counts['development'][axis],counts['validation'][axis]) for axis in ['gap','age','repeat','history','gap_x_repeat']})
    write_json(out/'audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['windows','input_predictions_sha256']},indent=2))
if __name__=='__main__':main()
