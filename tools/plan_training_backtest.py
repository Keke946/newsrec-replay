"""Plan expanding-origin folds strictly inside original training data; no fit."""
import json,sqlite3,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import write_json
LIMIT=1507757054762

def expanding_folds(times,limit=LIMIT):
    if len(times)<10 or any(b<a for a,b in zip(times,times[1:])) or times[-1]>limit:raise ValueError('Invalid protected training timeline')
    points=[int(times[max(0,int(len(times)*p)-1)]) for p in [.4,.6,.8]]+[limit]
    if len(set(points))!=4 or any(b<=a for a,b in zip(points,points[1:])):raise ValueError('Degenerate timestamp boundaries')
    return [dict(fold=k+1,fit_end_ms=a,target_end_ms=b,request_after_ms=a,requests_must_end_before_or_at=b) for k,(a,b) in enumerate(zip(points,points[1:]))]

def main():
    root=Path(__file__).resolve().parents[1];conn=sqlite3.connect(f'file:{root}/runs/recovered-audit/audit.sqlite?mode=ro',uri=True)
    times=[t for (t,) in conn.execute('SELECT timestamp FROM clicks WHERE role="train" AND timestamp<=? ORDER BY timestamp,ordinal',(LIMIT,))];conn.close()
    assert len(times)==778836
    out=dict(protocol='training-only-expanding-origin-backtest-v1',source_events=len(times),old_validation_start_ms=LIMIT,final_region_scored=False,training_executed=False,
             folds=expanding_folds(times),rules=['no boundary chosen from performance','all targets including unavailable retained','fit labels only when target_timestamp<=fit_end','own-current-click then reveal other tied events after requests','provided embeddings provenance remains unknown','these training regions were already explored; not independent final evaluation'],
             next_comparison='freeze original base15 versus fixed six-hour scheduled refits before execution; no content/slot rescue of failed fusion')
    write_json(root/'protocols/training_backtest_v1.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
