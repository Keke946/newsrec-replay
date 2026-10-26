"""Self-contained synthetic software demo; no competition data or quality claim."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.time_ranker import CONFIG,FEATURES
from newsrec.data import file_hash,write_json
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine,metadata_hash,encoder_hashes
from newsrec.refresh_journal import identity_for
from newsrec.state_snapshot import SnapshotRefreshJournal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026);x=rng.normal(size=(1000,15)).astype('float32');labels=np.tile(np.array([1]+[0]*19),50);c=CONFIG
    model=lgb.LGBMRanker(objective='lambdarank',n_estimators=c['n_estimators'],num_leaves=c['num_leaves'],learning_rate=c['learning_rate'],min_child_samples=c['min_child_samples'],reg_lambda=c['reg_lambda'],lambdarank_truncation_level=c['lambdarank_truncation_level'],random_state=c['seed'],n_jobs=1,deterministic=True,force_col_wise=True,verbosity=-1)
    entries=[];start=100000;activate=start+21600000+60000
    for name,y,cut,t in [('initial',labels,start,start),('refresh',np.roll(labels,1),start+21600000-1,activate)]:
        model.fit(x,y,group=[20]*50,feature_name=FEATURES);path=out/(name+'.txt');model.booster_.save_model(str(path));entries.append(dict(cutoff=cut,activate_ms=t,max_label_time=cut-1,file=path.name,sha256=file_hash(path)))
    metadata={i:dict(category=i%2,created=1,words=100) for i in [1,2,3]}
    write_json(out/'registry.json',dict(schema='precomputed-refresh-replay-v1',scope='synthetic demonstration, not benchmark',start_ms=start,database_sha256='synthetic-demo',metadata_sha256=metadata_hash(metadata),encoder_sha256=encoder_hashes(),feature_names=FEATURES,config=CONFIG,models=entries))
    def empty():return RefreshReplayEngine(metadata,ModelRegistry(out/'registry.json','synthetic-demo'))
    def factory():
        e=empty();e.process_batch(start-1,[dict(event_id=1,user_id=20,article_id=1)],[]);return e
    identity=identity_for(ModelRegistry(out/'registry.json','synthetic-demo'),start-1)
    def journal():return SnapshotRefreshJournal(out/'batches.jsonl',factory,identity,empty_factory=empty,snapshot_path=out/'state.gz')
    j=journal();before=j.process_batch(start+1,[dict(event_id=2,user_id=20,article_id=2)]);after=j.process_batch(activate,[dict(event_id=3,user_id=20,article_id=3)]);snapshot=j.checkpoint();j.close()
    j=journal();assert j.snapshot_status=='loaded';retry=j.process_batch(activate,[dict(event_id=3,user_id=20,article_id=3)]);assert retry==after
    result=dict(scope='synthetic software demo only; no accuracy measurement',initial_cutoff=before['model_cutoff'],refreshed_cutoff=after['model_cutoff'],snapshot_loaded=True,exact_retry_after_restart=True,checkpoint=snapshot,article_ids=retry['predictions'][0]['article_ids']);j.close();write_json(out/'demo.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
