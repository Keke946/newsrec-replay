"""One fixed in-batch vs budget-matched mixed-source contrast, internal dev only."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.two_tower import VectorStore,TwoTower,build_examples,train,evaluate
from newsrec.mixed_negatives import train_mixed,state_hash
from newsrec.data import source_hash,file_hash,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=json.loads(args.protocol.read_text())
    if p['validation_end_ms']!=1507757054762 or p['train_end_ms']!=1507647588945:
        raise ValueError('Only frozen internal development window authorized')
    args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    conn=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    m=dict(status='running',source_sha256=source_hash(),protocol=p,torch=torch.__version__,
           database_sha256=file_hash(args.database),models={},old_validation_scored=False,final_region_scored=False)
    write_json(args.output/'manifest.json',m)
    try:
        for name,key in [('article_vectors.f32','vector_sha256'),('article_ids.i64','ids_sha256')]:
            if file_hash(args.data_dir/name)!=p[key]:raise ValueError('Vector source differs')
        store=VectorStore(args.data_dir,conn)
        td=args.output/'train';vd=args.output/'development';td.mkdir();vd.mkdir()
        tq,tr,ti=build_examples(conn,store,td,p['train_end_ms'],training=True)
        vq,vr,vi=build_examples(conn,store,vd,p['validation_end_ms'],request_after=p['train_end_ms'])
        if vi['examples']!=107904:raise ValueError('Internal development cohort differs')
        m.update(training=ti,development=vi,
                 shared_record_sha256=file_hash(td/'records.npy'),shared_query_sha256=file_hash(td/'queries.npy'))
        for name in ['inbatch','mixed']:
            torch.manual_seed(2026);model=TwoTower();initial=state_hash(model)
            out=args.output/name;out.mkdir();clock=time.perf_counter()
            log=train(model,store,tq,tr) if name=='inbatch' else train_mixed(model,store,tq,tr,p['train_end_ms'])
            seconds=time.perf_counter()-clock
            torch.save(model.state_dict(),out/'model_state.pt')
            info=dict(initial_state_sha256=initial,trained_state_sha256=state_hash(model),training_seconds=seconds,training_log=log)
            write_json(out/'model_info.json',info);m['models'][name]=info
            write_json(args.output/'manifest.json',m)
            metrics=evaluate(model,store,vq,vr,out,model_name=name)
            write_json(out/'metrics.json',metrics)
        if m['models']['inbatch']['initial_state_sha256']!=m['models']['mixed']['initial_state_sha256']:
            raise ValueError('Different initial weights')
        m['status']='succeeded'
    except Exception as error:m.update(status='failed',error=repr(error));raise
    finally:
        conn.close();m['elapsed_seconds']=time.perf_counter()-start;write_json(args.output/'manifest.json',m)


if __name__=='__main__':main()
