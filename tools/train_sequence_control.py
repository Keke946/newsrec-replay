"""Exactly two fixed user encoders, shared data/initial item tower, full validation."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,source_hash,write_json


def main():
    import torch
    import numpy as np
    from newsrec.two_tower import VectorStore,train,evaluate
    from newsrec.sequence import SequenceTower,SequenceProvider,build_sequences
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--reference-records',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text())
    manifest=dict(status='running',source_sha256=source_hash(),script_sha256=file_hash(Path(__file__)),protocol=protocol,
                  config=dict(user_encoders=['pool','gru'],max_length=20,dimension=64,seed=2026,epochs=1,batch_size=256,
                              temperature=.1,lr=.001,threads=4,pool_decay=.7),
                  limitations=['provided embeddings have unknown time/training provenance',
                               'parameter counts differ; fixed epochs are not equal total compute',
                               'no final labels, ANN, freshness features or ranker fusion in this experiment'])
    started=time.perf_counter();connection=None
    write_json(args.output/'manifest.json',manifest)
    try:
        for name,key in [('train_click_log.csv','train_csv_sha256'),('articles.csv','articles_csv_sha256')]:
            if file_hash(args.data_dir/name)!=protocol[key]:raise ValueError('Unexpected source bytes')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
        connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
        store=VectorStore(args.data_dir,connection)
        manifest['vector_inputs']={name:file_hash(args.data_dir/name) for name in ['article_vectors.f32','article_ids.i64']}
        training=args.output/'train';validation=args.output/'validation';training.mkdir();validation.mkdir()
        train_ids,train_records,train_info=build_sequences(connection,store,training,protocol['train_end_ms'],training=True)
        valid_ids,valid_records,valid_info=build_sequences(connection,store,validation,protocol['validation_end_ms'],
                                                         request_after=protocol['train_end_ms'])
        reference=np.load(args.reference_records,mmap_mode='r')[:len(valid_records)]
        if len(valid_records)!=114855 or not np.array_equal(valid_records,reference):
            raise ValueError('Sequence/pooled validation records differ')
        if len(train_records)!=509468:raise ValueError('Training cohort differs from original two-tower')
        manifest.update(training=train_info,validation=valid_info,validation_records_exactly_equal=True,
                        training_indices_sha256=file_hash(training/'indices.npy'),
                        training_records_sha256=file_hash(training/'records.npy'),
                        validation_records_sha256=file_hash(validation/'records.npy'),models={})
        write_json(args.output/'manifest.json',manifest)
        initial_hash=None
        for kind in ['pool','gru']:
            directory=args.output/kind;directory.mkdir()
            torch.manual_seed(2026)
            model=SequenceTower(kind)
            current_hash=model.item_initialization_hash()
            if initial_hash is not None and current_hash!=initial_hash:raise ValueError('Item initialization differs')
            initial_hash=current_hash
            info=dict(item_initialization_sha256=current_hash,parameters=sum(value.numel() for value in model.parameters()))
            clock=time.perf_counter()
            info['training_log']=train(model,store,SequenceProvider(train_ids,store),train_records)
            info['training_seconds']=time.perf_counter()-clock
            torch.save(model.state_dict(),directory/'model_state.pt')
            clock=time.perf_counter()
            metrics=evaluate(model,store,SequenceProvider(valid_ids,store),valid_records,directory,
                             model_name='sequence_'+kind+'_20')
            info['inference_seconds']=time.perf_counter()-clock
            info['model_sha256']=file_hash(directory/'model_state.pt')
            write_json(directory/'metrics.json',metrics);write_json(directory/'model_info.json',info)
            manifest['models'][kind]=info
            write_json(args.output/'manifest.json',manifest)
        manifest['status']='succeeded'
    except Exception as error:
        manifest['status']='failed';manifest['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if connection:connection.close()
        manifest['wall_seconds']=time.perf_counter()-started
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
