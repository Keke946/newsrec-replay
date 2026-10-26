"""Nonlearned supplied-vector control on exactly the neural run's saved cohort."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from newsrec.data import file_hash, source_hash, write_json
from newsrec.two_tower import VectorStore, evaluate


class ContentCosine(torch.nn.Module):
    def users(self, vectors):
        return torch.nn.functional.normalize(vectors, dim=-1)

    def items(self, vectors):
        return torch.nn.functional.normalize(vectors, dim=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--neural-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    original=json.loads((args.neural_run/'manifest.json').read_text())
    if original['status']!='succeeded':
        raise ValueError('Need a successful neural cohort run')
    info=json.loads((args.neural_run/'validation/examples.json').read_text())
    protocol=original['protocol']
    if info['training'] or info['end']!=protocol['validation_end_ms'] or info['request_after']!=protocol['train_end_ms']:
        raise ValueError('Unexpected validation feature contract')
    if file_hash(args.database)!=original['database_sha256']:
        raise ValueError('Different audit database')
    for name,checksum in original['input_hashes'].items():
        if file_hash(args.data_dir/name)!=checksum:
            raise ValueError(f'Different vector input: {name}')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','source_sha256':source_hash(),'torch':torch.__version__,
              'protocol':protocol,'method':'nonlearned content cosine',
              'features':'same last-three provided-vector pooling as two-tower',
              'config':{'threads':4,'dimension':250,'batch_size':64,'k':10},
              'limitations':original['limitations'],
              'query_sha256':file_hash(args.neural_run/'validation/queries.npy'),
              'records_sha256':file_hash(args.neural_run/'validation/records.npy')}
    write_json(args.output/'manifest.json',manifest)
    started=time.perf_counter()
    connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    try:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        store=VectorStore(args.data_dir,connection)
        n=info['examples']
        queries=np.load(args.neural_run/'validation/queries.npy',mmap_mode='r')[:n]
        records=np.load(args.neural_run/'validation/records.npy',mmap_mode='r')[:n]
        metrics=evaluate(ContentCosine(),store,queries,records,args.output,dimension=250,
                         model_name='content_cosine')
        write_json(args.output/'metrics.json',metrics)
        manifest['status']='succeeded'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        connection.close()
        manifest['wall_seconds']=time.perf_counter()-started
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':
    main()
