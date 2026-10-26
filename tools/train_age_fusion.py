"""Run a paired content experiment; validation requires explicit confirmation mode."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.content_ranker import ContentVectors,run_pair,validate_protocol
from newsrec.age_fusion import AgeCandidates
from newsrec.data import file_hash,source_hash,write_json
from newsrec.time_ranker import CONFIG


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirm-validation',action='store_true',help='Explicitly allow the exact original validation window; never final')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text());validate_protocol(protocol,args.confirm_validation);start=time.perf_counter()
    import lightgbm,numpy,platform
    manifest=dict(status='running',source_sha256=source_hash(),protocol=protocol,config=CONFIG,
                  database_sha256=file_hash(args.database),python=platform.python_version(),lightgbm=lightgbm.__version__,numpy=numpy.__version__)
    write_json(args.output/'manifest.json',manifest)
    connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
    try:
        metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in connection.execute('SELECT * FROM articles')}
        vectors=ContentVectors.load(args.data_dir,metadata,protocol)
        result=run_pair(connection,vectors,metadata,args.output,protocol,allow_confirmation=args.confirm_validation,candidate_factory=lambda m,c:AgeCandidates(m,c,vectors))
        manifest.update(status='succeeded',elapsed_seconds=time.perf_counter()-start,
                        model_sha256={n:file_hash(args.output/(n+'.txt')) for n in ('base15','content18')})
        print(json.dumps({n:m['all'] for n,m in result['models'].items()},indent=2))
    except Exception as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:
        connection.close();write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
