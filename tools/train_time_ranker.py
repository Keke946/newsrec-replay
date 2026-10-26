import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,source_hash,write_json
from newsrec.time_ranker import run,CONFIG,FEATURES


def main():
    import lightgbm
    parser=argparse.ArgumentParser(description='Fixed first time-aware LambdaRank experiment')
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ablation',choices=['all','no-user-features'],default='all')
    parser.add_argument('--candidate-policy',choices=['base','wide-trend','catalog-fresh'],default='base')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text())
    config=dict(CONFIG)
    if args.candidate_policy=='wide-trend':config['candidate_trend_k']=60
    elif args.candidate_policy=='catalog-fresh':
        config.update(fresh_slots=10,fresh_max_age_ms=6*60*60*1000)
    if args.ablation=='no-user-features':
        config['feature_columns']=[0,1,2,3,4,5,6,12]
    manifest={'status':'running','source_sha256':source_hash(),'script_sha256':file_hash(Path(__file__)),
              'lightgbm':lightgbm.__version__,'config':config,'ablation':args.ablation,'candidate_policy':args.candidate_policy,'features':FEATURES,'protocol':protocol,
              'selection':('fixed full configuration before its first validation score' if args.ablation=='all'
                           else 'single explanatory feature ablation declared after first full-model result, before its own score'),
              'no_search':'no early stopping or hyperparameter search',
              'limitations':['same-pool controls still differ in supplied features and learning',
                             'training naturally retrieved, available positive groups only; selection bias disclosed',
                             'no final-region scores, embedding features, ANN or deployed service']}
    clock=time.perf_counter();connection=None
    try:
        for name,key in [('train_click_log.csv','train_csv_sha256'),('articles.csv','articles_csv_sha256')]:
            if file_hash(args.data_dir/name)!=protocol[key]:raise ValueError('Unexpected input bytes')
        manifest['database_sha256']=file_hash(args.database)
        write_json(args.output/'manifest.json',manifest)
        connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
        result=run(connection,args.output,protocol,config)
        write_json(args.output/'metrics.json',result)
        manifest['status']='succeeded'
    except Exception as exc:
        manifest['status']='failed';manifest['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if connection:connection.close()
        manifest['wall_seconds']=time.perf_counter()-clock
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
