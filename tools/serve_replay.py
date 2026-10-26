"""Start a local-only, one-worker replay prototype with an existing trained model."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash
from newsrec.serving import ReplayEngine,prime
from newsrec.api import create_app


def journal_identity(database,model_run):
    from importlib.metadata import version
    import platform
    import sqlite3
    from newsrec.data import source_hash
    manifest=json.loads((model_run/'manifest.json').read_text())
    return dict(database=file_hash(database),model=file_hash(model_run/'ranker.txt'),
                manifest=file_hash(model_run/'manifest.json'),source=source_hash(),
                config=manifest['config'],cutoff=manifest['protocol']['train_end_ms'],
                python=platform.python_version(),sqlite=sqlite3.sqlite_version,
                lightgbm=version('lightgbm'),numpy=version('numpy'))


def load_engine(database,model_run):
    import lightgbm as lgb
    manifest=json.loads((model_run/'manifest.json').read_text())
    if manifest['status']!='succeeded':raise ValueError('Model run did not succeed')
    if file_hash(database)!=manifest['database_sha256']:raise ValueError('Different source database')
    connection=sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True)
    try:
        metadata={item:dict(category=category,created=created,words=words)
                  for item,category,created,words in connection.execute('SELECT * FROM articles')}
        model=lgb.Booster(model_file=str(model_run/'ranker.txt'))
        engine=ReplayEngine(metadata,model,manifest['config'],version=file_hash(model_run/'ranker.txt')[:16])
        count=prime(engine,connection,manifest['protocol']['train_end_ms'])
        print(f'Loaded model; replayed {count} historical events.',flush=True)
        return engine
    finally:connection.close()


def load_journal_engine(path,database,model_run):
    from newsrec.journal import JournalEngine,JournalError
    if path.resolve()==database.resolve():raise JournalError('Journal cannot be the source database')
    identity=journal_identity(database,model_run)
    def factory():
        if journal_identity(database,model_run)!=identity:
            raise JournalError('Source artifacts or runtime changed during recovery')
        return load_engine(database,model_run)
    return JournalEngine(path,factory,identity)


def main():
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--model-run',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--journal',type=Path,help='Opt-in ordered durable action journal, exclusive one-process owner')
    args=parser.parse_args()
    if args.journal:
        engine=load_journal_engine(args.journal,args.database,args.model_run)
    else:
        engine=load_engine(args.database,args.model_run)
    try:uvicorn.run(create_app(engine),host='127.0.0.1',port=args.port,workers=1)
    finally:
        if args.journal:engine.close()


if __name__=='__main__':main()
