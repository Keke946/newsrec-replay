"""Launch explicitly selected in-memory precomputed replay on loopback."""
import argparse,sqlite3,sys
from contextlib import closing
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash
from newsrec.refresh_replay import ModelRegistry,RefreshReplayEngine,prime_refresh
from newsrec.refresh_api import create_refresh_app

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--database',type=Path,required=True);p.add_argument('--registry',type=Path,required=True);p.add_argument('--port',type=int,default=8001);p.add_argument('--journal',type=Path);p.add_argument('--snapshot',type=Path);a=p.parse_args()
    if a.snapshot and not a.journal:p.error('--snapshot requires --journal')
    digest=file_hash(a.database);registry=ModelRegistry(a.registry,digest)
    def empty_factory():
        if file_hash(a.database)!=digest:raise ValueError("Source database changed")
        return RefreshReplayEngine(metadata,ModelRegistry(a.registry,digest))
    with closing(sqlite3.connect(f"file:{a.database.resolve()}?mode=ro",uri=True)) as c:
        metadata={i:dict(category=x,created=t,words=w) for i,x,t,w in c.execute("SELECT * FROM articles")}
    def factory():
        if file_hash(a.database)!=digest:raise ValueError('Source database changed')
        with closing(sqlite3.connect(f'file:{a.database.resolve()}?mode=ro',uri=True)) as conn:
            metadata={i:dict(category=c,created=t,words=w) for i,c,t,w in conn.execute('SELECT * FROM articles')}
            e=RefreshReplayEngine(metadata,ModelRegistry(a.registry,digest));prime_refresh(e,conn,registry.start)
            return e
    if a.journal:
        from newsrec.refresh_journal import identity_for
        from newsrec.append_refresh_journal import AppendRefreshJournal
        with closing(sqlite3.connect(f'file:{a.database.resolve()}?mode=ro',uri=True)) as conn:
            through=conn.execute('SELECT max(timestamp) FROM clicks WHERE role="train" AND timestamp<=?',(registry.start,)).fetchone()[0]
        if a.snapshot:
            from newsrec.state_snapshot import SnapshotRefreshJournal
            engine=SnapshotRefreshJournal(a.journal,factory,identity_for(registry,through),empty_factory=empty_factory,snapshot_path=a.snapshot)
        else:engine=AppendRefreshJournal(a.journal,factory,identity_for(registry,through))
    else:engine=factory()
    import uvicorn
    try:uvicorn.run(create_refresh_app(engine),host='127.0.0.1',port=a.port,workers=1)
    finally:
        if a.journal:
            try:
                if a.snapshot:engine.checkpoint()
            finally:engine.close()
if __name__=='__main__':main()
