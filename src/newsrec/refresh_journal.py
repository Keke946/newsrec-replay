"""Versioned single-owner batch journal with pinned historical model choices."""
import fcntl,json,platform,sqlite3,threading
from pathlib import Path
from .serving import ContractError,integer
from .data import file_hash

class RefreshJournalError(ContractError):pass

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def identity_for(registry,warmup_through):
    import numpy,lightgbm
    here=Path(__file__).parent
    return dict(registry_sha256=registry.version,database_sha256=registry.info['database_sha256'],warmup_through=warmup_through,
                source_sha256={n:file_hash(here/n) for n in ['refresh_journal.py','append_refresh_journal.py','refresh_replay.py','time_ranker.py','baseline.py']},
                python=platform.python_version(),numpy=numpy.__version__,lightgbm=lightgbm.__version__)

class RefreshJournal:
    def __init__(self,path,factory,identity):
        self.lock=threading.RLock();self.path=Path(path);self.factory=factory;self.identity=identity;self.closed=False;self.connection=None
        self.path.parent.mkdir(parents=True,exist_ok=True);self.owner=Path(str(self.path)+'.lock').open('a+b')
        try:fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:self.owner.close();raise RefreshJournalError('Journal already owned') from e
        try:
            self.connection=sqlite3.connect(self.path,check_same_thread=False)
            tables={r[0] for r in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables-{'refresh_identity','refresh_batches'}:raise RefreshJournalError('Refusing unrelated/old-format database')
            expected=canonical(dict(format='refresh-batch-journal-v2',identity=identity))
            if 'refresh_identity' in tables:
                prior=self.connection.execute('SELECT payload FROM refresh_identity WHERE id=1').fetchone()
                if prior and prior[0]!=expected:raise RefreshJournalError('Journal format or identity differs; no implicit migration')
            if self.connection.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise RefreshJournalError('Journal integrity check failed')
            self.connection.execute('PRAGMA journal_mode=DELETE');self.connection.execute('PRAGMA synchronous=FULL')
            with self.connection:
                self.connection.execute('CREATE TABLE IF NOT EXISTS refresh_identity(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL)')
                self.connection.execute('CREATE TABLE IF NOT EXISTS refresh_batches(id INTEGER PRIMARY KEY,timestamp INTEGER UNIQUE NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL,switch TEXT NOT NULL)')
                expected=canonical(dict(format='refresh-batch-journal-v2',identity=identity));old=self.connection.execute('SELECT payload FROM refresh_identity WHERE id=1').fetchone()
                if old is None:
                    if self.connection.execute('SELECT count(*) FROM refresh_batches').fetchone()[0]:raise RefreshJournalError('Actions lack identity')
                    self.connection.execute('INSERT INTO refresh_identity VALUES(1,?)',(expected,))
                elif old[0]!=expected:raise RefreshJournalError('Registry/data/source/environment identity differs')
            self._restore()
        except BaseException:self.close();raise

    def _restore(self):
        rebuilt=self.factory()
        if rebuilt.registry.version!=self.identity['registry_sha256'] or rebuilt.observed_through!=self.identity['warmup_through']:raise RefreshJournalError('Factory identity/warmup differs')
        previous=None;count=0
        for _,timestamp,payload,result,switch in self._rows():
            request=json.loads(payload);expected=json.loads(result)
            if request['timestamp']!=timestamp:raise RefreshJournalError('Journal timestamp differs')
            actual=rebuilt.process_batch(**request,_pinned_cutoff=expected['model_cutoff'])
            transition=dict(from_cutoff=previous,to_cutoff=actual['model_cutoff'])
            if canonical(actual)!=result or canonical(transition)!=switch:raise RefreshJournalError('Committed replay or model switch differs')
            previous=actual['model_cutoff'];count+=1
        self.engine=rebuilt;self.replayed_batches=count;self.last_cutoff=previous

    def _rows(self):return self.connection.execute('SELECT * FROM refresh_batches ORDER BY id')
    def _lookup(self,timestamp):return self.connection.execute('SELECT payload,result FROM refresh_batches WHERE timestamp=?',(timestamp,)).fetchone()
    def _rollback(self):self.connection.rollback()

    def _persist(self,timestamp,payload,result,switch):
        with self.connection:
            self.connection.execute('INSERT INTO refresh_batches(timestamp,payload,result,switch) VALUES(?,?,?,?)',(timestamp,payload,result,switch))

    def process_batch(self,timestamp,events,score_event_ids=None):
        with self.lock:
            if self.closed:raise RefreshJournalError('Journal closed')
            integer(timestamp,'timestamp')
            payload=canonical(dict(timestamp=timestamp,events=events,score_event_ids=score_event_ids))
            try:
                prior=self._lookup(timestamp)
                if prior:
                    if prior[0]!=payload:raise ContractError('Committed timestamp batch has different contents')
                    return json.loads(prior[1])
                result=self.engine.process_batch(**json.loads(payload));switch=dict(from_cutoff=self.last_cutoff,to_cutoff=result['model_cutoff'])
                self._persist(timestamp,payload,canonical(result),canonical(switch));self.last_cutoff=result['model_cutoff']
                return result
            except Exception as original:
                try:self._rollback();self._restore()
                except Exception as recovery:self.close();raise RefreshJournalError('Recovery failed; closed') from recovery
                if isinstance(original,(sqlite3.Error,OSError)):raise RefreshJournalError('Write uncertain; committed state recovered, retry exact batch') from original
                raise

    @property
    def observed_through(self):return self.engine.observed_through
    @property
    def registry(self):return self.engine.registry
    def close(self):
        with self.lock:
            self.closed=True
            if self.connection is not None:self.connection.close();self.connection=None
            if not self.owner.closed:self.owner.close()
