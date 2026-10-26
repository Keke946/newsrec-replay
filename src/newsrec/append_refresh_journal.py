"""Single-owner append-only, hash-linked batch journal; no SQLite writes."""
import fcntl,hashlib,json,os,threading,uuid
from pathlib import Path
from .refresh_journal import RefreshJournal,RefreshJournalError,canonical

class AppendRefreshJournal(RefreshJournal):
    def __init__(self,path,factory,identity):
        self.lock=threading.RLock();self.path=Path(path);self.factory=factory;self.identity=identity;self.closed=False;self.connection=None;self.file=None
        self.path.parent.mkdir(parents=True,exist_ok=True);self.owner=Path(str(self.path)+'.lock').open('a+b')
        try:fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e:self.owner.close();raise RefreshJournalError('Journal already owned') from e
        try:
            if self.path.exists():self.file=self.path.open('r+b',buffering=0)
            else:
                self.file=self.path.open('x+b',buffering=0)
                header=canonical(dict(format='refresh-append-journal-v1',identity=identity)).encode()+b'\n'
                self._write_all(header);os.fsync(self.file.fileno())
            self._scan();self._restore()
        except BaseException:self.close();raise

    def _write_all(self,data):
        view=memoryview(data)
        while view:
            n=self.file.write(view)
            if not n:raise OSError('Incomplete append')
            view=view[n:]

    def _scan(self):
        self.file.seek(0);header=self.file.readline()
        expected=canonical(dict(format='refresh-append-journal-v1',identity=self.identity)).encode()+b'\n'
        if header!=expected:raise RefreshJournalError('Append format/identity differs; no migration')
        checksum=hashlib.sha256(header).hexdigest();records=[];index={};last_end=self.file.tell()
        while True:
            line=self.file.readline()
            if not line:break
            if not line.endswith(b'\n'):
                # Unterminated bytes cannot have been acknowledged by this writer.
                tail=self.path.with_name(self.path.name+'.uncommitted-'+uuid.uuid4().hex)
                tail.write_bytes(line);self.file.truncate(last_end);os.fsync(self.file.fileno());break
            try:
                record=json.loads(line);body=record['body']
                digest=hashlib.sha256((checksum+canonical(body)).encode()).hexdigest()
                if record['previous']!=checksum or record['sha256']!=digest:raise ValueError('Hash chain differs')
                n,t,p,r,s=body
                if n!=len(records)+1 or t in index:raise ValueError('Duplicate/order differs')
            except Exception as e:raise RefreshJournalError('Complete journal record corrupt') from e
            records.append(tuple(body));index[t]=(p,r);checksum=digest;last_end=self.file.tell()
        self.records=records;self.index=index;self.checksum=checksum;self.file.seek(0,2)

    def _rows(self):return iter(self.records)
    def _lookup(self,timestamp):return self.index.get(timestamp)
    def _rollback(self):
        os.fsync(self.file.fileno());self._scan()
    def _persist(self,timestamp,payload,result,switch):
        body=[len(self.records)+1,timestamp,payload,result,switch]
        digest=hashlib.sha256((self.checksum+canonical(body)).encode()).hexdigest()
        line=canonical(dict(body=body,previous=self.checksum,sha256=digest)).encode()+b'\n'
        self.file.seek(0,2);self._write_all(line);os.fsync(self.file.fileno())
        self.records.append(tuple(body));self.index[timestamp]=(payload,result);self.checksum=digest
    def close(self):
        with self.lock:
            self.closed=True
            if self.file is not None:self.file.close();self.file=None
            if not self.owner.closed:self.owner.close()
