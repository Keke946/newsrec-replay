"""Versioned JSON/gzip snapshots bound to a validated append-log prefix."""
from collections import Counter,defaultdict,deque
import gzip,hashlib,json,os,time,uuid
from pathlib import Path
from .append_refresh_journal import AppendRefreshJournal
from .refresh_journal import RefreshJournalError,canonical
from .data import file_hash


def codec_hash():return file_hash(Path(__file__))

def capture(engine):
    if engine.closed or engine.state.fresh is not None:raise RefreshJournalError('Unsupported/closed snapshot state')
    st=engine.state
    def window(w):return dict(events=list(w.events),counts=dict(w.counts))
    return dict(observed_through=engine.observed_through,histories={u:list(h) for u,h in engine.histories.items()},lengths=dict(engine.lengths),
                state=dict(last_query_timestamp=st.last_query_timestamp,seen=sorted(st.seen),hour=window(st.hour),quarter=window(st.quarter),
                           trend=dict(**window(st.trend),cached=st.trend.cached,bucket=st.trend.bucket,last_observed_timestamp=st.trend.last_observed_timestamp)),
                registry_clock=engine.registry.clock,current_cutoff=engine.registry.current['cutoff'] if engine.registry.current else None,switches=engine.switches)


def restore_state(engine,data):
    observed=data['observed_through'];st=engine.state
    if observed is None or type(observed) is not int:raise RefreshJournalError('Missing snapshot event boundary')
    if data['state']['last_query_timestamp']!=observed:raise RefreshJournalError('Snapshot clock differs')
    def window(w,value):
        events=[tuple(e) for e in value['events']];counts=Counter({int(i):n for i,n in value['counts'].items()})
        if any(type(t) is not int or t>observed or i not in engine.metadata for t,i in events):raise RefreshJournalError('Invalid snapshot window')
        if any(a[0]>b[0] for a,b in zip(events,events[1:])) or Counter(i for _,i in events)!=counts:raise RefreshJournalError('Snapshot window/count mismatch')
        w.events=deque(events);w.counts=counts
    for name in ['hour','quarter']:window(getattr(st,name),data['state'][name])
    trend=data['state']['trend'];window(st.trend,trend)
    if trend['last_observed_timestamp']!=observed or trend['bucket']!=observed//st.trend.refresh_ms:raise RefreshJournalError('Snapshot trend clock differs')
    cached=trend['cached']
    if len(cached)!=len(set(cached)) or any(i not in engine.metadata for i in cached):raise RefreshJournalError('Bad snapshot cached ranking')
    st.trend.cached=cached;st.trend.bucket=trend['bucket'];st.trend.last_observed_timestamp=observed
    st.last_query_timestamp=observed;st.seen=set(data['state']['seen'])
    if not st.seen<=engine.metadata.keys():raise RefreshJournalError('Unknown seen item')
    histories=defaultdict(lambda:deque(maxlen=20));lengths=Counter({int(u):n for u,n in data['lengths'].items()})
    for u,h in data['histories'].items():
        u=int(u)
        if len(h)>20 or lengths[u]<len(h) or any(i not in engine.metadata or t>observed for i,t in h):raise RefreshJournalError('Invalid snapshot history')
        histories[u]=deque([tuple(v) for v in h],maxlen=20)
    if set(lengths)!=set(histories) or any(type(v) is not int or v<=0 for v in lengths.values()):raise RefreshJournalError('History/length keys differ')
    engine.histories=histories;engine.lengths=lengths;engine.observed_through=observed
    cut,clock=data['current_cutoff'],data['registry_clock']
    if cut is None:
        if clock is not None or observed>engine.registry.start:raise RefreshJournalError('Snapshot missing active model')
    else:
        if clock!=observed:raise RefreshJournalError('Snapshot model clock differs')
        engine.registry.select_pinned(clock,cut)
    switches=data['switches']
    if cut is not None and (not switches or switches[-1]['cutoff']!=cut):raise RefreshJournalError('Switch state differs')
    engine.switches=switches
    return engine


class SnapshotRefreshJournal(AppendRefreshJournal):
    def __init__(self,path,factory,identity,*,empty_factory,snapshot_path=None):
        self.empty_factory=empty_factory;self.snapshot_path=Path(snapshot_path) if snapshot_path else None
        self.snapshot_status='none';self.snapshot_prefix=0
        super().__init__(path,factory,identity)

    def prefix_hash(self,count):
        digest=hashlib.sha256((canonical(dict(format='refresh-append-journal-v1',identity=self.identity))+'\n').encode()).hexdigest()
        for row in self.records[:count]:digest=hashlib.sha256((digest+canonical(list(row))).encode()).hexdigest()
        return digest

    def _restore(self):
        tick=time.perf_counter();self.snapshot_prefix=0;self.snapshot_status='none'
        if self.snapshot_path and self.snapshot_path.exists():
            try:
                wrapper=json.loads(gzip.decompress(self.snapshot_path.read_bytes()));body=wrapper['body']
                if wrapper['sha256']!=hashlib.sha256(canonical(body).encode()).hexdigest():raise ValueError('Snapshot checksum differs')
                if body['format']!='refresh-state-snapshot-v1' or body['identity']!=self.identity or body['codec_sha256']!=codec_hash():raise ValueError('Snapshot identity/codec differs')
                n=body['prefix_count']
                if type(n) is not int or not 0<=n<=len(self.records) or body['prefix_sha256']!=self.prefix_hash(n):raise ValueError('Snapshot log prefix differs')
                rebuilt=self.empty_factory()
                if rebuilt.registry.version!=self.identity['registry_sha256']:raise ValueError('Snapshot factory registry differs')
                restore_state(rebuilt,body['engine']);previous=body['last_cutoff']
                if previous!=(rebuilt.registry.current['cutoff'] if rebuilt.registry.current else None):raise ValueError('Snapshot active cutoff differs')
                expected_time=self.records[n-1][1] if n else self.identity['warmup_through']
                if rebuilt.observed_through!=expected_time:raise ValueError('Snapshot event boundary differs')
            except Exception as error:
                self.snapshot_status='rejected: '+type(error).__name__+': '+str(error)
                super()._restore();self.restore_seconds=time.perf_counter()-tick;return
            # A valid snapshot cannot forgive a bad tail replay.
            for _,t,payload,result,switch in self.records[n:]:
                request=json.loads(payload);expected=json.loads(result)
                actual=rebuilt.process_batch(**request,_pinned_cutoff=expected['model_cutoff'])
                if request['timestamp']!=t or canonical(actual)!=result or canonical(dict(from_cutoff=previous,to_cutoff=actual['model_cutoff']))!=switch:raise RefreshJournalError('Snapshot tail replay differs')
                previous=actual['model_cutoff']
            self.engine=rebuilt;self.last_cutoff=previous;self.replayed_batches=len(self.records)-n;self.snapshot_prefix=n;self.snapshot_status='loaded'
        else:super()._restore()
        self.restore_seconds=time.perf_counter()-tick

    def checkpoint(self):
        with self.lock:
            if self.closed or self.snapshot_path is None:raise RefreshJournalError('Checkpoint unavailable')
            body=dict(format='refresh-state-snapshot-v1',identity=self.identity,codec_sha256=codec_hash(),prefix_count=len(self.records),prefix_sha256=self.checksum,last_cutoff=self.last_cutoff,engine=capture(self.engine))
            # JSON object keys become strings; hash the stable decoded representation.
            body=json.loads(canonical(body))
            wrapper=dict(body=body,sha256=hashlib.sha256(canonical(body).encode()).hexdigest())
            raw=gzip.compress(canonical(wrapper).encode(),mtime=0)
            path=self.snapshot_path;path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
            try:
                with temporary.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,path)
                fd=os.open(path.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
            finally:
                if temporary.exists():temporary.unlink()
            return dict(bytes=len(raw),prefix_count=len(self.records),prefix_sha256=self.checksum,snapshot_sha256=file_hash(path))
