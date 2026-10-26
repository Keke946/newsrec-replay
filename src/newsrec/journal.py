"""Opt-in single-process ordered action journal; no pickle or engine snapshots.

The factory must always build the same primed engine. Every successful query
is replayed too: query-driven cache refresh is part of recommendation state.
"""
import fcntl
import json
from pathlib import Path
import sqlite3
import threading


class JournalError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class JournalEngine:
    def __init__(self, path, factory, identity):
        self.lock = threading.RLock()
        self.factory = factory
        self.closed = False
        self.connection = None
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # This sidecar must never be unlinked while an owner is running.
        self.owner = Path(str(self.path) + '.lock').open('a+b')
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.owner.close()
            raise JournalError('Journal already owned by another process') from error
        try:
            self.connection = sqlite3.connect(self.path, check_same_thread=False)
            tables={row[0] for row in self.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables-{'identity','actions'}:
                raise JournalError('Refusing to modify a database that is not an action journal')
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            with self.connection:
                self.connection.execute('CREATE TABLE IF NOT EXISTS identity (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL)')
                self.connection.execute('CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, result TEXT NOT NULL)')
                row = self.connection.execute('SELECT payload FROM identity WHERE id=1').fetchone()
                expected = canonical({'format': 1, 'identity': identity})
                if row is None:
                    if self.connection.execute('SELECT count(*) FROM actions').fetchone()[0]:
                        raise JournalError('Actions have no identity')
                    self.connection.execute('INSERT INTO identity VALUES (1,?)', (expected,))
                elif row[0] != expected:
                    raise JournalError('Journal model, data, source or configuration identity differs')
            self._restore()
        except BaseException:
            self.close()
            raise

    def _restore(self):
        rebuilt = self.factory()
        count = 0
        for _, kind, payload, result in self.connection.execute('SELECT * FROM actions ORDER BY id'):
            if kind not in ('ingest', 'recommend'):
                raise JournalError('Unknown journal action')
            actual = getattr(rebuilt, kind)(**json.loads(payload))
            if canonical(actual) != result:
                raise JournalError('Journal replay output differs; refusing to serve')
            count += 1
        self.engine = rebuilt
        self.replayed_actions = count

    def _persist(self, kind, payload, result):
        with self.connection:
            self.connection.execute('INSERT INTO actions(kind,payload,result) VALUES (?,?,?)',
                                    (kind, payload, result))

    def _apply(self, kind, payload):
        with self.lock:
            if self.closed:
                raise JournalError('Journal engine is closed')
            encoded = canonical(payload)
            try:
                result = getattr(self.engine, kind)(**json.loads(encoded))
                self._persist(kind, encoded, canonical(result))
                return result
            except Exception as original:
                # Rebuild even after model/contract errors: a query may already
                # have advanced clocks or expired a cache before it failed.
                # A commit may have completed before an I/O error was reported;
                # the committed journal, not a guessed rollback, is authoritative.
                try:
                    self.connection.rollback()
                    self._restore()
                except Exception as recovery:
                    self.close()
                    raise JournalError('Recovery failed; engine closed') from recovery
                if isinstance(original, sqlite3.Error):
                    raise JournalError('Journal write failed; committed state recovered') from original
                raise

    def ingest(self, timestamp, events):
        return self._apply('ingest', dict(timestamp=timestamp, events=events))

    def recommend(self, user_id, timestamp, current_article=None, k=10):
        return self._apply('recommend', dict(user_id=user_id, timestamp=timestamp,
                                            current_article=current_article, k=k))

    @property
    def version(self): return self.engine.version

    @property
    def observed_through(self): return self.engine.observed_through

    @property
    def durability(self): return 'ordered SQLite action journal, one process'

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                if self.connection is not None:
                    self.connection.close()
                self.owner.close()
