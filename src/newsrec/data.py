"""CSV contracts and a disk-backed, read-only-source audit.

Only this program's own SQLite database is written. Unknown pickle/model files
are never opened. CSV rows remain traceable by source role and row number.
"""
import csv
import hashlib
import json
import math
import os
import platform
import sqlite3
import subprocess
import sys
from collections import Counter
from pathlib import Path

CLICK_COLUMNS = ('user_id', 'click_article_id', 'click_timestamp', 'click_environment',
                 'click_deviceGroup', 'click_os', 'click_country', 'click_region',
                 'click_referrer_type')
ARTICLE_COLUMNS = ('article_id', 'category_id', 'created_at_ts', 'words_count')
ALIASES = {
    'train': ('train_click_log.csv', 'train_click_sequences.csv'),
    'testA': ('testA_click_log.csv', 'testA_click_sequences.csv'),
    'testB': ('testB_click_log_Test_B.csv', 'testB_click_log.csv', 'testB_click_sequences.csv'),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    gpu = {'status': 'unavailable-or-command-missing'}
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=8)
        gpu = {'status': 'ok' if proc.returncode == 0 else 'command-failed',
               'devices': proc.stdout.strip().splitlines()}
    except (OSError, subprocess.TimeoutExpired):
        pass
    memory = None
    try:
        memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (ValueError, OSError, AttributeError):
        pass
    return {'python': sys.version, 'platform': platform.platform(),
            'cpu_count': os.cpu_count(), 'physical_memory_bytes': memory, 'gpu': gpu}


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def discover(directory):
    directory = Path(directory).resolve()
    files = {'articles': directory / 'articles.csv'}
    for role, names in ALIASES.items():
        found = [directory / name for name in names if (directory / name).is_file()]
        if len(found) > 1:
            raise ValueError(f'Ambiguous {role} inputs: {[p.name for p in found]}')
        if found:
            files[role] = found[0]
    if 'train' not in files or not files['articles'].is_file():
        raise ValueError('data-dir must contain train clicks and articles.csv directly')
    if (directory / 'articles_emb.csv').is_file():
        files['embeddings'] = directory / 'articles_emb.csv'
    return files


def integer_rows(path, columns):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(columns):
            raise ValueError(f'{Path(path).name}: unexpected header {reader.fieldnames}')
        for ordinal, row in enumerate(reader, 2):
            try:
                if None in row:
                    raise ValueError('extra CSV columns')
                values = tuple(int(row[col]) for col in columns)
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(f'{Path(path).name}: invalid integer row {ordinal}') from exc
            if any(value < 0 for value in values):
                raise ValueError(f'{Path(path).name}: negative field at row {ordinal}')
            yield ordinal, values


def load_database(files, destination):
    connection = sqlite3.connect(destination)
    connection.execute('CREATE TABLE articles(article INTEGER PRIMARY KEY, category INTEGER, created INTEGER, words INTEGER)')
    connection.execute('CREATE TABLE clicks(role TEXT, ordinal INTEGER, user INTEGER, article INTEGER, timestamp INTEGER, '
                       'environment INTEGER, device INTEGER, os INTEGER, country INTEGER, region INTEGER, referrer INTEGER)')
    with connection:
        connection.executemany('INSERT INTO articles VALUES(?,?,?,?)',
                               (row for _, row in integer_rows(files['articles'], ARTICLE_COLUMNS)))
    for role in ALIASES:
        if role not in files:
            continue
        print(f'Loading {role} CSV into run-local SQLite...', flush=True)
        with connection:
            connection.executemany('INSERT INTO clicks VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                                   ((role, ordinal, *row) for ordinal, row in integer_rows(files[role], CLICK_COLUMNS)))
    with connection:
        connection.execute('CREATE INDEX by_role_user_time ON clicks(role,user,timestamp,ordinal)')
        connection.execute('CREATE INDEX by_role_time ON clicks(role,timestamp,ordinal)')
        connection.execute('CREATE INDEX by_article ON clicks(article)')
    return connection


def summarize_clicks(connection, role):
    def scalar(sql):
        return connection.execute(sql, (role,)).fetchone()[0]
    count, users, items, low, high = connection.execute(
        'SELECT count(*),count(DISTINCT user),count(DISTINCT article),min(timestamp),max(timestamp) '
        'FROM clicks WHERE role=?', (role,)).fetchone()
    sizes = Counter(n for n, in connection.execute(
        'SELECT count(*) FROM clicks WHERE role=? GROUP BY user', (role,)))
    bins = {'1': 0, '2-5': 0, '6-10': 0, '>10': 0}
    for size, frequency in sizes.items():
        bins['1' if size == 1 else '2-5' if size <= 5 else '6-10' if size <= 10 else '>10'] += frequency
    anomalies = scalar('SELECT count(*) FROM clicks c JOIN articles a ON c.article=a.article '
                       'WHERE role=? AND c.timestamp<a.created')
    repeated_pairs = scalar('SELECT coalesce(sum(n-1),0) FROM '
                            '(SELECT count(*) n FROM clicks WHERE role=? GROUP BY user,article)')
    identical = scalar('SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM clicks WHERE role=? '
                       'GROUP BY user,article,timestamp,environment,device,os,country,region,referrer)')
    unsorted = scalar('SELECT count(*) FROM (SELECT user FROM '
                      '(SELECT user,timestamp,lag(timestamp) OVER(PARTITION BY user ORDER BY ordinal) previous '
                      'FROM clicks WHERE role=?) WHERE timestamp<previous GROUP BY user)')
    missing = scalar('SELECT count(*) FROM clicks c LEFT JOIN articles a ON a.article=c.article '
                     'WHERE role=? AND a.article IS NULL')
    return {'clicks': count, 'users': users, 'articles': items, 'min_timestamp': low,
            'max_timestamp': high, 'mean_history': count / users if users else None,
            'history_length_user_bins': bins, 'repeated_user_article_extra_rows': repeated_pairs,
            'identical_event_extra_rows': identical, 'users_unsorted_in_source': unsorted,
            'click_before_creation': anomalies, 'click_rows_without_metadata': missing}


def embedding_audit(path, connection):
    """Stream 250-D vectors; never materialize the complete matrix in RAM."""
    connection.execute('CREATE TABLE embedding_ids(article INTEGER PRIMARY KEY)')
    rows = duplicates = nonfinite = zero_norm = 0
    minimum, maximum = math.inf, -math.inf
    with Path(path).open(newline='', encoding='utf-8-sig') as stream, connection:
        reader = csv.reader(stream)
        header = next(reader)
        expected = ['article_id'] + [f'emb_{i}' for i in range(250)]
        if header != expected:
            raise ValueError('Embedding schema must be article_id plus emb_0 ... emb_249')
        for line, row in enumerate(reader, 2):
            if len(row) != 251:
                raise ValueError(f'Embedding row {line}: expected 251 columns')
            article = int(row[0])
            if article < 0:
                raise ValueError(f'Negative embedding article ID at row {line}')
            values = [float(value) for value in row[1:]]
            rows += 1
            if not all(math.isfinite(value) for value in values):
                nonfinite += 1
            else:
                norm = math.sqrt(math.fsum(v * v for v in values))
                if not math.isfinite(norm):
                    nonfinite += 1
                else:
                    zero_norm += norm == 0
                    minimum, maximum = min(minimum, norm), max(maximum, norm)
            cursor = connection.execute('INSERT OR IGNORE INTO embedding_ids VALUES(?)', (article,))
            duplicates += cursor.rowcount == 0
            if rows % 50000 == 0:
                print(f'Embedding rows scanned: {rows}', flush=True)
    return {'rows': rows, 'dimensions': 250, 'duplicate_ids': duplicates,
            'nonfinite_rows': nonfinite, 'zero_norm_rows': zero_norm,
            'norm_min': minimum if minimum != math.inf else None,
            'norm_max': maximum if maximum != -math.inf else None,
            'articles_without_embedding': connection.execute(
                'SELECT count(*) FROM articles a LEFT JOIN embedding_ids e ON a.article=e.article '
                'WHERE e.article IS NULL').fetchone()[0],
            'click_rows_without_embedding': connection.execute(
                'SELECT count(*) FROM clicks c LEFT JOIN embedding_ids e ON c.article=e.article '
                'WHERE e.article IS NULL').fetchone()[0]}


def audit(connection, files, skip_embeddings=False):
    result = {'roles': {}, 'cross_role': [], 'articles': {}, 'embeddings': {'status': 'absent'}}
    articles = connection.execute('SELECT count(*),min(created),max(created) FROM articles').fetchone()
    result['articles'] = dict(zip(('rows', 'min_created_at', 'max_created_at'), articles))
    for role in ALIASES:
        if role in files:
            result['roles'][role] = summarize_clicks(connection, role)
    roles = list(result['roles'])
    for i, first in enumerate(roles):
        for second in roles[i + 1:]:
            overlap = connection.execute(
                'SELECT count(*) FROM (SELECT DISTINCT user FROM clicks WHERE role=? '
                'INTERSECT SELECT DISTINCT user FROM clicks WHERE role=?)', (first, second)).fetchone()[0]
            event_overlap = connection.execute(
                'SELECT count(*) FROM (SELECT user,article,timestamp FROM clicks WHERE role=? '
                'INTERSECT SELECT user,article,timestamp FROM clicks WHERE role=?)', (first, second)).fetchone()[0]
            result['cross_role'].append({'roles': [first, second], 'shared_users': overlap,
                                         'shared_distinct_user_article_time': event_overlap})
    train = result['roles']['train']
    count = train['clicks']
    if count:
        cutoffs = [connection.execute('SELECT timestamp FROM clicks WHERE role=? ORDER BY timestamp '
                                     'LIMIT 1 OFFSET ?', ('train', int((count - 1) * q))).fetchone()[0]
                   for q in (.7, .85)]
        result['proposed_time_cutoffs'] = {'train_end_ms': cutoffs[0], 'validation_end_ms': cutoffs[1],
                                           'status': 'diagnostic proposal, not frozen protocol'}
        cold = connection.execute(
            'SELECT count(*),sum(CASE WHEN article NOT IN '
            '(SELECT article FROM clicks WHERE role="train" AND timestamp<=?) THEN 1 ELSE 0 END) '
            'FROM clicks WHERE role="train" AND timestamp>? AND timestamp<=?',
            (cutoffs[0], cutoffs[0], cutoffs[1])).fetchone()
        result['validation_event_catalog_diagnostic'] = {
            'events': cold[0], 'events_on_items_unseen_at_train_end': cold[1] or 0,
            'warning': 'event-level diagnostic; NOT request-time cold-start query metric'}
        query_stats = connection.execute('''
            WITH sequence AS (
                SELECT article,timestamp,
                    lead(article) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target,
                    lead(timestamp) OVER(PARTITION BY user ORDER BY timestamp,ordinal) target_time,
                    row_number() OVER(PARTITION BY user ORDER BY timestamp,ordinal) history_length
                FROM clicks WHERE role='train'
            ), first_seen AS (
                SELECT article,min(timestamp) first_time FROM clicks WHERE role='train' GROUP BY article
            )
            SELECT count(*),
                coalesce(sum(CASE WHEN a.created<=s.timestamp THEN 1 ELSE 0 END),0),
                coalesce(sum(CASE WHEN f.first_time>s.timestamp THEN 1 ELSE 0 END),0),
                coalesce(sum(CASE WHEN history_length=1 THEN 1 ELSE 0 END),0),
                coalesce(sum(CASE WHEN target=s.article THEN 1 ELSE 0 END),0)
            FROM sequence s LEFT JOIN articles a ON s.target=a.article
            LEFT JOIN first_seen f ON f.article=s.target
            WHERE s.timestamp>? AND s.target_time>s.timestamp AND s.target_time<=?
        ''', tuple(cutoffs)).fetchone()
        result['proposed_validation_queries'] = dict(zip(
            ('requests', 'targets_published_by_request', 'targets_zero_behavior_at_request',
             'requests_with_one_history_event', 'immediate_repeat_targets'), query_stats))
        result['proposed_validation_queries']['warning'] = (
            'next-click proxy, train-only observed corpus, diagnostic windows; '
            'unobserved catalog availability and embedding provenance remain unknown')
    if 'embeddings' in files:
        result['embeddings'] = ({'status': 'explicitly-skipped'} if skip_embeddings else
                                {'status': 'scanned', **embedding_audit(files['embeddings'], connection)})
    return result
