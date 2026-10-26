"""Prepare two private runtime data ZIPs; read-only sources, bounded vector RAM.

python3 prepare_cloud_data.py --data-dir /path/tcdata --p0-manifest manifest.json
    --output /path/new-directory
No dependencies, no network, no pickle, no training. Original CSVs unchanged.
"""
import argparse
import array
import csv
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def checked_source(directory, reference, role):
    entry = reference['inputs'][role]
    name = entry['name']
    if Path(name).name != name:
        raise ValueError('Manifest filenames must be basenames')
    path = directory / name
    if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError(f'{role}: source differs from verified P0 input')
    return path


def make_core(directory, reference, output):
    paths = {role: checked_source(directory, reference, role) for role in ('train', 'articles')}
    partial = output / 'NewsRec_CoreData.zip.partial'
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for role, path in paths.items():
            archive.write(path, path.name)
            # Verify the original bytes again after archival; ZIP is lossless.
            if sha256(path) != reference['inputs'][role]['sha256']:
                raise ValueError(f'{role} changed during archival')
        archive.writestr('SOURCE_MANIFEST.json', json_bytes({
            'scope': 'complete train role and article catalog; no test roles',
            'conversion': 'none; original CSV bytes',
            'inputs': {role: reference['inputs'][role] for role in paths}}))
    final = output / 'NewsRec_CoreData.zip'
    partial.rename(final)
    print(f'CORE READY: {final} ({final.stat().st_size} bytes); can upload this now.', flush=True)
    return final


def make_vectors(directory, reference, output):
    source = checked_source(directory, reference, 'embeddings')
    original_stat = (source.stat().st_size, source.stat().st_mtime_ns)
    article_source = checked_source(directory, reference, 'articles')
    with article_source.open(newline='', encoding='utf-8-sig') as stream:
        catalog_ids = {int(row['article_id']) for row in csv.DictReader(stream)}
    ids = array.array('q')
    if ids.itemsize != 8 or array.array('f').itemsize != 4:
        raise RuntimeError('Expected IEEE 4-byte float and 8-byte signed integer arrays')
    seen = set()
    rows = 0
    max_absolute_error = squared_error = 0.
    partial = output / 'NewsRec_Vectors_F32.zip.partial'
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        with source.open(newline='', encoding='utf-8-sig') as stream, archive.open('article_vectors.f32', 'w', force_zip64=True) as binary:
            reader = csv.reader(stream)
            if next(reader) != ['article_id'] + [f'emb_{i}' for i in range(250)]:
                raise ValueError('Unexpected embedding header')
            for line, row in enumerate(reader, 2):
                if len(row) != 251:
                    raise ValueError(f'Invalid embedding width on CSV line {line}')
                item = int(row[0])
                if item < 0 or item in seen:
                    raise ValueError(f'Negative/duplicate item on CSV line {line}')
                seen.add(item)
                original = [float(value) for value in row[1:]]
                if not all(math.isfinite(value) for value in original):
                    raise ValueError(f'Non-finite vector on line {line}')
                values = array.array('f', original)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(f'float32 overflow on line {line}')
                for old, new in zip(original, values):
                    delta = old - new
                    max_absolute_error = max(max_absolute_error, abs(delta))
                    squared_error += delta * delta
                if sys.byteorder != 'little':
                    values.byteswap()
                binary.write(values.tobytes())
                ids.append(item)
                rows += 1
                if rows % 50000 == 0:
                    print(f'Converted {rows} / {len(catalog_ids)} vectors...', flush=True)
        if seen != catalog_ids:
            raise ValueError('Embedding IDs differ from complete article catalog')
        if (source.stat().st_size, source.stat().st_mtime_ns) != original_stat or sha256(source) != reference['inputs']['embeddings']['sha256']:
            raise ValueError('Original embeddings changed during conversion')
        if sys.byteorder != 'little':
            ids.byteswap()
        archive.writestr('article_ids.i64', ids.tobytes())
        archive.writestr('VECTOR_MANIFEST.json', json_bytes({
            'rows': rows, 'dimensions': 250, 'layout': 'row-major',
            'vectors_dtype': '<f4', 'article_ids_dtype': '<i8',
            'row_alignment': 'row j of vectors corresponds to element j of article_ids',
            'conversion': 'numeric float32 conversion, NOT claimed lossless relative to decimal CSV',
            'max_absolute_rounding_error': max_absolute_error,
            'rmse': math.sqrt(squared_error / (rows * 250)) if rows else None,
            'source': reference['inputs']['embeddings'],
            'provenance_limit': 'Original embedding training/availability time is unknown'}))
    final = output / 'NewsRec_Vectors_F32.zip'
    partial.rename(final)
    print(f'VECTORS READY: {final} ({final.stat().st_size} bytes)', flush=True)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--p0-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--core-only', action='store_true')
    args = parser.parse_args(argv)
    source, output = args.data_dir.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error('Output must be separate from the input data tree')
    reference = json.loads(args.p0_manifest.read_text())
    if reference['status'] != 'succeeded' or reference['command'] != 'audit':
        parser.error('Need the successfully completed P0 manifest')
    output.mkdir(parents=True, exist_ok=False)
    files = [make_core(source, reference, output)]
    if not args.core_only:
        files.append(make_vectors(source, reference, output))
    package_manifest = {'status': 'succeeded', 'sources_read_only': True,
                        'core_only': args.core_only, 'artifacts': []}
    for path in files:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Failed ZIP integrity: {path.name}')
        package_manifest['artifacts'].append({'name': path.name, 'bytes': path.stat().st_size,
                                               'sha256': sha256(path)})
    (output / 'TRANSFER_MANIFEST.json').write_bytes(json_bytes(package_manifest))
    print('Finished. Upload CORE first, then VECTORS and TRANSFER_MANIFEST.json.', flush=True)


if __name__ == '__main__':
    main()
