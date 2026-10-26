"""Data-transfer integrity checks; no private raw data fixtures."""
import csv
import importlib.util
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

script = Path(__file__).resolve().parents[1] / 'tools/prepare_cloud_data.py'
spec = importlib.util.spec_from_file_location('prepare_cloud_data', script)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TransferTests(unittest.TestCase):
    def fixture(self, root):
        data = root / 'data'
        data.mkdir()
        (data / 'train_click_log.csv').write_text('user_id,click_article_id\n1,10\n')
        (data / 'articles.csv').write_text('article_id,category_id,created_at_ts,words_count\n10,1,0,2\n20,1,0,3\n')
        with (data / 'articles_emb.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['article_id'] + [f'emb_{i}' for i in range(250)])
            writer.writerow([20] + [.123456789] * 250)
            writer.writerow([10] + [-.987654321] * 250)
        reference = {'status': 'succeeded', 'command': 'audit', 'inputs': {}}
        for role, name in [('train', 'train_click_log.csv'), ('articles', 'articles.csv'),
                           ('embeddings', 'articles_emb.csv')]:
            path = data / name
            reference['inputs'][role] = {'name': name, 'bytes': path.stat().st_size,
                                         'sha256': prepare.sha256(path)}
        manifest = root / 'p0.json'
        manifest.write_text(json.dumps(reference))
        return data, manifest

    def test_runtime_packages_preserve_core_bytes_vector_alignment_and_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = self.fixture(root)
            snapshot = {path.name: path.read_bytes() for path in data.iterdir()}
            output = root / 'output'
            prepare.main(['--data-dir', str(data), '--p0-manifest', str(manifest), '--output', str(output)])
            with zipfile.ZipFile(output / 'NewsRec_CoreData.zip') as archive:
                self.assertEqual(archive.read('train_click_log.csv'), snapshot['train_click_log.csv'])
                self.assertEqual(archive.read('articles.csv'), snapshot['articles.csv'])
            with zipfile.ZipFile(output / 'NewsRec_Vectors_F32.zip') as archive:
                self.assertEqual(struct.unpack('<qq', archive.read('article_ids.i64')), (20, 10))
                matrix = struct.unpack('<500f', archive.read('article_vectors.f32'))
                self.assertAlmostEqual(matrix[0], .123456789, places=6)
                self.assertAlmostEqual(matrix[250], -.987654321, places=6)
                report = json.loads(archive.read('VECTOR_MANIFEST.json'))
                self.assertLess(report['max_absolute_rounding_error'], 1e-6)
                self.assertGreater(report['max_absolute_rounding_error'], 0.)
            self.assertEqual(snapshot, {path.name: path.read_bytes() for path in data.iterdir()})

    def test_source_hash_mismatch_is_not_silently_repackaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest = self.fixture(root)
            with (data / 'train_click_log.csv').open('a') as stream:
                stream.write('2,20\n')
            output = root / 'output'
            with self.assertRaises(ValueError):
                prepare.main(['--data-dir', str(data), '--p0-manifest', str(manifest), '--output', str(output)])
            self.assertFalse((output / 'NewsRec_CoreData.zip').exists())


if __name__ == '__main__':
    unittest.main()
