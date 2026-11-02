"""Migration indexes preserve original logs and never guess across runs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mri_magnet.artifact_paths import build_legacy_path_index, write_legacy_path_index, resolve_artifact


class ArtifactPathTests(unittest.TestCase):
    def _run(self, root, name, original_name=None):
        folder = root / 'results' / name / '详细数据' / 'optimization'
        folder.mkdir(parents=True)
        old = root / 'runs' / (original_name or name) / 'optimization'
        files = {'records': str(old / 'optimization_records.jsonl'), 'result': str(old / 'optimization_result.json')}
        document = folder / 'optimization_result.json'
        document.write_text(json.dumps({'metadata': {'files': files}}), encoding='utf-8')
        (folder / 'optimization_records.jsonl').write_text(name, encoding='utf-8')
        return folder, document, files

    def test_two_runs_with_same_filenames_resolve_to_their_own_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, first_doc, first_paths = self._run(root, 'A')
            second, _, second_paths = self._run(root, 'B')
            before = hashlib.sha256(first_doc.read_bytes()).hexdigest()
            index = write_legacy_path_index(root, root / 'support' / '路径迁移索引.json')
            self.assertEqual(resolve_artifact(first_paths['records'], root / 'results' / 'A', index), first / 'optimization_records.jsonl')
            self.assertEqual(resolve_artifact(second_paths['records'], root / 'results' / 'B', index), second / 'optimization_records.jsonl')
            self.assertEqual(hashlib.sha256(first_doc.read_bytes()).hexdigest(), before)
            self.assertFalse(Path(index[first_paths['records']]).is_absolute())

    def test_renamed_run_is_anchored_by_its_own_original_result_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, _, paths = self._run(root, '历史合并候选', 'manufacturing_run')
            index = build_legacy_path_index(root)
            self.assertEqual(resolve_artifact(paths['records'], folder, index), folder / 'optimization_records.jsonl')
            self.assertEqual(resolve_artifact('optimization_records.jsonl', folder), folder / 'optimization_records.jsonl')
            self.assertEqual(resolve_artifact(str(folder / 'optimization_records.jsonl'), folder), folder / 'optimization_records.jsonl')

    def test_missing_same_run_file_does_not_fall_back_to_another_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, _, paths = self._run(root, 'A')
            self._run(root, 'B')
            (folder / 'optimization_records.jsonl').unlink()
            index = build_legacy_path_index(root)
            self.assertNotIn(paths['records'], index)
            with self.assertRaises(FileNotFoundError):
                resolve_artifact(paths['records'], folder, index)

    def test_ambiguous_copies_and_cross_run_references_are_not_guessed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, document, paths = self._run(root, 'A', 'shared')
            self._run(root, 'B', 'shared')
            self.assertNotIn(paths['records'], build_legacy_path_index(root))
            different = str(root / 'runs' / 'unrelated' / 'optimization' / 'optimization_records.jsonl')
            data = json.loads(document.read_text(encoding='utf-8'))
            data['metadata']['files']['records'] = different
            document.write_text(json.dumps(data), encoding='utf-8')
            self.assertNotIn(different, build_legacy_path_index(root))

    def test_explicit_index_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder, _, paths = self._run(root, 'A')
            with self.assertRaises(ValueError):
                resolve_artifact(paths['records'], folder, {paths['records']: '../outside.txt'})


if __name__ == '__main__':
    unittest.main()
