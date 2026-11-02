"""Resolve migrated run artifacts without rewriting historical result logs."""
from __future__ import annotations

from collections.abc import Mapping
import json
import ntpath
from pathlib import Path, PureWindowsPath


def _absolute(value):
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _key(value):
    value = str(value)
    return ntpath.normcase(ntpath.normpath(value)) if PureWindowsPath(value).is_absolute() else str(Path(value))


def _old_run(value):
    if not isinstance(value, str) or not _absolute(value):
        return None
    parts = PureWindowsPath(value).parts if PureWindowsPath(value).is_absolute() else Path(value).parts
    for position, item in enumerate(parts[:-1]):
        if item.lower() == 'runs' and position + 1 < len(parts):
            return parts[position + 1].lower()
    return None


def _filename(value):
    return PureWindowsPath(value).name if PureWindowsPath(value).is_absolute() else Path(value).name


def _parent_key(value):
    return _key(str(PureWindowsPath(value).parent if PureWindowsPath(value).is_absolute() else Path(value).parent))


def build_legacy_path_index(project_root):
    """Map missing old runs paths to verified artifacts beside their own JSON.

    Only results/<run>/详细数据/**/optimization_result.json is read. A missing
    reference is matched within the same metadata file's original run/folder,
    never by searching other runs for a matching filename. Ambiguous old paths
    copied into multiple current runs are omitted instead of guessed.
    """
    root = Path(project_root).resolve()
    results = root / 'results'
    targets = {}
    if not results.is_dir():
        return {}
    for run in sorted(path for path in results.iterdir() if path.is_dir()):
        details = run / '详细数据'
        if not details.is_dir():
            continue
        for document in sorted(details.rglob('optimization_result.json')):
            try:
                resolved = document.resolve()
                if not resolved.is_relative_to(run.resolve()) or not resolved.is_relative_to(root):
                    continue
                files = json.loads(document.read_text(encoding='utf-8-sig')).get('metadata', {}).get('files', {})
            except (OSError, ValueError, AttributeError):
                continue
            if not isinstance(files, Mapping):
                continue
            original_self = files.get('result')
            self_is_old = (_old_run(original_self) is not None and _filename(original_self) == document.name)
            original_run = _old_run(original_self) if self_is_old else run.name.lower()
            for reference in files.values():
                if _old_run(reference) != original_run or not isinstance(reference, str) or Path(reference).is_file():
                    continue
                if self_is_old and _parent_key(reference) != _parent_key(original_self):
                    continue
                candidate = document.parent / _filename(reference)
                if not candidate.is_file():
                    continue
                candidate = candidate.resolve()
                if not candidate.is_relative_to(run.resolve()) or not candidate.is_relative_to(root):
                    continue
                entry = targets.setdefault(_key(reference), {'references': set(), 'targets': set()})
                entry['references'].add(reference)
                entry['targets'].add(candidate.relative_to(root).as_posix())
    return {reference: next(iter(entry['targets'])) for entry in targets.values()
            if len(entry['targets']) == 1 for reference in sorted(entry['references'])}


def write_legacy_path_index(project_root, output_path):
    """Write only the explicit old->project-relative mapping; preserve old bytes."""
    index = build_legacy_path_index(project_root)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return index


def _project_root(run_root):
    for candidate in (run_root, *run_root.parents):
        if candidate.name.lower() == 'results':
            return candidate.parent
        if (candidate / 'results').is_dir():
            return candidate
    return run_root


def resolve_artifact(reference, run_root, index=None):
    """Resolve existing paths, then an explicit migration index, or raise.

    Current relative paths are anchored at run_root; callers reading a local
    metadata.files entry may pass the JSON parent as run_root. Index values
    are always relative to the code project and cannot escape it via '..'.
    """
    reference = str(reference)
    anchor = Path(run_root).resolve()
    target = Path(reference)
    if _absolute(reference):
        if target.is_file():
            return target.resolve()
    else:
        local = anchor / target
        if local.is_file():
            return local.resolve()
        project_relative = _project_root(anchor) / target
        if project_relative.is_file():
            return project_relative.resolve()
    if index is not None:
        if not isinstance(index, Mapping):
            raise TypeError('index must be a mapping loaded from the migration JSON')
        entries = index.get('paths', index)
        if not isinstance(entries, Mapping):
            raise TypeError('index paths must be a mapping')
        matches = [value for key, value in entries.items() if _key(key) == _key(reference)]
        if len(set(str(value) for value in matches)) == 1:
            relative = str(matches[0])
            if _absolute(relative):
                raise ValueError('Migration index targets must be project-relative')
            root = _project_root(anchor)
            migrated = (root / relative).resolve()
            if not migrated.is_relative_to(root):
                raise ValueError('Migration index target escapes the project directory')
            if migrated.is_file():
                return migrated
    raise FileNotFoundError('Artifact not found; no verified migration entry: ' + reference)
