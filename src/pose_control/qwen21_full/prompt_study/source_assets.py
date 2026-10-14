"""Bind the actual RGB and geometry bytes consumed by the locked study."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .protocol import sha256_file


def source_asset_paths(row: dict[str, Any]) -> dict[str, Path]:
    fields = ('source_rgb', 'target_rgb', 'source_person_mask', 'target_person_mask',
              'target_normal', 'target_depth_vis', 'target_dwpose_json')
    paths = {field: Path(row[field]).resolve() for field in fields}
    paths['target_dwpose'] = Path(row.get('target_dwpose_png') or row['target_dwpose']).resolve()
    target_root = paths['target_normal'].parent.parent
    part = target_root / 'part_512' / f"{row['target_stem']}.png"
    if not part.exists() and part.with_suffix('.jpg').exists():
        part = part.with_suffix('.jpg')
    paths.update(target_part=part, target_v6_conditions=target_root / 'v6_conditions.pt')
    source_root = target_root.parent / str(row.get('source_appearance', row['sequence_id']))
    for side, root in (('source', source_root), ('target', target_root)):
        paths[f'{side}_smplestx_params'] = root / 'smplestx/params.pt'
        paths[f'{side}_smplestx_frame_map'] = root / 'smplestx/frame_map.json'
    return {key: path.resolve() for key, path in paths.items()}


def add_source_asset_hashes(records: list[dict[str, Any]], pairs: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = {(split, row['pair_id']): row for split, row in pairs}
    hashes: dict[Path, str] = {}
    result = []
    for record in records:
        split = 'diagnostic' if record['split'] == 'diag' else record['split']
        row = rows[(split, record['pair_id'])]
        files = {}
        for role, path in source_asset_paths(row).items():
            if path not in hashes:
                hashes[path] = sha256_file(path)
            files[role] = {'path': str(path), 'sha256': hashes[path]}
        result.append({**record, 'split': split, 'source_files': files})
    if len(result) != len(rows):
        raise ValueError('Source asset pair coverage differs')
    return result


def validate_source_asset_manifest(data_dir: Path, lock: dict[str, Any], split_rows: dict[str, list], *, verify_files: bool) -> None:
    expected = lock.get('source_assets_manifest')
    if not isinstance(expected, dict):
        raise ValueError('Missing locked source asset manifest')
    path = (data_dir / expected['path']).resolve()
    if not path.is_relative_to(data_dir.resolve()) or sha256_file(path) != expected['sha256']:
        raise ValueError('Changed source asset manifest')
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    wanted = {(split, row['pair_id']): row for split, rows in split_rows.items() for row in rows}
    keys = [(r['split'], r['pair_id']) for r in records]
    if len(records) != expected['count'] or len(set(keys)) != len(keys) or set(keys) != set(wanted):
        raise ValueError('Source asset manifest pair coverage differs')
    checked: dict[Path, str] = {}
    for record in records:
        paths = source_asset_paths(wanted[(record['split'], record['pair_id'])])
        files = record.get('source_files', {})
        if set(files) != set(paths):
            raise ValueError('Source asset role coverage differs')
        for role, source_path in paths.items():
            asset = files[role]
            if Path(asset['path']).resolve() != source_path:
                raise ValueError(f'Changed source asset binding: {role}')
            if verify_files:
                if source_path not in checked:
                    checked[source_path] = sha256_file(source_path)
                if checked[source_path] != asset['sha256']:
                    raise ValueError(f'Changed source asset: {source_path}')
