#!/usr/bin/env python3
"""Build portable, saved-image stage/variant reports for the joint BFS pilot.

No GPU, detector or metric model is needed. Per-case output crops use the same
B geometry ROI across all methods; no output detection or flattering recentering.
Run this after copying the full evaluation tree to the local computer.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.eval.eval_qwen21_joint_pilot import case_stem, crop_normalized, write_json

VARIANTS = ('init', 'J', 'Qwen', 'BFSBody', 'BFSHead')
STAGES = ('initial', 'common', 'hotload', 'mid', 'final')
METADATA_KEYS = ('pair_id', 'source_appearance', 'target_appearance', 'source_stem', 'target_stem')


def artifact_path(run, value):
    """Resolve portable paths; remap old absolute paths only inside this run."""
    if not value:
        return None
    text = str(value).replace('\\', '/')
    original = Path(text)
    if not text.startswith('/') and not original.is_absolute() and not (len(text) > 2 and text[1] == ':'):
        candidate = (run / original).resolve()
        if candidate.is_relative_to(run.resolve()):
            return candidate
        raise ValueError(f'Artifact path escapes evaluation directory: {value}')
    for folder in ('images', 'faces', 'panels'):
        marker = f'/{folder}/'
        if marker in text:
            suffix = text.rsplit(marker, 1)[1]
            candidate = (run / folder / suffix).resolve()
            if candidate.is_relative_to(run.resolve()):
                return candidate
    return None


def link(path, output):
    return quote(Path(os.path.relpath(path, output)).as_posix(), safe='/') if path is not None else None


def open_image(path):
    if path is None or not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return image.convert('RGB')
    except (OSError, ValueError):
        return None


def discover_runs(root, allow_empty=False):
    runs = []
    for records in sorted(root.rglob('per_sample.jsonl')):
        directory = records.parent
        if directory.is_relative_to(root / 'diagnostic_failures'):
            continue
        config_path = directory / 'config.json'
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if not config.get('variant') or not config.get('stage'):
            # Unrelated historical evaluations cannot masquerade as joint runs.
            continue
        rows = [json.loads(line) for line in records.read_text(encoding='utf-8').splitlines() if line.strip()]
        scores = {}
        if (directory / 'scores.json').is_file():
            score_payload = json.loads((directory / 'scores.json').read_text(encoding='utf-8'))
            scores = {row['pair_index']: row for row in score_payload.get('per_sample', [])}
        runs.append({'directory': directory, 'config': config, 'rows': rows, 'scores': scores})
    if not runs and not allow_empty:
        raise ValueError('No joint-pilot evaluation directories with config.json + per_sample.jsonl were found')
    return runs


def archived_attempts(root, output):
    attempts = []
    archive_root = root / 'diagnostic_failures'
    if not archive_root.is_dir():
        return attempts
    for records_path in sorted(archive_root.rglob('per_sample.jsonl')):
        directory = records_path.parent
        rows = [json.loads(line) for line in records_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        reason_path = directory / 'rerun_reason.json'
        reason = json.loads(reason_path.read_text(encoding='utf-8')) if reason_path.is_file() else {'category': 'unspecified archived diagnostic'}
        image_rows = []
        for row in rows:
            links = {}
            for name in ('image', 'source', 'target', 'normal', 'pose', 'comparison', 'face_comparison'):
                try:
                    path = artifact_path(directory, row.get(name))
                except ValueError:
                    path = None
                if path is not None and path.is_file():
                    links[name] = link(path, output)
            image_rows.append({**row, 'artifact_links': links})
        attempts.append({'name': directory.name, 'records_n': len(rows),
            'complete': sum(row.get('status') == 'complete' for row in rows),
            'failed': sum(row.get('status') != 'complete' for row in rows), 'reason': reason,
            'records_link': link(records_path, output), 'config_link': link(directory / 'config.json', output),
            'scores_link': link(directory / 'scores.json', output) if (directory / 'scores.json').is_file() else None,
            'reason_link': link(reason_path, output) if reason_path.is_file() else None, 'records': image_rows})
    return attempts


def collect_cases(runs):
    cases = {}
    profiles = {}
    for run in runs:
        config = run['config']
        dataset = config.get('dataset_type', 'unknown')
        profile = {key: config.get(key) for key in ('pairs_sha256', 'seed', 'steps', 'resolution', 'align_mode', 'prompt')}
        if dataset in profiles and profiles[dataset] != profile:
            raise ValueError(f'Fixed sampling/data configuration differs for {dataset}')
        profiles[dataset] = profile
        seen = set()
        for row in run['rows']:
            index = row['pair_index']
            if index in seen:
                raise ValueError(f'Duplicate pair index {index} in {run["directory"]}')
            seen.add(index)
            key = (dataset, index)
            metadata = row.get('metadata', {})
            fingerprint = {name: metadata.get(name) for name in METADATA_KEYS}
            if key not in cases:
                cases[key] = {'dataset': dataset, 'index': index, 'metadata': metadata,
                              'fingerprint': fingerprint, 'entries': {}, 'target_box': row.get('target_face_box'),
                              'source_box': row.get('source_face_box'), 'reference': (run, row)}
            case = cases[key]
            if fingerprint != case['fingerprint']:
                raise ValueError(f'Pair metadata differs for {dataset} index {index}')
            for kind in ('target', 'source'):
                old, new = case[f'{kind}_box'], row.get(f'{kind}_face_box')
                if old is None and new is not None:
                    case[f'{kind}_box'] = new
                elif old is not None and new is not None and any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(old, new)):
                    raise ValueError(f'Fixed {kind} face ROI differs for {dataset} index {index}')
            slot = (config['stage'], config['variant'])
            if slot in case['entries']:
                raise ValueError(f'Multiple runs occupy {dataset} {index} {slot}; choose a separate experiment tree')
            entry = {'run': run, 'row': row, 'image_path': artifact_path(run['directory'], row.get('image'))}
            generated = open_image(entry['image_path']) if row.get('status') == 'complete' else None
            entry['failure'] = row.get('failure') if row.get('status') != 'complete' else None
            if generated is None:
                entry['failure'] = entry['failure'] or 'Missing or unreadable generated output'
            case['entries'][slot] = entry
            # Prefer a usable reference, without any filtering of generation failures.
            reference_run, reference_row = case['reference']
            if open_image(artifact_path(reference_run['directory'], reference_row.get('source'))) is None and row.get('source'):
                case['reference'] = (run, row)
    return [cases[key] for key in sorted(cases)]


def paste_tile(canvas, image, x, y, side, message='Unavailable'):
    draw = ImageDraw.Draw(canvas)
    if image is None:
        draw.rectangle((x, y, x + side - 1, y + side - 1), fill='#512b30')
        draw.text((x + 6, y + side // 2), message[:35], fill='white')
    else:
        tile = ImageOps.contain(image, (side, side), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x + (side - tile.width) // 2, y + (side - tile.height) // 2))


def render_case(case, output, variants, stages, side):
    label_width, row_height = 120, side + 35
    whole = Image.new('RGB', (label_width + side * len(variants), row_height * len(stages) + side + 76), '#efefef')
    faces = Image.new('RGB', (whole.width, whole.height), '#efefef')
    wd, fd = ImageDraw.Draw(whole), ImageDraw.Draw(faces)
    reference_run, reference_row = case['reference']
    references = [open_image(artifact_path(reference_run['directory'], reference_row.get(name)))
                  for name in ('source', 'target', 'normal', 'pose')]
    ref_labels = ['Source A', 'Target B (offline)', 'B normal', 'B pose']
    # Reference strip needs four tiles even when a partial run has fewer variants.
    reference_side = min(side, (whole.width - label_width) // 4)
    for column, (image, title) in enumerate(zip(references, ref_labels)):
        x = label_width + column * reference_side
        wd.text((x + 4, 6), title, fill='#222222')
        fd.text((x + 4, 6), title, fill='#222222')
        paste_tile(whole, image, x, 28, reference_side)
        box = case['source_box'] if column == 0 else case['target_box']
        crop = crop_normalized(image, box, side=reference_side)[0] if image is not None else None
        paste_tile(faces, crop, x, 28, reference_side, 'No valid face ROI')
    for column, variant in enumerate(variants):
        x = label_width + column * side
        wd.text((x + 6, side + 44), variant, fill='#222222')
        fd.text((x + 6, side + 44), variant, fill='#222222')
    for row_index, stage in enumerate(stages):
        y = side + 76 + row_index * row_height
        wd.text((5, y + 8), stage, fill='#222222')
        fd.text((5, y + 8), stage, fill='#222222')
        for column, variant in enumerate(variants):
            x = label_width + column * side
            entry = case['entries'].get((stage, variant))
            generated = open_image(entry['image_path']) if entry and not entry['failure'] else None
            message = 'FAILED' if entry else 'Not evaluated'
            paste_tile(whole, generated, x, y, side, message)
            crop = crop_normalized(generated, case['target_box'], side=side)[0] if generated is not None else None
            paste_tile(faces, crop, x, y, side, message if generated is None else 'No valid B target ROI')
            if entry:
                text = entry['run']['config'].get('label', f'{variant} {stage}')[:35]
                wd.text((x + 4, y + side + 6), text, fill='#333333')
                fd.text((x + 4, y + side + 6), text, fill='#333333')
    stem = f'{case["dataset"]}_{case_stem(case["index"], case["metadata"])}'
    whole_path, face_path = output / 'panels' / f'{stem}_whole.jpg', output / 'panels' / f'{stem}_face.jpg'
    whole.save(whole_path, quality=92)
    faces.save(face_path, quality=92)
    return whole_path, face_path, faces


def build_report(root, output=None, thumb_side=192):
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / 'report'
    if thumb_side < 64:
        raise ValueError('Thumbnail side must be >=64')
    output.mkdir(parents=True, exist_ok=True)
    for directory in ('panels', 'face_sheets'):
        (output / directory).mkdir(exist_ok=True)
    archives = archived_attempts(root, output)
    runs = discover_runs(root, allow_empty=bool(archives))
    cases = collect_cases(runs)
    present_variants = {run['config']['variant'] for run in runs}
    present_stages = {run['config']['stage'] for run in runs}
    variants = [name for name in VARIANTS if name in present_variants] + sorted(present_variants - set(VARIANTS))
    stages = [name for name in STAGES if name in present_stages] + sorted(present_stages - set(STAGES))
    pages, groups, audit = [], {}, []
    total, failed = 0, 0
    for case in cases:
        whole_path, face_path, face_image = render_case(case, output, variants, stages, thumb_side)
        title = f'{case["dataset"]} #{case["index"]}: {case["metadata"].get("pair_id", "pair")}'
        appearance = case['metadata'].get('source_appearance', case['dataset'])
        groups.setdefault(f'{case["dataset"]}_{appearance}', []).append((title, face_image))
        rows = []
        for stage in stages:
            cells = [f'<th>{html.escape(stage)}</th>']
            for variant in variants:
                entry = case['entries'].get((stage, variant))
                if entry is None:
                    cells.append('<td class="missing">Not evaluated</td>')
                    continue
                total += 1
                row = entry['row']
                raw_link = link(entry['image_path'], output)
                label = entry['run']['config'].get('label', variant)
                text = html.escape(label)
                if entry['failure']:
                    failed += 1
                    text += f'<p class="failure">FAILED: {html.escape(str(entry["failure"]))}</p>'
                else:
                    text += f'<p><a href="{raw_link}">Raw output PNG</a></p>'
                text += '<p>Face condition: ' + ('valid' if row.get('face_valid') else 'invalid / unavailable') + '</p>'
                score = entry['run']['scores'].get(case['index'])
                if score:
                    fields = ('identity_A', 'face_count', 'pose_pck_002')
                    text += '<p>' + ' · '.join(f'{html.escape(name)}={html.escape(str(score.get(name)))}' for name in fields) + '</p>'
                text += f'<a href="{link(entry["run"]["directory"] / "per_sample.jsonl", output)}">Case records</a>'
                cells.append(f'<td>{text}</td>')
                audit.append({'dataset': case['dataset'], 'pair_index': case['index'], 'stage': stage, 'variant': variant,
                              'status': 'failed' if entry['failure'] else 'complete', 'failure': entry['failure'],
                              'image': raw_link, 'target_face_box': case['target_box']})
            rows.append('<tr>' + ''.join(cells) + '</tr>')
        header = '<tr><th>Stage</th>' + ''.join(f'<th>{html.escape(variant)}</th>' for variant in variants) + '</tr>'
        ref_run, ref_row = case['reference']
        reference_links = []
        for key, label in (('source', 'Source A'), ('target', 'Target B (offline)'), ('normal', 'B normal'), ('pose', 'B pose')):
            path = artifact_path(ref_run['directory'], ref_row.get(key))
            if path is not None:
                reference_links.append(f'<a href="{link(path, output)}">{label}</a>')
        pages.append(f'<article><h2>{html.escape(title)}</h2><p>{" · ".join(reference_links)}</p>'
                     f'<p>B target face ROI (normalized xyxy): {html.escape(str(case["target_box"]))}. A uses its source ROI.</p>'
                     f'<details open><summary>Whole images: stage rows × variant columns</summary><a href="{link(whole_path, output)}"><img loading="lazy" src="{link(whole_path, output)}" alt="Whole image comparison"></a></details>'
                     f'<details open><summary>Fixed B ROI crops: same location for all outputs</summary><a href="{link(face_path, output)}"><img loading="lazy" src="{link(face_path, output)}" alt="Fixed ROI face comparison"></a></details>'
                     f'<div class="scroll"><table>{header}{"".join(rows)}</table></div></article>')
    sheets = []
    for group, images in sorted(groups.items()):
        safe_group = hashlib.sha256(group.encode()).hexdigest()[:10]
        for offset in range(0, len(images), 4):
            chunk = images[offset:offset + 4]
            sheet = Image.new('RGB', (max(image.width for _, image in chunk), sum(image.height + 30 for _, image in chunk)), '#eeeeee')
            draw, y = ImageDraw.Draw(sheet), 0
            for title, image in chunk:
                draw.text((6, y + 7), title, fill='#222222')
                sheet.paste(image, (0, y + 30))
                y += image.height + 30
            path = output / 'face_sheets' / f'{safe_group}_{offset // 4:02d}.jpg'
            sheet.save(path, quality=92)
            sheets.append(f'<a href="{link(path, output)}">{html.escape(group)} / {offset // 4 + 1}</a>')
    run_links = []
    for run in runs:
        label = run['config'].get('label', str(run['directory'].name))
        run_links.append(f'<li>{html.escape(label)}: <a href="{link(run["directory"] / "config.json", output)}">config</a>'
                         f' · <a href="{link(run["directory"] / "per_sample.jsonl", output)}">records</a>'
                         + (f' · <a href="{link(run["directory"] / "scores.json", output)}">scores</a>' if (run['directory'] / 'scores.json').is_file() else '') + '</li>')
    archive_html = []
    for attempt in archives:
        entries = []
        for row in attempt['records']:
            artifacts = ' · '.join(f'<a href="{url}">{html.escape(name)}</a>' for name, url in row['artifact_links'].items())
            entries.append(f'<li>#{row.get("pair_index")} {html.escape(str(row.get("status")))}: '
                           f'{html.escape(str(row.get("failure") or ""))} {artifacts}</li>')
        extra = ''.join(f' · <a href="{attempt[key]}">{title}</a>' for key, title in
                        (('config_link', 'config'), ('scores_link', 'scores'), ('reason_link', 'rerun reason')) if attempt.get(key))
        archive_html.append(f'<details><summary>{html.escape(attempt["name"])}: {attempt["complete"]} complete / {attempt["failed"]} failed / {attempt["records_n"]} attempted</summary>'
            f'<p>{html.escape(json.dumps(attempt["reason"], ensure_ascii=False))}</p><p><a href="{attempt["records_link"]}">Archived case records</a>{extra}</p><ul>{"".join(entries)}</ul></details>')
    summary_path = output / 'summary.md'
    if not summary_path.is_file():
        summary_path = root / 'report/summary.md'
    summary_html = f'<p><a href="{link(summary_path, output)}">中文实际结果汇总 / summary.md</a></p>' if summary_path.is_file() else ''
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Body + Face joint BFS pilot</title><style>body{font:15px/1.6 system-ui,sans-serif;margin:0 auto;padding:24px;max-width:1600px;background:#fafafa;color:#222}h1{font-size:28px}h2{font-size:20px}article{border-top:1px solid #bbb;margin-top:36px;padding-top:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:10px;vertical-align:top;min-width:130px}th{background:#e8e8e8}td p{margin:6px 0}a{color:#155b98}img{max-width:100%;height:auto}.scroll{overflow:auto}.failure{color:#a02c34;font-weight:600}.missing{color:#777}summary{cursor:pointer;font-weight:600;margin:12px 0}</style></head><body>'
                f'<h1>Body + Face joint BFS pilot</h1><p>{len(cases)} fixed cases · {len(runs)} evaluation runs · {total - failed}/{total} saved outputs · {failed} failed or missing.</p>'
                '<p>Each invocation uses Source A RGB, B body geometry and B face geometry with A identity features. Target B RGB is an offline reference only. Shared Detail, Hand and Interaction are off.</p>'
                '<p>Faces are cropped at a fixed B geometry ROI across stages and variants. Missing/invalid ROIs and failed outputs remain visible. These panels do not establish identity, expression or pose quality by themselves.</p>'
                '<p>Stage rows: initial / common J / hotload (0) / mid (128) / final (256). Variant columns identify the actual frozen base. Empty cells mean that stage was not evaluated.</p>'
                '<p>Face contact sheets: ' + ' · '.join(sheets) + '</p><details><summary>Run provenance</summary><ul>' + ''.join(run_links) + '</ul></details>'
                + summary_html + ('<section><h2>Archived infrastructure attempts</h2><p>These attempts remain auditable and are excluded from primary stage/variant comparisons and metric denominators. Corrected runs use the same fixed cases.</p>' + ''.join(archive_html) + '</section>' if archives else '')
                + ''.join(pages) + '</body></html>')
    (output / 'index.html').write_text(document, encoding='utf-8')
    summary = {'cases': len(cases), 'runs': len(runs), 'outputs': total, 'complete': total - failed, 'failed': failed,
               'stages': stages, 'variants': variants, 'index': 'index.html', 'raw_path_policy': 'relative to report directory',
               'offline_B_reference': True, 'per_output': audit, 'archived_attempts': archives}
    write_json(output / 'report_manifest.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Experiment root containing evaluation directories')
    parser.add_argument('--output', type=Path, help='Default: RUN/report')
    parser.add_argument('--thumb-side', type=int, default=192)
    args = parser.parse_args(argv)
    summary = build_report(args.run, args.output, args.thumb_side)
    compact = {key: value for key, value in summary.items() if key not in ('per_output', 'archived_attempts')}
    compact.update(archived_attempts_count=len(summary['archived_attempts']),
                   archived_case_records=sum(attempt['records_n'] for attempt in summary['archived_attempts']),
                   archived_failures=sum(attempt['failed'] for attempt in summary['archived_attempts']))
    print(json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
