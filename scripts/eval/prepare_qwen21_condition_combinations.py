#!/usr/bin/env python3
"""Archive the old report and lock a supplement without changing old assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.eval.qwen21_condition_combinations import COMBINATION_METHODS, NEW_METHODS, combination_prompt, output_key
from scripts.eval.qwen21_native_value_prompts import build_prompt

DEFAULT_PARENT = Path('/home/shangguanrz/nihaochen/qwen21_full/native_value_audit_20261008')
REPORT = Path('/home/shangguanrz/nihaochen/docs/Qwen基座与控制模块_价值核对_20261008.docx')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def save_once(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Refusing to replace locked supplementary data: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f'Archived source differs: {source}')
    else:
        shutil.copy2(source, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, default=DEFAULT_PARENT)
    a = p.parse_args(); parent = a.parent; root = parent/'supplement_combinations'
    root.mkdir(parents=True, exist_ok=True)
    original = read_rows(parent/'generations.jsonl')
    pairs = read_rows(parent/'data/pairs.jsonl')
    dev = read_rows(parent/'data/dev_pairs.jsonl')
    if len(original) != 140 or len(pairs) != 12 or len(dev) != 4:
        raise ValueError('Unexpected original experiment size')
    report_meta = json.loads((parent/'report_manifest.json').read_text())
    if sha(REPORT) != report_meta['sha256']:
        raise ValueError('The report changed since the previous manifest')
    source_paths = [parent/'generations.jsonl', parent/'prompts/locked_prompts.json',
                    parent/'data/pairs.jsonl', parent/'data/dev_pairs.jsonl', parent/'data/report_selection.json',
                    parent/'evaluation/evaluated_samples.json', parent/'evaluation/metric_summary.json',
                    parent/'evaluation/face_review.json', parent/'report_manifest.json',
                    parent/'final_observations.json', parent/'acceptance/experiment_lock.json']
    originals = {str(path):sha(path) for path in source_paths}
    for path in source_paths:
        copy_once(path, root/'archive'/path.relative_to(parent))
    copy_once(REPORT, root/'archive/report_before_supplement.docx')
    for name in ('pairs.jsonl','dev_pairs.jsonl','report_selection.json'):
        copy_once(parent/'data'/name, root/'data'/name)
    legacy_prompts = json.loads((parent/'prompts/locked_prompts.json').read_text())
    formal_prompts = {}
    for row in pairs:
        base = legacy_prompts[row['pair_id']]['native_skeleton']
        if base != build_prompt(row, 'native_skeleton'):
            raise ValueError('The original detailed baseline prompt cannot be reproduced')
        formal_prompts[row['pair_id']] = {**legacy_prompts[row['pair_id']],
            **{method:combination_prompt(base,method) for method in NEW_METHODS}}
    save_once(root/'prompts/locked_prompts.json', formal_prompts)
    dev_prompts = {row['pair_id']:{method:combination_prompt(build_prompt(row,'native_skeleton'),method)
                   for method in COMBINATION_METHODS} for row in dev}
    save_once(root/'prompts/development_prompts.json', dev_prompts)
    highres = {row['pair_id'] for row in pairs if row['highres_check']}
    if highres != {'iper_00','iper_01','tiktok_00','tiktok_01'}:
        raise ValueError('High-resolution sample selection differs')
    reused = [row for row in original if row['method']=='native_skeleton' and row['steps']==30]
    if len(reused) != 16:
        raise ValueError('Expected 16 reusable skeleton results')
    for row in original:
        if sha(row['image']) != row['image_sha256']:
            raise ValueError(f'Old output changed: {row["image"]}')
        if row['prompt'] != legacy_prompts[row['pair_id']][row['method'] if row['method']!='simple_body1024' else 'simple_body']:
            raise ValueError('Legacy result/prompt mismatch')
        for ref in row['references']:
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Legacy reference changed')
    additional = [dict(pair_id=row['pair_id'],resolution=res,steps=30,method=m)
        for res in (512,1024) for row in pairs if res==512 or row['pair_id'] in highres for m in NEW_METHODS]
    expected = [dict(pair_id=x['pair_id'],resolution=x['resolution'],steps=x['steps'],method=x['method']) for x in original]+additional
    protocol = {'architecture':'native_condition_combinations_v1','expected_outputs':expected,
                'new_outputs':additional,'combination_methods':list(COMBINATION_METHODS),
                'new_methods':list(NEW_METHODS),'new_formal_count':112,'reused_baseline_count':16,
                'combination_count':128,'merged_formal_count':252,'development_count':8,
                'highres_pair_ids':sorted(highres),'seed':42,'steps':30,'true_cfg_scale':1.0,
                'use_kv_cache':False,'training':False,'lora_loaded':False}
    if len(expected)!=252 or len({output_key(x) for x in expected})!=252 or len(additional)!=112:
        raise AssertionError('Supplementary output count mismatch')
    save_once(root/'protocol.json',protocol)
    if not (root/'generations.jsonl').exists():
        copy_once(parent/'generations.jsonl',root/'generations.jsonl')
    for name in ('evaluated_samples.json','face_detection_cache.json','face_review.json'):
        if not (root/'evaluation'/name).exists():
            copy_once(parent/'evaluation'/name,root/'evaluation'/name)
    frozen = {str(root/name):sha(root/name) for name in ('protocol.json','data/pairs.jsonl','data/dev_pairs.jsonl',
              'data/report_selection.json','prompts/locked_prompts.json','prompts/development_prompts.json')}
    save_once(root/'acceptance/input_lock.json',{'old_sources_sha256':originals,'supplement_sha256':frozen,
        'old_report_sha256':sha(REPORT),'archive_report':str(root/'archive/report_before_supplement.docx')})
    manifest = root/'run_manifest.json'
    if not manifest.exists():
        start = datetime.fromtimestamp((PROJECT/'scripts/eval/qwen21_condition_combinations.py').stat().st_mtime,timezone.utc)
        save_once(manifest,{'start_utc':start.isoformat(),'prepared_utc':datetime.now(timezone.utc).isoformat(),
            'completed':False,'training':False,'lora_loaded':False,'planned_new_formal_outputs':112,
            'planned_development_outputs':8,'expected_merged_outputs':252,'old_report_archived':True})
    print(json.dumps({'root':str(root),'additional':len(additional),'reused':len(reused),'expected':len(expected)},ensure_ascii=False))


if __name__=='__main__': main()
