#!/usr/bin/env python3
"""Evaluate the three fixed BFS skeleton arms; preserve the older protocol."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import torch
from scripts.eval.eval_qwen21_bfs_skeleton import DEFAULT, HIGHRES, VARIANTS, PREVIOUS, sha256
from scripts.eval import score_qwen21_native_value as shared
from scripts.reports import build_qwen21_native_value_report as figures

LABELS = {'native': '原生 A＋骨架', 'bfs_body': 'BFS Body LoRA', 'bfs_head': 'BFS Head LoRA'}


def expected_keys(pairs):
    return {(p['pair_id'], res, method) for p in pairs for res in (512,1024)
            if res == 512 or p['pair_id'] in HIGHRES for method in VARIANTS}


def validate_index(root, partial=False):
    lock = json.loads((root/'experiment_lock.json').read_text())
    pairs = json.loads((root/'data/pairs.json').read_text())
    prompts = json.loads((root/'data/prompts.json').read_text())
    rows = shared.read_jsonl(root/'generations.jsonl')
    keys = [(r['pair_id'],r['resolution'],r['method']) for r in rows]
    if len(keys) != len(set(keys)) or (not partial and set(keys) != expected_keys(pairs)):
        raise ValueError('Generation coverage differs from the locked 48 outputs')
    if not set(keys) <= expected_keys(pairs):
        raise ValueError('Unexpected result key')
    for r in rows:
        if any(r.get(k) != v for k,v in dict(steps=30,seed=42,true_cfg_scale=1.,use_kv_cache=False,
                                           control_forwards=0,target_rgb_in_native_references=False).items()):
            raise ValueError('Sampling/control protocol differs')
        if r['prompt'] != prompts[r['pair_id']] or r['prompt_sha256'] != hashlib.sha256(r['prompt'].encode()).hexdigest():
            raise ValueError('Prompt changed')
        if [x['role'] for x in r['references']] != ['A','B_skeleton']:
            raise ValueError('Unexpected reference roles')
        for ref in r['references']:
            if sha256(ref['path']) != ref['sha256']:
                raise ValueError('Reference checksum changed')
        if sha256(r['image']) != r['image_sha256']:
            raise ValueError('Output checksum changed')
        if r['lora_loaded'] != (r['method'] != 'native'):
            raise ValueError('LoRA enabled-state differs')
        if r['method'] != 'native' and r['lora_strength'] != 1.:
            raise ValueError('LoRA strength differs')
        if r['runtime_versions'] != lock['versions']:
            raise ValueError('Runtime differs')
    return pairs, rows


def summary(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for r in rows:
        groups[shared.group_key(r)][r['method']].append(r)
    result={'groups':{},'paired_differences':{},'formal_identity_review_complete':all(
        r['face_selection'][role]['reviewed'] for r in rows for role in ('source','generated'))}
    for group, methods in sorted(groups.items()):
        result['groups'][group]={m:shared.aggregate(methods[m]) for m in VARIANTS if m in methods}
        result['paired_differences'][group]={m+'_minus_native':shared.paired_delta(methods['native'],methods[m])
            for m in VARIANTS[1:] if m in methods and 'native' in methods}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--review-only',action='store_true')
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args(); root=args.root
    pairs, indexed=validate_index(root,args.allow_partial)
    torch.set_num_threads(4)
    directory=root/'evaluation';directory.mkdir(exist_ok=True)
    cache=directory/'face_detection_cache.json'
    prior=PREVIOUS/'supplement_combinations/evaluation'
    if not cache.exists():
        cache.write_bytes((prior/'face_detection_cache.json').read_bytes())
    manual_path=directory/'face_review.json'
    if not manual_path.exists():
        manual_path.write_bytes((prior/'face_review.json').read_bytes())
    old_path=directory/'evaluated_samples.json'
    old=json.loads(old_path.read_text()) if old_path.exists() else []
    old_by={shared.row_key(r):r for r in old}
    baseline={ (r['pair_id'],r['resolution']):r for r in json.loads((prior/'evaluated_samples.json').read_text())
              if r['method']=='native_skeleton' and r['steps']==30}
    scorer=None
    rows=[];reused=0
    by_id={p['pair_id']:p for p in pairs}
    for r in sorted(indexed,key=shared.row_key):
        previous=old_by.get(shared.row_key(r))
        if args.review_only and previous is None:
            raise ValueError('Review-only requires cached scores for every output')
        if previous and previous['image_sha256']==r['image_sha256']:
            value={**previous,**r};reused+=1
        elif r['method']=='native':
            previous=baseline[r['pair_id'],r['resolution']]
            if previous['image_sha256'] != r['image_sha256']:
                raise ValueError('Native cached metric image differs')
            value={**previous,**r};reused+=1
        else:
            if scorer is None:
                scorer=shared.Scorer(root)
            value=scorer.score(r,by_id[r['pair_id']])
        rows.append(value)
        print(json.dumps({'evaluated':len(rows),'pair':r['pair_id'],'method':r['method'],'res':r['resolution'],
                          'pck':value['metrics']['pose_pck_002']}),flush=True)
    manual=json.loads(manual_path.read_text())
    shared.apply_face_review(rows,manual)
    stats=summary(rows)
    shared.write_tables(root,rows,stats)
    definitions=json.loads((directory/'metric_definitions.json').read_text())
    definitions['multi_stage_cost']='Not applicable: every formal arm is one edit, with no second editing stage.'
    definitions['cost_measurement']='Native is historical; BFS arms are current-run measurements.'
    shared.save(directory/'metric_definitions.json',definitions)
    # Only this process's presentation registry changes. Older CLI defaults stay intact.
    shared.ALL_METHODS=VARIANTS;shared.LABELS.update(LABELS)
    shared.review_material(root,rows)
    figures.CN.update(LABELS)
    panels=[]
    for p in pairs:
        for res in (512,1024):
            if res==1024 and p['pair_id'] not in HIGHRES:continue
            panel=figures.figure(root,rows,p,res,VARIANTS,'bfs_direct')
            if panel:panels.append(panel)
    shared.save(root/'report_figures/manifest.json',panels)
    shared.save(directory/'evaluation_protocol.json',{
        'rows':len(rows),'metrics_reused':reused,'new_metric_evaluations':len(rows)-reused,
        'scoring_source':str(Path(shared.__file__)),'scoring_source_sha256':sha256(shared.__file__),
        'face_reliable_rule':'manual real visible face + score>=0.5 + short side>=32 at 512',
        'identity_primary':'Three-arm common formally reviewed subset; n<3 cannot rank stably',
        'pose_background_resolution':512,'target_rgb_reconstruction_metrics':False,
        'native_cost':'historical','lora_cost':'current','all_panels':len(panels),
        'formal_identity_review_complete':stats['formal_identity_review_complete']})
    print(json.dumps({'complete':len(rows)==48,'panels':len(panels),
                      'manual_complete':stats['formal_identity_review_complete']}),flush=True)


if __name__=='__main__':main()
