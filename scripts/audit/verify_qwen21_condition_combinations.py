#!/usr/bin/env python3
"""Artifact-based acceptance of the native-only supplementary experiment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from scripts.eval.qwen21_condition_combinations import (COMBINATION_REFERENCES,NEW_METHODS,
    output_key,require_protocol_outputs)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--skip-report',action='store_true');a=p.parse_args();r=a.root
    lock=json.loads((r/'acceptance/input_lock.json').read_text())
    for group in ('old_sources_sha256','supplement_sha256'):
        for path,value in lock[group].items():assert sha(path)==value,f'Locked file changed: {path}'
    assert sha(lock['archive_report'])==lock['old_report_sha256']
    development_review=json.loads((r/'acceptance/development_review.json').read_text())
    assert development_review['passed'] and development_review['reviewed_development_outputs']==8
    for path,value in development_review['generation_source_sha256'].items():
        assert sha(path)==value,f'Inference source changed after prompt lock: {path}'
    rows=[json.loads(x) for x in (r/'generations.jsonl').read_text().splitlines() if x.strip()]
    expected=require_protocol_outputs(r/'protocol.json',rows)
    old=[json.loads(x) for x in (r/'archive/generations.jsonl').read_text().splitlines() if x.strip()]
    index={output_key(x):x for x in rows}
    assert all(index[output_key(x)]==x for x in old),'Legacy output records changed'
    new=[x for x in rows if x['method'] in NEW_METHODS];assert len(new)==112
    dev=[json.loads(x) for x in (r/'development_generations.jsonl').read_text().splitlines() if x.strip()]
    assert len(dev)==8 and len({output_key(x) for x in dev})==8
    prompts=json.loads((r/'prompts/locked_prompts.json').read_text())
    devprompts=json.loads((r/'prompts/development_prompts.json').read_text())
    checkpoints={}
    for row in rows+dev:
        assert sha(row['image'])==row['image_sha256']
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        for ref in row['references']:assert sha(ref['path'])==ref['sha256']
        if row.get('checkpoint'):checkpoints[row['checkpoint']]=row['checkpoint_sha256']
    for path,value in checkpoints.items():assert sha(path)==value,f'Weight changed: {path}'
    for row in new+dev:
        assert row['prompt']==(devprompts if row in dev else prompts)[row['pair_id']][row['method']]
        assert row['steps']==30 and row['seed']==42 and row['true_cfg_scale']==1.
        assert row['use_kv_cache'] is False and row['lora_loaded'] is False
        assert row['checkpoint'] is None and row['control_forwards']==0
        recipe=COMBINATION_REFERENCES[row['method']]
        assert [x['role'] for x in row['references']]==list(recipe)
        assert [x['index'] for x in row['references']]==list(range(1,len(recipe)+1))
        assert row['target_rgb_in_native_references']==('B' in recipe)
        witness=row['correctness_observation'];tokens=(row['resolution']//16)**2
        assert witness['native_transformer_calls']==30 and witness['target_token_count']==tokens
        assert witness['native_prefix_image_token_counts']==[len(recipe)*tokens]
        assert witness['control_context_installed'] is False and witness['control_prefix_bitwise_checks']==0
    evaluated=json.loads((r/'evaluation/evaluated_samples.json').read_text())
    assert {output_key(x) for x in evaluated}==expected and len(evaluated)==252
    assert all(x['face_selection'][role]['reviewed'] for x in evaluated for role in ('source','target','generated'))
    summary=json.loads((r/'evaluation/metric_summary.json').read_text())
    assert summary['formal_identity_review_complete']
    assert sum(n['n_images'] for group in summary['native_combination_groups'].values() for n in group.values())==128
    assert all(len(group)==12 for group in summary['combination_factor_differences'].values())
    result={'passed':True,'merged_formal_outputs':252,'native_combination_outputs':128,
            'new_formal_outputs':112,'reused_baseline_outputs':16,'development_outputs':8,
            'old_data_unchanged':True,'old_weights_unchanged':True,'training':False,'lora_loaded':False,
            'review_complete':True,'reference_order_and_prompt_match':True,'word_checked':False}
    if not a.skip_report:
        report=json.loads((r/'report_manifest.json').read_text())
        assert sha(report['docx'])==report['sha256'] and report['embedded_figures']==37
        with zipfile.ZipFile(report['docx']) as z:
            assert z.testzip() is None
            media=[x for x in z.namelist() if x.startswith('word/media/')]
            assert len(media)==37
            assert {hashlib.sha256(z.read(x)).hexdigest() for x in media}=={x['sha256'] for x in report['figures']}
        result['word_checked']=True
    (r/'acceptance/final_deliverables.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
