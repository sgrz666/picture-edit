#!/usr/bin/env python3
"""Hash the delivered source and verify inherited neural modules stayed unchanged."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import ast
import hashlib
import json
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[2]
FAMILY=Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1')
CORE=('src/data/canvas_contract.py','src/pose_control/qwen21_full/native_body.py',
    'src/pose_control/qwen21_full/native_geometry.py','src/pose_control/qwen21_full/condition_injector.py',
    'src/pose_control/qwen21_full/token_encoders.py','src/pose_control/qwen21_full/reasoning/person_geometry.py',
    'src/pose_control/qwen21_full/native_depth_v2.py','src/pose_control/qwen21_full/native_dual_runtime.py',
    'src/pose_control/qwen21_full/native_full_runtime.py','src/pose_control/qwen21_full/native_v5/runtime.py',
    'src/pose_control/qwen21_full/native_v5/integration.py','src/pose_control/qwen21_full/native_v5/frontend.py',
    'src/pose_control/qwen21_full/native_v5/local.py','src/pose_control/qwen21_full/native_v5/detail.py',
    'src/pose_control/qwen21_full/native_v5/strength.py')


def sha(value):return hashlib.sha256(value).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new immutable audit directory')
    if not a.output.resolve().is_relative_to('/data/shangguanrz'):raise ValueError('Source archive belongs on persistent storage')
    original=FAMILY/'source/baseline_20261025T094000Z/source.tar.gz'
    with tarfile.open(original) as archive:
        protected={path:{'baseline':sha(archive.extractfile(path).read()),'current':sha((ROOT/path).read_bytes())} for path in CORE}
    if any(v['baseline']!=v['current'] for v in protected.values()):raise ValueError('Inherited core neural module changed unexpectedly')
    files=set(CORE)
    for folder in ('src/pose_control/qwen21_full/native_v5','scripts/audit','scripts/data','scripts/train','scripts/run','scripts/reports','tests'):
        for path in (ROOT/folder).glob('**/*.py'):
            rel=path.relative_to(ROOT).as_posix()
            if '/native_v5/' in rel or any(k in path.name for k in ('fixedpair','multiref','face72_migration','dwpose_response','v5_real_tiny')):
                files.add(rel)
    files.update(('src/pose_control/qwen21_full/qwen_runtime.py','src/pose_control/qwen21_full/dual_data_runtime.py',
        'src/pose_control/qwen21_full/native_hybrid_assets.py','src/pose_control/qwen21_full/native_hybrid_cache.py',
        'scripts/data/build_qwen21_native_dual_cache.py','scripts/run/run_qwen21_native_full_v5.py',
        'tests/test_qwen21_native_v5_training.py','tests/test_qwen21_native_full_v5.py',
        'tests/test_qwen21_native_hybrid_v41.py','scripts/audit/debug_qwen21_v5_checkpoint.py',
        'docs/Qwen21_V5固定样本训练使用说明_20261026.md',
        'docs/Qwen21_V5固定样本结果报告工具_20261026.md',
        'scripts/reports/render_qwen21_fixedpair_report.cjs'))
    records={}
    for name in sorted(files):
        data=(ROOT/name).read_bytes()
        if name.endswith('.py'):ast.parse(data,filename=name)
        records[name]={'sha256':sha(data),'bytes':len(data),'ast_parsed':name.endswith('.py')}
    a.output.mkdir(parents=True,exist_ok=False)
    destination=a.output/'source.tar.gz'
    with tarfile.open(destination,'w:gz') as archive:
        for name in sorted(files):archive.add(ROOT/name,arcname=name,recursive=False)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        'files':records,'protected_neural_modules':protected,'protected_sha_unchanged':True,
        'archive':{'path':str(destination),'sha256':sha(destination.read_bytes())},
        'weights_in_archive':False,'media_in_archive':False,'training_completion_not_inferred':True}
    with (a.output/'manifest.json').open('x') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps({'files':len(records),'protected_modules':len(protected),'protected_sha_unchanged':True,'archive':report['archive']}))


if __name__=='__main__':main()
