"""Verify a delivered private runtime and seal its code, without starting a job."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import py_compile
import subprocess
import sys
import os

from supervisor import (DEFAULT_RUN, audit_references, file_ref, read_ref, resources,
                        runtime_manifest_path, snapshot_runtime_code, status, verify_runtime_code, write_new_json)

TESTS = ('test_native_spatial_masks.py','test_native_spatial_weights.py','test_native_spatial_audit.py',
         'test_qwen21_native_v5_training.py','test_native_unattended_core.py','test_single_spatial_data_prepare.py',
         'test_single_spatial_supervisor.py','test_qwen21_fixedpair_training_v5.py','test_native_training_throughput.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default=DEFAULT_RUN)
    parser.add_argument('--deployment-manifest', required=True)
    parser.add_argument('--repo-root')
    parser.add_argument('--report-path')
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    repo = Path(args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    scripts = Path(__file__).resolve().parent
    manifest_ref = file_ref(args.deployment_manifest)
    manifest = read_ref(manifest_ref)
    prefix = Path('experiments/bilibili-single-spatial-training')
    for name, expected in manifest.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Deployment path escapes private runtime')
        targets = [repo/path]
        if path.parent == prefix/'scripts':
            targets.append(scripts/path.name)
        if path.parent == prefix/'source/legacy':
            targets.append(root/'source/legacy'/path.name)
        for target in targets:
            if file_ref(target)['sha256'] != expected:
                raise ValueError('Delivered SHA differs: '+str(target))
        if path.suffix == '.py':
            py_compile.compile(str(repo/path), doraise=True)
    subprocess.run(['bash','-n',str(scripts/'launch.sh')],check=True)
    report_path = Path(args.report_path or root/'source/delivery_verification.json')
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf8'))
        audit_references(report)
        verify_runtime_code(report)
        print(json.dumps({'existing_verified_report':file_ref(report_path)}))
        return
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log = root/'source'/('cpu_verification_'+tag+'.log')
    env = os.environ.copy()
    env.pop('NATIVE_V5_TRAINING_ACCEPTANCE_CONFIG',None)
    env.pop('NATIVE_SPATIAL_ABLATION_SCRIPT',None)
    env.pop('NATIVE_TRAINING_SPEED_CUDA',None)
    with log.open('xb') as stream:
        result = subprocess.run([sys.executable,'-m','pytest',*[str(Path('tests')/name) for name in TESTS],'-q'],
                                cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
    print(log.read_text(encoding='utf8'),flush=True)
    if result.returncode:
        raise RuntimeError('CPU regression failed; see '+str(log))
    roots = [repo,scripts,root/'source/legacy']
    code = snapshot_runtime_code(roots,runtime_manifest_path(root,roots))
    spatial = root/'data_audits/spatial_v2.json'
    spatial_report = json.loads(spatial.read_text(encoding='utf8'))
    if spatial_report.get('passed') is not True or spatial_report.get('frames') != 1960:
        raise ValueError('Full 1960-frame spatial audit did not pass')
    reuse_log = root/'source/reuse_verification_20261107_final.log'
    reuse_lines = reuse_log.read_text(encoding='utf8').splitlines()
    reuse_rows = [json.loads(line) for line in reuse_lines if line.startswith('{')]
    sources = [row['completed'] for row in reuse_rows if row.get('event') == 'reused_source_verified']
    pairs = [row['pairs'] for row in reuse_rows if row.get('event') == 'real_native_reuse_verified']
    if (not sources or max(sources) != 1960 or pairs != list(range(1,562)) or
            any('Traceback' in line for line in reuse_lines)):
        raise ValueError('Full historical Source/pair reuse audit is incomplete')
    baseline = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/tiktok-fixedpair-v5_20261025_v1')
    current = status(root)
    report = {'schema':'bilibili_training_flow_delivery_v1','passed':True,'recorded_at':tag,
              'deployment_manifest':manifest_ref,'runtime_code_manifest':code,
              'cpu_test_log':file_ref(log),'cpu_test_summary':log.read_text(encoding='utf8').splitlines()[-1],
              'spatial_audit':file_ref(spatial),'initialization':file_ref(baseline/'initialization/native_v5_init_v2.pt'),
              'reuse_audit':{'log':file_ref(reuse_log),'source_frames':1960,'native_pairs_read':561,'passed':True},
              'model_manifest':file_ref(baseline/'source/qwen_model_files.json'),
              'authorization_ref':file_ref(root/'authorization.json'),'resources':resources('0'),
              'training_started':False,'optimizer_updates':0,'supervisor_status':current['status'],
              'pending':['remaining Native assembly','12156-pair full reader gate','frozen Qwen cache',
                         'per-stage production gradient gate','20000 successful updates','final val/test/export']}
    verify_runtime_code(report)
    print(json.dumps({'delivery_report':write_new_json(report_path,report)},ensure_ascii=False))


if __name__ == '__main__':
    main()
