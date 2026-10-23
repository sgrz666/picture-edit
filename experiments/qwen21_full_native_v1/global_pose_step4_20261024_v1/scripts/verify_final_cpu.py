"""封存测试输出，并比较原训练 loop / AdamW / loss 的 AST。"""
from pathlib import Path
import ast
import json
import subprocess
import sys

repo = Path('/home/shangguanrz/project/pic-edit')
run = Path('/data/shangguanrz/pic-edit-experiments/qwen21_full_native_v1/global_pose_step4_20261024_v1')
source = run / 'source/baseline_source'
source.mkdir(exist_ok=True)
import tarfile
with tarfile.open(run / 'source_20261024.tar.gz') as tar:
    for path in ('scripts/train/train_qwen21_native_depth.py', 'src/pose_control/qwen21_full/native_depth_runtime.py'):
        dest = source / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(tar.extractfile(path).read())

def tree(root, path):
    return ast.parse((root / path).read_text())

def pick(node, predicate):
    return next(x for x in ast.walk(node) if predicate(x))

def same(predicate, path):
    before, after = (pick(tree(root, path), predicate) for root in (source, repo))
    return ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)

training = 'scripts/train/train_qwen21_native_depth.py'
checks = {
    'training_loop_ast_equal': same(lambda x: isinstance(x, ast.For) and isinstance(x.target, ast.Name) and x.target.id == 'step', training),
    'optimizer_ast_equal': same(lambda x: isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr == 'AdamW', training),
    'loss_ast_equal': same(lambda x: isinstance(x, ast.FunctionDef) and x.name == 'native_velocity_loss', 'src/pose_control/qwen21_full/native_depth_runtime.py')}
assert all(checks.values()), checks
targets = ['test_qwen21_global_pose.py', 'test_qwen21_pose_cache.py', 'test_qwen21_full_preservation.py',
           'test_qwen21_native_geometry.py', 'test_qwen21_native_depth_runtime.py', 'test_qwen21_native_depth.py',
           'test_qwen21_native_body.py', 'test_canvas_contract.py', 'test_geometry_pair_dataset_v2.py']
command = [sys.executable, '-m', 'pytest', *['tests/' + p for p in targets], '-q', '--tb=short',
           '--junitxml', str(run / 'report/unit_tests_final_v1.xml')]
result = subprocess.run(command, cwd=repo, text=True, capture_output=True)
with (run / 'report/unit_tests_final_v1.txt').open('x') as stream:
    stream.write(result.stdout + result.stderr)
with (run / 'report/training_policy_audit_v1.json').open('x') as stream:
    json.dump({**checks, 'pytest_exit_code': result.returncode, 'command': command, 'no_optimizer_run': True}, stream, indent=2)
print(result.stdout)
print(result.stderr)
print(json.dumps(checks))
raise SystemExit(result.returncode)
