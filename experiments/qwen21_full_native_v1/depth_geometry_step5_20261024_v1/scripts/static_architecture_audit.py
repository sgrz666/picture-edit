"""仅stdlib：语法编译、AST/来源核对；不import项目，不执行模型或测试。"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--report-name', default='static_architecture_final')
    args = p.parse_args()
    root, run = args.repo.resolve(), args.run.resolve()
    before = run / 'source/remote'
    report = run / 'report'
    if Path(args.report_name).name != args.report_name:
        raise ValueError('report-name must be a basename')
    bytecode = report / (args.report_name + '_bytecode')
    bytecode.mkdir(parents=True, exist_ok=True)
    source_files = sorted(set(
        subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
        + subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=root, text=True).splitlines()))
    source_files = [s for s in source_files if s.endswith('.py')]
    for name in source_files:
        text = (root / name).read_text(encoding='utf8')
        ast.parse(text, filename=name)
        cache_name = hashlib.sha256(name.encode('utf8')).hexdigest()[:16] + '.pyc'
        py_compile.compile(str(root / name), cfile=str(bytecode / cache_name), doraise=True)
    def tree(base, name):
        return ast.parse((base / name).read_text(encoding='utf8'))
    def pick(t, predicate):
        return next(n for n in ast.walk(t) if predicate(n))
    def same(path, predicate):
        return ast.dump(pick(tree(before, path), predicate)) == ast.dump(pick(tree(root, path), predicate))
    checks = {}
    for name in ('native_depth.py', 'depth_condition.py', 'native_geometry.py', 'native_body.py', 'body_pose.py'):
        path = 'src/pose_control/qwen21_full/' + name
        checks[path + '_byte_identical'] = (before / path).read_bytes() == (root / path).read_bytes()
    checks['AdamW_AST_unchanged'] = same('scripts/train/train_qwen21_native_depth.py',
        lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'AdamW')
    checks['velocity_loss_AST_unchanged'] = same('src/pose_control/qwen21_full/native_depth_runtime.py',
        lambda n: isinstance(n, ast.FunctionDef) and n.name == 'native_velocity_loss')
    loops = []
    for base in (before, root):
        node = deepcopy(pick(tree(base, 'scripts/train/train_qwen21_native_depth.py'),
             lambda n: isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'step'))
        node.body = [n for n in node.body if not (isinstance(n, ast.Assign) and any(
             isinstance(t, ast.Name) and t.id in ('depth_grad', 'depth_conv_grad') for t in n.targets))]
        loops.append(ast.dump(node))
    checks['training_loop_AST_unchanged_except_frozen_gradient_logging'] = loops[0] == loops[1]
    condition = tree(root, 'src/pose_control/qwen21_full/depth_condition_v2.py')
    constants = {n.targets[0].id: n.value for n in condition.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    detail_channels = ast.literal_eval(constants['DETAIL_CHANNELS'])
    gate_count = sum(ast.literal_eval(n.value.generators[0].iter.args[0]) if isinstance(n, ast.Starred) else 1
                     for n in constants['GATE_CHANNELS'].elts)
    checks['detail_channels_23'] = len(detail_channels) == 23
    checks['gate_context_29'] = gate_count == 29
    assert all(checks.values()), checks
    result = {'status': 'static_only_runtime_acceptance_not_run', 'checks': checks,
              'compiled_python_files': source_files, 'detail_channels': detail_channels,
              'declared_shape_chain': ['2→CHAMP→256@1/16', '23→16→32→64→128→256@1/16',
                    'concat512→mixer256', 'gate:(64+64+32)160→128→512', 'zero_projection:256→512'],
              'not_executed': ['tests', 'project_imports', 'model_forward', 'model_backward', 'optimizer',
                    'training', 'rendering', 'cache_export', 'checkpoint_load_or_migration'],
              'source_sha256': {s: hashlib.sha256((root / s).read_bytes()).hexdigest() for s in source_files}}
    with (report / (args.report_name + '.json')).open('x', encoding='utf8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'compiled': len(source_files), 'checks': checks}))


if __name__ == '__main__':
    main()
