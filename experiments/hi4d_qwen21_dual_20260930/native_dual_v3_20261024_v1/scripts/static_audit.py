"""Stdlib syntax/AST/byte provenance audit only. Never imports project modules/tests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import py_compile
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-name', default='static_architecture_v1')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[4]
    run = Path(__file__).resolve().parents[1]
    report = run/'report'
    report.mkdir(exist_ok=True)
    baseline = json.loads((run/'source/baseline.json').read_text())
    # Explicit source inventory remains useful after committing the feature.
    paths = list((root/'src/pose_control/qwen21_full').glob('*dual*.py'))
    for directory in ('scripts/data', 'scripts/run', 'scripts/train'):
        paths += list((root/directory).glob('*native_dual*.py'))
    paths.append(root/'tests/test_qwen21_native_dual_v3.py')
    names = {str(p.relative_to(root)).replace('\\', '/') for p in paths}
    names.update(str(p.relative_to(root)).replace('\\', '/') for p in (run/'scripts').glob('*.py'))
    files = sorted(n for n in names if n.endswith('.py'))
    bytecode = report/(args.report_name+'_bytecode')
    bytecode.mkdir(exist_ok=False)
    trees = {}
    for name in files:
        path = root/name
        trees[name] = ast.parse(path.read_text(encoding='utf8'), filename=name)
        py_compile.compile(str(path), cfile=str(bytecode/(hashlib.sha256(name.encode()).hexdigest()[:16]+'.pyc')), doraise=True)
    unchanged = {}
    for name, digest in baseline['files'].items():
        if name.startswith('src/pose_control/qwen21_full/') and name.endswith('.py'):
            unchanged[name] = hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    assert all(unchanged.values()), {n: v for n, v in unchanged.items() if not v}
    path = 'src/pose_control/qwen21_full/dual_depth_condition.py'
    constants = {n.targets[0].id: n.value for n in trees[path].body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert len(ast.literal_eval(constants['RELATION_CHANNELS'])) == 8
    assert len(ast.literal_eval(constants['INTERACTION_CHANNELS'])) == 12
    runtime = trees['src/pose_control/qwen21_full/native_dual_runtime.py']
    flags = next(n for n in runtime.body if isinstance(n, ast.ClassDef) and n.name == 'DualTrainability')
    assert sum(isinstance(n, ast.AnnAssign) for n in flags.body) == 12
    identity = trees['src/pose_control/qwen21_full/native_dual_identity.py']
    roi_sizes = [ast.literal_eval(n.args[-1]) for n in ast.walk(identity)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_roi']
    assert sorted(roi_sizes) == [(4, 4), (8, 4)]
    assert sum(h*w for h, w in roi_sizes) == 48
    relations = trees['src/pose_control/qwen21_full/native_dual_depth.py']
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'randn'
               and [ast.literal_eval(x) for x in n.args] == [28, 256] for n in ast.walk(relations))
    declarations = ast.parse((root/'src/pose_control/qwen21_full/depth_condition_v2.py').read_text(encoding='utf8'))
    legacy_constants = {n.targets[0].id: n.value for n in declarations.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert len(ast.literal_eval(legacy_constants['DETAIL_CHANNELS'])) == 23
    gate_count = sum(ast.literal_eval(n.value.generators[0].iter.args[0]) if isinstance(n, ast.Starred) else 1
                     for n in legacy_constants['GATE_CHANNELS'].elts)
    assert gate_count == 29
    dual_type = next(n for n in trees[path].body if isinstance(n, ast.ClassDef) and n.name == 'DualDepthConditionV3')
    fields = sum(isinstance(n, ast.AnnAssign) for n in dual_type.body)
    builder = trees['src/pose_control/qwen21_full/dual_depth_geometry.py']
    constructor = next(n for n in ast.walk(builder) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                       and n.func.id == 'DualDepthConditionV3')
    assert fields == sum(2 if isinstance(n, ast.Starred) else 1 for n in constructor.args)
    def call(tree, attr):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr)
    original = ast.parse((root/'scripts/train/train_qwen21_native_depth.py').read_text(encoding='utf8'))
    current = trees['scripts/train/train_qwen21_native_dual.py']
    optimizer_same = ast.dump(call(original, 'AdamW')) == ast.dump(call(current, 'AdamW'))
    assert optimizer_same
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'load_state_dict':
                assert any(k.arg == 'strict' and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords), name
    result = {'status': 'architecture_wired_not_runtime_accepted', 'remote_baseline_head': baseline['head'],
        'compiled_python_files': files, 'legacy_python_byte_identical': unchanged,
        'static_interfaces': {'relation_raster': 8, 'person_detail': 23, 'depth_gate': 29,
            'interaction_context': 12, 'identity_context': 14, 'pair_numeric': 9,
            'relation_summary_tokens': 28, 'identity_tokens_per_person': 48, 'native_width': 512,
            'parameter_groups': 12, 'qwen_new_tokens': 0},
        'AdamW_AST_unchanged': optimizer_same, 'all_state_dict_loads_explicit_strict_true': True,
        'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},
        'not_executed': ['tests', 'project_imports', 'models', 'forward', 'backward', 'optimizers',
                         'training', 'rendering', 'cache_export', 'weight_migration', 'package_installation']}
    with (report/(args.report_name+'.json')).open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'compiled': len(files), 'legacy_unchanged': len(unchanged), 'status': result['status']}))


if __name__ == '__main__':
    main()
