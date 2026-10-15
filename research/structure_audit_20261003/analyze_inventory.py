"""Derive reviewable, conservative classifications from the read-only snapshot."""
import ast
import csv
import json
import hashlib
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / 'remote_inventory.json').read_text(encoding='utf-8-sig'))
root = data['root'] + '/'
sources = data['sources']
delta_path = HERE / 'concurrent_delta.json'
if delta_path.exists():
    delta = json.loads(delta_path.read_text(encoding='utf-8-sig'))
    merged = {source['path']: source for source in sources}
    for update in delta['changed_sources'] + delta['new_sources']:
        if 'text' not in update:
            continue
        body = update['text']
        source = {'path': update['path'], 'text': body, 'bytes': len(body.encode('utf-8')), 'lines': len(body.splitlines()), 'sha256': update.get('current_sha256', hashlib.sha256(body.encode('utf-8')).hexdigest()), 'checked_at': delta['checked_at']}
        if update['path'].endswith('.py'):
            try:
                tree = ast.parse(body)
                source['symbols'] = [{'name': n.name, 'line': n.lineno, 'kind': type(n).__name__} for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))]
            except SyntaxError as error:
                source['parse_error'] = str(error)
        merged[update['path']] = source
    sources = sorted(merged.values(), key=lambda source: source['path'])
by_path = {s['path']: s for s in sources}

def proposal(path):
    if not path.startswith(root):
        if '/nihaochen/docs/' in path:
            return '项目外报告与证据', 'reports/（建立索引后迁入）'
        name = Path(path).name
        if name.startswith(('run_', 'watch_')) or name.endswith('.sh'):
            return '项目外运行编排', 'scripts/run/（逐项核验）'
        if name.startswith(('generate_', 'render_', 'draw_', 'build_16', 'build_full', 'quick_grid', 'summarize_')):
            return '项目外报告与可视化工具', 'scripts/reports/（逐项核验）'
        if name.startswith(('collect_', 'curate_', 'commons_', 'build_interaction', 'compose_conditions')):
            return '项目外数据准备工具', 'scripts/data/（逐项核验）'
        return '项目外工具或临时诊断', 'scripts/diagnostics/ 或 archive/scratch/（逐项核验）'
    rel = path[len(root):]
    name = Path(rel).name
    if rel.startswith('src/'):
        return '核心源码（保留）', rel
    if rel.startswith('models/'):
        return '模型配套运行实现（保留，不能按权重缓存删除）', rel
    if rel.startswith('model-qwen-edit/'):
        return '旧 Qwen Edit 20B 资产与独立下载/启动入口', '旧模型整组归档至 /data，保留兼容路径'
    if rel.startswith('experiments/16-test/'):
        return '16 组横评可复现入口', 'scripts/benchmarks/16_test/（路径及结果配置解耦后）'
    if rel.startswith('tests/'):
        return '回归测试（保留）', rel
    if rel.startswith('third_party/') or rel.startswith('vram_lab/'):
        return '依赖或可执行 Python 包（保留）', rel
    if rel.startswith('add/'):
        if any(k in name for k in ['_before_', '_backup_']):
            return '历史备份候选', 'archive/data_pipeline/（核验后）'
        return '独立数据制备工具链（保留）', rel
    if rel.startswith('docs/'):
        return '架构、运行说明或历史计划', rel
    if rel.startswith('research/') and not name.endswith('.py'):
        return '研究资料或审计证据', rel
    if name in {'README.md', 'pytest.ini', 'THIRD_PARTY_NOTICES.md', 'requirements_lock.txt', 'requirements_qwen21.txt', 'system_environment_lock.md'}:
        return '项目说明、依赖或环境入口（保留）', rel
    if name == 'test_control_signals.py':
        return '生产依赖：合成控制生成工具（保留）', rel + '（以后提取到 src/controls/）'
    if name.startswith('train_'):
        return '训练入口', 'scripts/train/' + name
    if name.startswith(('eval_', 'multi_stage_eval')):
        return '评测入口', 'scripts/eval/' + name
    if name.startswith(('audit_', 'check_', 'verify_', 'inventory_', 'probe_', 'sample_')):
        return '审计、验证或诊断入口', 'scripts/audit/' + name
    if name.startswith(('write_', 'plot_', 'visualize_', 'render_', 'draw_', 'generate_')) or ('build_' in name and any(k in name for k in ['docx', 'panel', 'report'])):
        return '报告或可视化生成', 'scripts/reports/' + name
    if name.startswith(('precompute_', 'prepare_', 'split_', 'build_', 'fill_', 'finalize_', 'create_')):
        return '数据、条件或实验收尾入口', 'scripts/data/' + name
    if name.startswith(('inherit_', 'run_qwen21_weight_inheritance')):
        return '检查点迁移工具', 'scripts/weights/' + name
    return '实验编排、运行或辅助工具', 'scripts/run/' + name

tracked = set(data['git_tracked']['stdout'].splitlines())
status_lines = delta['git_status'].splitlines() if delta_path.exists() else data['git_status']['stdout'].splitlines()
statuses = {line[3:]: line[:2] for line in status_lines}

def git_state(path):
    if not path.startswith(root):
        return '项目仓库外'
    relative = path[len(root):]
    if relative in statuses:
        return statuses[relative]
    if relative in tracked:
        return '已跟踪，未列出改动'
    if any(relative.startswith(name) for name, state in statuses.items() if state == '??' and name.endswith('/')):
        return '??'
    if relative.startswith(('add/', 'models/', 'model-qwen-edit/', 'experiments/')):
        return '未跟踪，目录被忽略'
    return '未在跟踪清单，另核 ignore'

rows = []
for source in sources:
    name = Path(source['path']).name
    stem = Path(name).stem
    references = []
    for other in sources:
        if other['path'] != source['path'] and (name in other['text'] or (source['path'].endswith('.py') and re.search(r'\b' + re.escape(stem) + r'\b', other['text']))):
            references.append(other['path'])
    description = ''
    if name.endswith('.py'):
        try:
            description = (ast.get_docstring(ast.parse(source['text'])) or '').split('\n')[0]
        except SyntaxError:
            pass
    if not description:
        description = ', '.join(s['name'] for s in source.get('symbols', [])[:5])
    kind, target = proposal(source['path'])
    rows.append({'path': source['path'], 'git_state': git_state(source['path']), 'classification': kind, 'suggested_destination': target, 'lines': source['lines'], 'purpose_hint': description, 'static_reference_file_count': len(references), 'referenced_by': '; '.join(references), 'root_derivation_lines': '; '.join(f'{i}: {line.strip()}' for i, line in enumerate(source['text'].splitlines(), 1) if '__file__' in line or 'sys.path' in line), 'sha256': source['sha256']})

with (HERE / 'file_catalog.csv').open('w', newline='', encoding='utf-8-sig') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

root_entries = [r for r in rows if r['path'].startswith(root) and '/' not in r['path'][len(root):] and r['path'].endswith(('.py', '.sh'))]
core_pairs = []
for path, source in by_path.items():
    if path.startswith(root + 'src/pose_control/v6/'):
        copied_path = path.replace('/src/pose_control/v6/', '/src/pose_control/qwen21_full/')
        if copied_path in by_path:
            core_pairs.append({'relative_path': path.split('/v6/')[1], 'same_sha256': source['sha256'] == by_path[copied_path]['sha256']})

def refs(term):
    return [{'path': s['path'], 'lines': [i for i, line in enumerate(s['text'].splitlines(), 1) if term in line]} for s in sources if term in s['text']]

checkpoints = [a for a in data['artifacts'] if 'inode' in a]
inodes = {}
for checkpoint in checkpoints:
    inodes.setdefault((checkpoint['device'], checkpoint['inode']), []).append(checkpoint['path'])
derived = {
    'source_file_count': len(sources), 'root_executable_count': len(root_entries),
    'source_syntax_errors': [{'path': s['path'], 'error': s['parse_error']} for s in sources if 'parse_error' in s],
    'core_copy_pairs': core_pairs,
    'checkpoint_file_count': len(checkpoints),
    'checkpoint_apparent_total_gib': sum(a['bytes'] for a in checkpoints) / 2**30,
    'checkpoint_unique_blocks_gib': sum(next(a['blocks_bytes'] for a in checkpoints if (a['device'], a['inode']) == key) for key in inodes) / 2**30,
    'checkpoint_hardlink_groups': [paths for paths in inodes.values() if len(paths) > 1],
    'references': {term: refs(term) for term in ['model-qwen-edit', 'iper_adapter_finetune_v1/condition_cache', 'qwen21_hi4d_controls', 'test_control_signals', 'train_iper_v6_native', '/home/shangguanrz/nihaochen', '/home/shangguanrz/project/pic-edit', 'add/outputs/3dpw_assets_all_v1', '3dpw_single_final_v2', 'TikTok_v3', 'native_interaction_v4_2']},
    'root_entries': root_entries,
}
(HERE / 'derived_findings.json').write_text(json.dumps(derived, ensure_ascii=False, indent=2), encoding='utf-8')

lines = ['# 逐文件用途与归类索引', '', '基于 2026-10-03 的 SSH 只读快照。分类是建议，未移动任何文件。静态引用为代码或文档文本命中，不能据零命中断言文件无用。详细引用、根目录定位代码和 SHA-256 见同目录 CSV。', '', '| 原路径 | 类型 | 建议归属 | 行数 | 用途线索 | 静态引用文件数 |', '|---|---|---|---:|---|---:|']
for row in rows:
    values = [row['path'], row['classification'], row['suggested_destination'], str(row['lines']), row['purpose_hint'], str(row['static_reference_file_count'])]
    lines.append('| ' + ' | '.join(v.replace('|', '\\|').replace('\n', ' ') for v in values) + ' |')
(HERE / 'FILE_CATALOG.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(json.dumps({k: v for k, v in derived.items() if k not in {'references', 'root_entries', 'core_copy_pairs'}}, ensure_ascii=False, indent=2))
print('identical_core_copy_pairs:', sum(p['same_sha256'] for p in core_pairs), '/', len(core_pairs))
for term, references in derived['references'].items():
    print(term, '=>', len(references), 'source/doc files')
print('root executable names:', ', '.join(Path(r['path']).name for r in root_entries))
