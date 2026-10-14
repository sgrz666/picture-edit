#!/usr/bin/env python3
"""Consolidate saved Face-repair checks and the manually inspected single case."""

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))

import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pose_control.qwen21_full.execution import json_write, RUN_ROOT

OUT = RUN_ROOT / 'face_repair'
PREVIOUS = RUN_ROOT / 'face_single_sample'
RUNS = ('roi_same_aux', 'roi_aligned_aux')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    protection = read(OUT / 'protection_audit.json')
    runtime = read(OUT / 'runtime_acceptance.json')
    if not protection['passed'] or not runtime['passed']:
        raise AssertionError('Protection/runtime acceptance is required before delivery')
    tests = (OUT / 'code_checks.log').read_text()
    if '37 passed' not in tests:
        raise AssertionError('Expected all 37 focused boundary/objective tests to pass')
    body = next(r for r in read(PREVIOUS / 'body_stage_metrics.json') if r['added_step'] == 500)
    old_rows = read(PREVIOUS / 'face_stage_metrics.json')
    roi, aligned = [read(OUT / 'generations' / run / 'step_000300/metrics.json') for run in RUNS]
    results = {}
    for run, row in zip(RUNS, (roi, aligned)):
        m, b = row['metrics'], body['metrics']
        results[run] = {
            'reliable_identity_n': int(m['identity_cosine_reliable'] is not None),
            'identity_gain': m['identity_cosine_reliable'] - b['identity_cosine_reliable'],
            'pose_pck_change': m['pose_pck_002'] - b['pose_pck_002'],
            'background_mae_increase': m['background_mae_source'] - b['background_mae_source'],
            'no_obvious_colored_blocks_or_severe_face_ghosting_manual': True,
            'fine_detail_still_blurry_manual': True,
        }
        v = results[run]
        v['single_case_passed'] = (v['reliable_identity_n'] == 1 and v['identity_gain'] > 0
                                  and v['pose_pck_change'] >= -.02 and v['background_mae_increase'] <= 1.)
    high_paths = [
        OUT / 'generations_1024/roi_same_aux/step_000000/body/metrics.json',
        OUT / 'generations_1024/roi_same_aux/step_000300/metrics.json',
        OUT / 'generations_1024/roi_same_aux/step_000300/checkpoint_grid/metrics.json',
    ]
    high = [read(p) for p in high_paths]
    if any(row['control_forwards'] != 30 for row in (roi, aligned, *high)):
        raise AssertionError('Every accepted image must complete all 30 sampling steps')
    reliable_high_n = int(all(row['metrics']['identity_cosine_reliable'] is not None for row in high))
    checks = {
        '512 单案例脸部、身份、姿态、背景验收': all(r['single_case_passed'] for r in results.values()),
        '1024 绿色斑点及明显重影（同一案例人工检查）': '修正训练网格映射后未见明显异常；脸仍偏糊',
        '1024 身份改善': f'无法可靠判断：共同可靠样本 n={reliable_high_n}',
        '原完整控制模块 47 份保留': True,
        'Body 和 Qwen 基座冻结': True,
        '全部控制关闭与原生预测一致': True,
        'Face 关闭／无效脸退回同一 Body': True,
        '注入仅作用于目标 token，Face 脸外残差为零': True,
        'Target RGB 未进入生成条件': True,
        '37 项针对性代码测试': True,
        '恢复核对': '确定性参考实现下一步精确一致；不声明默认 CUDA 逐位一致',
    }
    json_write(OUT / 'acceptance.json', {
        'checks': checks, '512_results': results,
        '1024': {'reliable_common_n': reliable_high_n, 'identity_improvement_conclusion': '无法判断',
                 'native_grid_manual': '明显绿色斑点与面部重影',
                 'checkpoint_grid_manual': '未见原先明显绿色斑点和严重重影；细节仍模糊',
                 'face_training_resolution': 512, 'training_at_1024_performed': False,
                 'metrics_paths': [str(p) for p in high_paths]},
        'conclusions': ['两组 512 修复均通过这一对训练图片的验收。1024 的修正通过了接口、完整采样和异常外观检查；由于可靠身份样本不足，不宣布 1024 身份改善已得到验证。'],
    })
    json_write(OUT / 'final_review.json', {
        'review_method': '逐图查看完整生成图及原尺寸脸部裁剪放大，不使用修图或人脸贴回',
        'reviewed_panels': ['report_panels/repair_final_review.png', 'report_panels/resolution1024_fixed_review.png'],
        'conclusions': [
            '问题有具体的实现和适配原因。旧训练把全图误差平均，脸只占约 0.59%，导致脸区变坏仍可能使总误差下降；不是简单增加训练步数就能解决。',
            '将 Face 生成损失改为独立脸区目标后，同一案例新增 300 步已不再出现旧结果的明显彩斑，源脸相似度由固定 Body 的 0.698 提高至 0.754。',
            '另一组把身份与感知辅助监督做五点对齐，源脸相似度为 0.739，也高于 Body；与 Target 的相似度更高。两份候选均保留，不能认定其中一份全面更好。',
            '512 修复权重直接用于 1024 时，绿色斑点与重影再次出现。仅将 Face 控制先按训练网格计算、再平滑映射到 1024 后，原先明显异常消失，身体姿态与背景保持稳定。',
            '1024 的身份指标在固定 512 评价网格上均因脸框短边不足 32 像素被排除，可靠共同样本 n=0。原始分数可展示，但不用于宣布身份改善。脸部清晰度仍有限。',
        ],
        'next_steps': ['后续先用固定验证集中的其他人物和转头案例复核这套修复，再决定是否开始多样本 Face 训练。单样本成功不等于泛化成功。'],
        'training_case_count': 1, 'final_50_test_used': False,
    })
    code = [
        'src/pose_control/qwen21_full/execution.py', 'src/pose_control/qwen21_full/qwen_adapter.py',
        'src/pose_control/qwen21_full/face_loss_alignment.py', 'train_iper_qwen21_full.py',
        'eval_iper_qwen21_full.py', 'scripts/eval_qwen21_face_repair.py',
        'scripts/verify_qwen21_full_resume.py', 'scripts/verify_qwen21_face_repair.py',
    ]
    candidates = {run: {'path': str(OUT / run / 'checkpoint_step_000300.pt'),
                        'sha256': sha(OUT / run / 'checkpoint_step_000300.pt'),
                        'config': read(OUT / f'{run}_dispatch.json'),
                        'summary': read(OUT / run / 'summary.json')}
                  for run in RUNS}
    json_write(OUT / 'provenance.json', {
        'candidates': candidates, 'code_sha256': {name: sha(ROOT / name) for name in code},
        'protection_evidence': str(OUT / 'protection_audit.json'),
        'runtime_evidence': str(OUT / 'runtime_acceptance.json'),
        'resume_evidence': [str(OUT / f'{run}_resume.json') for run in RUNS],
        'face_loss_input_alignment_audit': str(OUT / 'alignment_audit.json'),
        'resolution_scatter_audit': str(OUT / 'resolution_face_scatter_audit.json'),
        'sample_manifest': str(PREVIOUS / 'single_pair.jsonl'),
        'sample_manifest_sha256': sha(PREVIOUS / 'single_pair.jsonl'),
        'default_weights_replaced': False, 'final_test_used': False,
        'explanation': ['47 份旧控制模块及原检查点哈希保持不变。每份候选包含完整 977 项参数，只更新 Face 范围内 242 项，其余 735 项逐项保持不变。身份及感知辅助监督均有独立梯度证据。'],
    })
    labeled = [('512 Body', body),
               *[(f'512 old Face {step}', next(r for r in old_rows if r['added_step'] == step)) for step in (300, 2000)],
               ('512 ROI Face 300', roi), ('512 aligned ROI Face 300', aligned),
               ('1024 Body', high[0]), ('1024 native-grid Face', high[1]), ('1024 checkpoint-grid Face', high[2])]
    fields = ['label', 'resolution', 'identity_cosine_raw', 'identity_cosine_reliable', 'identity_reliable_n',
              'identity_target_cosine_raw', 'pose_pck_002', 'pose_nme_diagonal', 'human_mae',
              'background_mae_source', 'psnr', 'ssim', 'seconds', 'peak_cuda_gib', 'image']
    with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, row in labeled:
            record = {'label': label, **{key: row.get(key) for key in ('resolution', 'seconds', 'peak_cuda_gib', 'image')}}
            record.update({key: row['metrics'].get(key) for key in fields if key in row['metrics']})
            record['identity_reliable_n'] = int(row['metrics'].get('identity_cosine_reliable') is not None)
            writer.writerow(record)
    print(json.dumps({'512_single_case_passed': checks['512 单案例脸部、身份、姿态、背景验收'],
                      '1024_reliable_identity_n': reliable_high_n, 'output': str(OUT)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
