#!/usr/bin/env python3
"""Create final plots and a source-backed Chinese readout from committed pilot artifacts."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from core import DWPOSE_TO_COCO, read_json, read_jsonl, write_json


def fmt(value, digits=4):
    return f'{value:.{digits}f}' if value is not None else '无有效检测'


def response_diagnostics(config, inference):
    sys.path.insert(0, config['project_root'])
    from src.evaluation.joint_pilot_metrics import PoseMetric
    detector = PoseMetric(Path(config['project_root']) / 'models')
    rows = read_jsonl(inference / 'outputs.jsonl')
    grouped = defaultdict(dict)
    for row in rows:
        if row['case'] in ('normal_00', 'swap_pose', 'blank_pose'):
            grouped[row['method']][row['case']] = row
    output = []
    for method, cases in grouped.items():
        reference = cases['normal_00']
        with Image.open(reference['image']) as image:
            base_detected = detector.detect(image)
        base_meta = read_json(reference['cache_row']['target_pose_metadata'])
        diagonal = np.hypot(*config['canvas_wh'])
        for case in ('swap_pose', 'blank_pose'):
            if case not in cases:
                continue
            row = cases[case]
            with Image.open(row['image']) as image:
                detected = detector.detect(image)
                rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
            with Image.open(reference['image']) as image:
                base_rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
            result = {'method': method, 'case': case, 'source_A_unchanged':
                      row['cache_row']['condition_sha256'][0] == reference['cache_row']['condition_sha256'][0],
                      'prompt_unchanged': row['cache_row']['prompt'] == reference['cache_row']['prompt'],
                      'seed_unchanged': row['seed'] == reference['seed'],
                      'rgb_mae_0_255': float(np.abs(rgb-base_rgb).mean()),
                      'output_pose_change_diagonal': None, 'guide_pose_change_diagonal': None,
                      'matched_output_joints': 0}
            if base_detected is not None and detected is not None:
                good = (base_detected[1] > .3) & (detected[1] > .3)
                if good.any():
                    result['output_pose_change_diagonal'] = float(np.linalg.norm(
                        base_detected[0][good]-detected[0][good], axis=1).mean()/diagonal)
                    result['matched_output_joints'] = int(good.sum())
            if case == 'swap_pose':
                alternate = read_json(row['cache_row']['target_pose_metadata'])
                good = np.asarray(base_meta['valid'])[DWPOSE_TO_COCO] & np.asarray(alternate['valid'])[DWPOSE_TO_COCO]
                a, b = np.asarray(base_meta['keypoints_xy'])[DWPOSE_TO_COCO], np.asarray(alternate['keypoints_xy'])[DWPOSE_TO_COCO]
                if good.any():
                    result['guide_pose_change_diagonal'] = float(np.linalg.norm(a[good]-b[good], axis=1).mean()/diagonal)
            output.append(result)
    return output


def build(config, tag):
    root = Path(config['workspace_root'])
    training = root / 'output' / f'pilot_{tag}'
    inference = root / 'output' / f'comparison_{tag}'
    report = root / 'report' / f'pilot_report_{tag}'
    metrics = read_json(report / 'metrics.json')
    updates = []
    for path in sorted(training.glob('updates_*.jsonl')):
        updates.extend(read_jsonl(path))
    if [r['step'] for r in updates] != list(range(1, config['steps']+1)):
        raise ValueError('The committed logs do not cover exactly 300 consecutive updates')
    resources = []
    first_stage = next(row for row in read_jsonl(root / 'output' / f'campaign_{tag}' / 'events.jsonl')
                       if row['event'] == 'started')
    command = first_stage['command']
    cache_path = Path(command[command.index('--cache') + 1])
    paths = [*training.glob('resources_*.json'), inference / 'resources.json', report / 'resources.json',
             cache_path / 'resources.json',
             root / 'output' / f'preflight_{tag}' / 'resources_000000_000001.json',
             root / 'output' / f'probe_inference_{tag}' / 'resources.json']
    for path in paths:
        record = read_json(path)
        resources.append({'stage': str(path.relative_to(root)), **{k: record.get(k) for k in
                          ['pid', 'error', 'violation', 'peak_cuda_allocated_mib', 'peak_cuda_reserved_mib',
                           'peak_host_gpu_used_mib', 'min_gpu_free_mib', 'peak_process_rss_mib', 'min_ram_available_mib']}})
    response = response_diagnostics(config, inference)
    write_json(report / 'pose_response.json', response)
    write_json(report / 'resource_summary.json', resources)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    steps = np.asarray([r['step'] for r in updates])
    losses = np.asarray([r['loss'] for r in updates])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    axes[0].plot(steps, losses, lw=.7, alpha=.5, label='Per-update loss')
    axes[0].plot(steps[19:], np.convolve(losses, np.ones(20)/20, 'valid'), label='20-update mean')
    axes[0].axvline(100, color='gray', ls='--', lw=1, label='Checkpoint / process restart')
    axes[0].set(xlabel='Update', ylabel='Target flow matching MSE', title='Training (random timesteps)')
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, np.asarray([r['cuda_allocated_mib'] for r in updates])/1024)
    axes[1].set(xlabel='Update', ylabel='Process CUDA allocated (GiB)', title='Post-update allocator usage')
    fig.savefig(report / 'training_curves.png', dpi=160)
    plt.close(fig)
    resume = read_json(training / 'resume_000100.json')
    reload = read_json(inference / 'native_lora_reload.json')
    checkpoint = read_json(training / 'checkpoint-000300/checkpoint.json')
    lines = ['# Bilibili 双图 Edit LoRA 首轮实测', '',
             '输入为 A RGB、B 原始 DWPose 骨架和统一短指令；监督为 B RGB。基座、Qwen3-VL、VAE 冻结。', '',
             f'完成 {len(updates)} 次更新；32 对训练样本来自 8 个视频，10 对验证样本来自另外 5 个视频。测试集未使用。',
             '画布 384×672；rank/alpha=16，BF16，AdamW，lr=5e-5，前 10 步 warmup，batch=1。', '',
             f'首 20 步平均 loss={losses[:20].mean():.6f}，末 20 步={losses[-20:].mean():.6f}。'
             '随机 timestep 会影响 loss；该趋势不单独证明姿态或身份质量提高。', '',
             '## 对照指标', '',
             'PCK 使用 COCO17 对应关节、图像对角线归一化；未检测到的有效目标关节计为 miss。身份相似度只统计可检测人脸，并单列失败数。', '',
             '| 方法 | 划分 | 图数 | PCK@0.02 | PCK@0.05 | PCK@0.10 | ID cosine | ID有效数 | 姿态失败 | 人脸失败 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in metrics['summary']:
        if row['intervention'] != 'normal':
            continue
        lines.append(f'| {row["method"]} | {row["split"]} | {row["images"]} | '
                     f'{fmt(row["pck_002_diagonal"])} | {fmt(row["pck_005_diagonal"])} | {fmt(row["pck_010_diagonal"])} | '
                     f'{fmt(row["identity_a_cosine"])} | {row["identity_a_cosine_n"]} | '
                     f'{row["pose_detection_failures"]} | {row["generated_face_failures"]} |')
    lines += ['', '## 换骨架诊断', '',
              '固定 A、Prompt、seed。换骨架时 VLM 和 VAE 两条路径同时更换；空白骨架也经过两条路径。'
              '输出发生变化只证明模型对条件敏感，还需结合 PCK 与对照图判断方向是否正确。', '',
              '| 方法 | 条件变化 | RGB MAE (0–255) | 输出关节位移/对角线 | 骨架关节位移/对角线 |',
              '|---|---|---:|---:|---:|']
    for row in response:
        lines.append(f'| {row["method"]} | {row["case"]} | {fmt(row["rgb_mae_0_255"])} | '
                     f'{fmt(row["output_pose_change_diagonal"])} | {fmt(row["guide_pose_change_diagonal"])} |')
    lines += ['', '## 保存、重载与恢复', '',
              f'- 第 100 步恢复：LoRA 权重逐张量一致={resume["lora_exact"]}；优化器状态数={resume["optimizer_states"]}；RNG 已恢复={resume["rng_restored"]}。',
              f'- 最终 LoRA 原生加载：转换到加载 dtype 后逐张量一致={reload["exact_after_dtype_cast"]}，最大权重差={reload["max_absolute_weight_difference"]}。',
              f'- 第 300 步相对初始化的 LoRA L2 变化={checkpoint["lora_delta_l2"]:.6f}。',
              '- 本轮没有做连续 300 步与中断恢复轨迹的逐步 bit-exact 对照；恢复验证范围是权重、优化器、scheduler、RNG 状态和后续真实更新。', '',
              '## 显存与内存', '',
              'NVML 每 2 秒采样，全卡占用包含并行 V5；CUDA allocated/reserved 是本进程 allocator 峰值。RSS 含文件映射页。', '',
              '| 阶段 | 本进程 CUDA reserved GiB | 全卡峰值 GiB | 最小空余显存 GiB | 进程 RSS 峰值 GiB | 最小可用内存 GiB |',
              '|---|---:|---:|---:|---:|---:|']
    for row in resources:
        def gib(key):
            return fmt(row[key]/1024, 2) if row[key] is not None else '—'
        lines.append(f'| {row["stage"]} | {gib("peak_cuda_reserved_mib")} | {gib("peak_host_gpu_used_mib")} | '
                     f'{gib("min_gpu_free_mib")} | {gib("peak_process_rss_mib")} | {gib("min_ram_available_mib")} |')
    lines += ['', '## 产物与使用', '',
              f'- 权重与完整恢复状态：`{training}/checkpoint-000300/`。',
              f'- 原始对照图与推理记录：`{inference}/`。',
              '- 本目录 `metrics.json`、`per_image.jsonl`、`pose_response.json`、`resource_summary.json` 为可复核数字；'
              '`validation_overview.png` 为验证集前 4 对概览，`training_curves.png` 为训练曲线。',
              '- `normal_*_comparison.png` 的列顺序是 A、B 骨架、B RGB、原始 Qwen、LoRA。',
              '- 原始视频可能含逐帧变化的字幕，本轮保留原 RGB 整图监督。小样本结果不代表全量训练或跨人泛化。', '']
    with (report / 'REPORT.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print(report / 'REPORT.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    build(read_json(args.config), args.tag)
