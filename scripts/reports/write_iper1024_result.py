#!/usr/bin/env python3
"""Summarize paired 512/1024 Qwen21 iPER results without mixing pixel scales."""

from __future__ import annotations

# Project root for the categorized entrypoint.
from pathlib import Path as _EntryPath
import sys as _entry_sys
_entry_sys.path.insert(0, str(_EntryPath(__file__).resolve().parents[2]))


import argparse
import json
from pathlib import Path
from statistics import mean


MODES = ('native', 'body', 'body_face', 'body_face_crop')
TABLE_MODES = ('body', 'body_face', 'body_face_crop')
LABELS = {'native': '原生', 'body': 'Body', 'body_face': 'Body＋Face',
          'body_face_crop': 'Body＋Face＋A 脸图'}
FIELDS = (
    ('可靠脸源身份余弦 ↑', 'identity_cosine', 'strict_joint', True),
    ('姿态 PCK@0.02 ↑', 'pose_pck_002', 'all', True),
    ('背景 MAE ↓', 'background_mae_source', 'all', False),
    ('人体 MAE ↓', 'human_mae', 'all', False),
    ('PSNR ↑', 'psnr', 'all', True),
    ('SSIM ↑', 'ssim', 'all', True),
    ('单图耗时／秒 ↓', 'inference_seconds', 'all', False),
    ('峰值显存／GiB ↓', 'peak_cuda_gib', 'all', False),
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-512', type=Path, required=True)
    parser.add_argument('--metrics-1024', type=Path, required=True)
    parser.add_argument('--panels-512', type=Path)
    parser.add_argument('--panels-1024', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    return parser.parse_args()


def table(summary):
    lines = ['| 指标 | ' + ' | '.join(LABELS[mode] for mode in TABLE_MODES) + ' |',
             '|---|' + '|'.join('---:' for _ in TABLE_MODES) + '|']
    for label, key, subset, higher in FIELDS:
        values = [summary['metrics'][key][mode][subset]['mean'] for mode in TABLE_MODES]
        present = [value for value in values if value is not None]
        best = (max(present) if higher else min(present)) if present else None
        cells = []
        for mode, value in zip(TABLE_MODES, values):
            info = summary['metrics'][key][mode][subset]
            if value is None:
                cells.append('—')
                continue
            rendered = f'{value:.3f}'
            if key == 'identity_cosine':
                rendered += f' (n={info["n"]})'
            if best is not None and abs(value - best) < 1e-10:
                rendered = f'**{rendered}**'
            cells.append(rendered)
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    args = arguments()
    small = json.loads(args.metrics_512.read_text())
    large = json.loads(args.metrics_1024.read_text())
    for summary in (small, large):
        if summary['pairs'] != 50 or any(mode not in summary['modes'] for mode in MODES):
            raise ValueError('final report requires the same 50 pairs and four core modes')
    rows_small = json.loads((args.metrics_512.parent / 'per_sample.json').read_text())
    rows_large = json.loads((args.metrics_1024.parent / 'per_sample.json').read_text())
    if [row['stem'] for row in rows_small] != [row['stem'] for row in rows_large]:
        raise ValueError('512 and 1024 sample order differs')
    reliable = []
    for a, b in zip(rows_small, rows_large):
        if all(row['strict_reference_usable'] and all(
            row['variants'][mode]['generated_face_reliable'] and
            row['variants'][mode]['identity_cosine'] is not None for mode in MODES
        ) for row in (a, b)):
            reliable.append((a, b))
    for position, summary in enumerate((small, large)):
        for mode in MODES:
            values = [pair[position]['variants'][mode]['identity_cosine'] for pair in reliable]
            summary['metrics']['identity_cosine'][mode]['strict_joint'] = {
                'mean': mean(values) if values else None, 'n': len(values),
            }
    lines = [
        '# iPER 1024×1024 Qwen21 编辑实验：结果', '',
        '两种尺寸均使用同一批 50 对 iPER 测试样本、30 步、种子 42；对应方案使用相同的文本提示词与 Body／Face 控制设置。'
        '1024 的 Body 和 Face 为在 512 检查点基础上继续训练所得；模型基座冻结。'
        '其中“＋A 脸图”组沿用说明两张图同源的双图提示词，其余组沿用单图提示词；'
        '因此它与单图组的差异同时包含参考图和提示词的变化。', '',
        '## 512 基线', '', table(small), '',
        '## 1024 结果', '', table(large), '',
        '以上 PSNR、SSIM、人体及背景像素误差均由同一评测程序把 RGB 图统一缩至 512 后计算，'
        '用于跨尺寸比较；姿态 PCK 使用归一化位置误差。身份余弦只统计两种尺寸、四组生成脸共同可靠可见的样本，'
        '每项标出有效数 n。表格仅展示三个受控方案；身份指标沿用前版包含原生组的共同可评样本，以保持本次删列前后数值不变。'
        '加粗表示当前表内按指标箭头方向比较的最优值，依据未四舍五入的数据判断；显示值相同不一定代表实际并列。'
        '耗时和峰值显存来自各次完整采样。', '',
        '## 代表性图片', '',
    ]
    if args.panels_512 and args.panels_1024:
        for index in (0, 7, 14, 21, 28, 35, 42, 49):
            matches_small = sorted(args.panels_512.glob(f'{index:03d}_*.png'))
            matches_large = sorted(args.panels_1024.glob(f'{index:03d}_*.png'))
            if matches_small and matches_large:
                lines.extend((f'### 样本 {index:03d}', '',
                              f'512：![512 对照]({matches_small[0]})', '',
                              f'1024：![1024 对照]({matches_large[0]})', ''))
    lines.extend(('## 解释边界', '',
                  '本实验同时提高了完整 A 图参考尺寸和输出尺寸；只凭最终 512／1024 两表，不能把变化进一步拆成单独的输入尺寸或输出尺寸效应。'
                  '旧 512 权重直接用于 1024 的验证结果与 1024 继续训练后的结果须分开讨论。'
                  '1024 使用的姿态、法线等几何条件仍来源于现有 512 资产并插值到目标 token 网格。', ''))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines), encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
