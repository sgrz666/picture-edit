#!/usr/bin/env python3
"""Render the benchmark report: markdown tables plus figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve, roc_auc_score  # noqa: E402

MODELS = ('glintr100', 'w600k_r50', 'lvface_b', 'adaface_kprpe')
LABELS = {'glintr100': 'glintr100 (ArcFace R100, Glint360K)',
          'w600k_r50': 'w600k_r50 (ArcFace R50, WebFace600K)',
          'lvface_b': 'LVFace-B (ViT-B, Glint360K)',
          'adaface_kprpe': 'AdaFace ViT-B KP-RPE (WebFace4M)'}
SHORT = {'glintr100': 'glintr100', 'w600k_r50': 'w600k_r50',
         'lvface_b': 'LVFace-B', 'adaface_kprpe': 'AdaFace-KPRPE'}
COLORS = {'glintr100': '#4C6EF5', 'w600k_r50': '#12B886',
          'lvface_b': '#F76707', 'adaface_kprpe': '#AE3EC9'}
CONDITION_LABELS = {
    'clean': 'clean (5pt ArcFace align)',
    'align_j5': 'align jitter ±5° / 4% / 3px',
    'align_j10': 'align jitter ±10° / 8% / 6px',
    'blur1.5': 'Gaussian blur σ=1.5',
    'lowres48': 'downsample 48px',
    'lowres32': 'downsample 32px',
    'jpeg20': 'JPEG q=20',
    'occl': 'lower-face occlusion',
    'genlike': 'generation-like composite',
}


def fmt(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, float) and (np.isnan(value)):
        return '—'
    return f'{value:.{digits}f}'


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |',
             '|' + '|'.join('---' for _ in headers) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
    return '\n'.join(lines)


def load_scores(emb_path, pairs_path):
    bundle = np.load(emb_path, allow_pickle=False)
    ids = {key: index for index, key in enumerate(bundle['ids'].tolist())}
    pairs = json.loads(Path(pairs_path).read_text(encoding='utf-8'))['pairs']
    output = {}
    for model in MODELS:
        if model not in bundle.files:
            continue
        embeddings = bundle[model]
        same, diff = [], []
        for pair in pairs:
            if pair['tier'] != 1:
                continue
            a, b = ids.get(pair['a']['key']), ids.get(pair['b']['key'])
            if a is None or b is None:
                continue
            value = float(embeddings[a] @ embeddings[b])
            (same if pair['kind'] == 'same' else diff).append(value)
        output[model] = (np.asarray(same), np.asarray(diff))
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--results-flip', type=Path)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--emb', type=Path, required=True)
    parser.add_argument('--det-sweep', type=Path)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    figures = args.out / 'figures'
    figures.mkdir(exist_ok=True)

    results = json.loads(args.results.read_text(encoding='utf-8'))
    conditions = list(results['conditions'])
    clean = results['conditions']['clean']

    # ---------------------------------------------------------------- main table
    rows = []
    for model in MODELS:
        entry = clean['models'][model]['tier1']
        ci = entry.get('auc_ci95', [float('nan')] * 2)
        delta = clean['delta_vs_baseline'].get(model)
        delta_text = '—' if not delta else f'{delta["delta"]:+.4f} [{delta["ci95"][0]:+.4f},{delta["ci95"][1]:+.4f}]'
        rows.append([SHORT[model], fmt(entry['auc']), f'[{fmt(ci[0])}, {fmt(ci[1])}]',
                     fmt(entry['eer']), fmt(entry['tar@0.01']), fmt(entry['tar@0.001']),
                     fmt(entry['same_mean']), fmt(entry['diff_mean']), fmt(entry['gap']),
                     fmt(entry['dprime'], 2), fmt(entry['rank_snr'], 2), delta_text])
    main_table = table(['模型', 'AUC', 'AUC 95% CI', 'EER', 'TAR@FAR=1e-2', 'TAR@FAR=1e-3',
                        '同人均值', '异人均值', '间隔', "d'", '组内SNR', 'ΔAUC vs 基线'], rows)

    # ------------------------------------------------------- per-domain tables
    domain_rows = []
    for model in MODELS:
        entry = clean['models'][model]
        domain_rows.append([SHORT[model]] + [
            f'{fmt(entry[key]["auc"])} / {fmt(entry[key]["eer"])}'
            for key in ('tiktok', 'iper', 'cross_clip')] +
            [fmt(entry['tier2']['same_mean']) if 'same_mean' in entry['tier2'] else '—'])
    domain_table = table(['模型', 'TikTok (AUC/EER)', 'iPER (AUC/EER)',
                          '跨片段难正例 (AUC/EER)', '真实→生成 同人cos'], domain_rows)

    # ------------------------------------------------------------- binding
    binding_section = ''
    if args.binding and args.binding.is_file():
        binding = json.loads(args.binding.read_text(encoding='utf-8'))
        rows = []
        for model in MODELS:
            entry = binding.get(model)
            if not entry:
                continue
            rows.append([SHORT[model], entry['frames'], f"{entry['binding_accuracy'] * 100:.1f}%",
                         fmt(entry['mean_diagonal']), fmt(entry['mean_offdiagonal']),
                         fmt(entry['margin']),
                         f"{entry['largest_face_rule_wrong_rate'] * 100:.1f}%"])
        binding_section = (
            '## 3. 双人绑定（beach14 生成图，49 帧有效）\n\n' +
            table(['模型', '帧数', '对角占优', '同人cos', '异人cos', '间隔',
                   '最大脸规则绑错率'], rows) +
            '\n\n`对角占优` = 源图左/右人物与生成图左/右人物的同人相似度同时高于交叉组合；'
            '`最大脸规则绑错率` = 按"取面积最大人脸"绑定时，把右侧人物当成 Person 0 的帧占比。'
            '注意这一档只有 2 个身份，AUC 没有判别意义，只能看绝对 cosine 与绑定关系。\n')

    # -------------------------------------------------------------- robustness
    robustness_rows = []
    for model in MODELS:
        cells = [SHORT[model]]
        for condition in conditions:
            entry = results['conditions'][condition]['models'][model]['tier1']
            cells.append(fmt(entry.get('auc')))
        robustness_rows.append(cells)
    robustness_table = table(['模型'] + [CONDITION_LABELS.get(c, c) for c in conditions],
                             robustness_rows)

    # ----------------------------------------------------------------- strata
    strata_sections = []
    for key, title in (('size', '人脸尺寸'), ('pose', '姿态（yaw 代理）'), ('sharp', '清晰度')):
        buckets = clean['strata'][key]
        rows = []
        for bucket, values in buckets.items():
            rows.append([bucket] + [f'{fmt(values[model]["auc"])} (n={values[model]["pairs_same"]})'
                                    if values[model].get('status') != 'insufficient'
                                    else 'n/a' for model in MODELS])
        strata_sections.append(f'**{title}**\n\n' +
                               table(['分档'] + [SHORT[m] for m in MODELS], rows))

    # ------------------------------------------------------------------ fusion
    fusion_rows = []
    for label, entry in clean['fusion'].items():
        if label.endswith('_weights'):
            continue
        fusion_rows.append([label, fmt(entry['auc']), fmt(entry['eer']),
                            fmt(entry['tar@0.001'])])
    fusion_table = table(['融合方式', 'AUC', 'EER', 'TAR@FAR=1e-3'], fusion_rows)

    # ------------------------------------------------------------ flip TTA
    flip_section = ''
    if args.results_flip and args.results_flip.is_file():
        flip = json.loads(args.results_flip.read_text(encoding='utf-8'))['conditions']['clean_flip']
        rows = []
        for model in MODELS:
            plain = clean['models'][model]['tier1']
            mirrored = flip['models'][model]['tier1']
            rows.append([SHORT[model], fmt(plain['auc']), fmt(mirrored['auc']),
                         f'{mirrored["auc"] - plain["auc"]:+.4f}',
                         fmt(plain['tar@0.001']), fmt(mirrored['tar@0.001'])])
        flip_section = ('## 7. 水平翻转 TTA（LVFace 官方 IJB-C 协议）\n\n' +
                        table(['模型', 'AUC 无TTA', 'AUC 翻转TTA', 'ΔAUC',
                               'TAR@1e-3 无TTA', 'TAR@1e-3 翻转TTA'], rows) +
                        '\n\nLVFace 官方 IJB-C 评测把原图与镜像图的特征拼接使用；这里对四个模型'
                        '统一做"原图+镜像平均"，用来确认排名不是被 TTA 协议拉偏的。\n')

    # ------------------------------------------------------------- detection
    detection_rows = []
    for source, stats in results['detection'].items():
        detection_rows.append([source, stats['images'], stats['with_face'],
                               f"{stats['fail_rate'] * 100:.2f}%"])
    sweep = {}
    if args.det_sweep and args.det_sweep.is_file():
        sweep = json.loads(args.det_sweep.read_text(encoding='utf-8'))
        for source, stats in sweep.items():
            detection_rows.append([f'{source} @det640 vs 1024', stats['images'],
                                   f"{stats['recall640'] * 100:.1f}% vs {stats['recall1024'] * 100:.1f}%",
                                   f"median side {stats['median_side640']:.0f}px"])
    detection_table = table(['数据/口径', '图片数', '检出', '失败率'], detection_rows)

    # ---------------------------------------------------------------- figures
    # 1. ROC on clean
    scores = load_scores(args.emb / 'clean.npz', args.pairs)
    figure, axis = plt.subplots(figsize=(5.2, 4.6), dpi=170)
    for model, (same, diff) in scores.items():
        labels = np.r_[np.ones_like(same), np.zeros_like(diff)]
        values = np.r_[same, diff]
        fpr, tpr, _ = roc_curve(labels, values)
        axis.plot(fpr, tpr, color=COLORS[model], linewidth=1.6,
                  label=f'{SHORT[model]}  AUC={roc_auc_score(labels, values):.4f}')
    axis.set_xscale('log')
    axis.set_xlim(1e-4, 1)
    axis.set_ylim(0.80, 1.005)
    axis.set_xlabel('False accept rate')
    axis.set_ylabel('True accept rate')
    axis.set_title('Clean verification, tier-1 (2 786 pairs)', fontsize=10)
    axis.grid(alpha=0.3, which='both')
    axis.legend(fontsize=7, loc='lower right')
    figure.tight_layout()
    figure.savefig(figures / 'roc-clean.png')
    plt.close(figure)

    # 2. robustness curves
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), dpi=170, sharey=True)
    groups = [('Alignment jitter', ['clean', 'align_j5', 'align_j10']),
              ('Resolution / blur', ['clean', 'blur1.5', 'lowres48', 'lowres32']),
              ('Compression / occlusion', ['clean', 'jpeg20', 'occl', 'genlike'])]
    for axis, (title, members) in zip(axes, groups):
        for model in MODELS:
            values = [results['conditions'][c]['models'][model]['tier1']['auc'] for c in members]
            axis.plot(range(len(members)), values, marker='o', markersize=3.5,
                      color=COLORS[model], label=SHORT[model], linewidth=1.4)
        axis.set_xticks(range(len(members)))
        axis.set_xticklabels([CONDITION_LABELS[c].split(' (')[0] for c in members],
                             rotation=20, ha='right', fontsize=7)
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel('AUC (tier-1)')
    axes[0].legend(fontsize=6.5)
    figure.tight_layout()
    figure.savefig(figures / 'robustness.png')
    plt.close(figure)

    # 3. same/diff distributions
    figure, axis = plt.subplots(figsize=(6.4, 3.6), dpi=170)
    positions, data, colours = [], [], []
    for index, model in enumerate(MODELS):
        same, diff = scores[model]
        positions.extend([index * 2.4, index * 2.4 + 0.8])
        data.extend([same, diff])
        colours.extend([COLORS[model], '#ADB5BD'])
    parts = axis.violinplot(data, positions=positions, widths=0.75, showextrema=False)
    for patch, colour in zip(parts['bodies'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)
    axis.set_xticks([index * 2.4 + 0.4 for index in range(len(MODELS))])
    axis.set_xticklabels([SHORT[m] for m in MODELS], fontsize=7.5)
    axis.set_ylabel('cosine similarity')
    axis.set_title('Same-person (coloured) vs different-person (grey)', fontsize=9)
    axis.grid(alpha=0.25, axis='y')
    figure.tight_layout()
    figure.savefig(figures / 'score-distributions.png')
    plt.close(figure)

    # ------------------------------------------------------------------- write
    counts = results.get('split', {})
    pair_counts = {}
    for pair in json.loads(Path(args.pairs).read_text(encoding='utf-8'))['pairs']:
        pair_counts[(pair['tier'], pair['kind'])] = pair_counts.get((pair['tier'], pair['kind']), 0) + 1
    text = f"""# 人脸身份后端对照实验结果

自动生成，来源 `output/results-v1/results.json`。样本：tier-1 {counts.get('identities', '?')} 个身份、
{pair_counts.get((1, 'same'), '?')} 同人 + {pair_counts.get((1, 'diff'), '?')} 异人对；
tier-2 为 beach14 真实→生成 {pair_counts.get((2, 'same'), '?')} + {pair_counts.get((2, 'diff'), '?')} 对。

## 1. 主结果（tier-1，clean，5 点 ArcFace 对齐）

{main_table}

## 2. 分域与难度

{domain_table}

`跨片段难正例` 只取同一身份不同 clip（换衣/换光/换机位）的正例，是对 RL 场景最接近的真实近似。

{binding_section}## 4. 鲁棒性消融（tier-1 AUC）

{robustness_table}

## 5. 分档（clean）

{chr(10).join(strata_sections)}

## 6. 双模型融合

{fusion_table}

融合在留出身份上评测；`z:` 为按校准身份做标准化后等权平均，`lr:` 为逻辑回归学习权重。

{flip_section}## 8. 人脸检测

{detection_table}

## 9. 图

- `figures/roc-clean.png`
- `figures/robustness.png`
- `figures/score-distributions.png`
"""
    (args.out / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
