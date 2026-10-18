#!/usr/bin/env python3
"""Build saved-image BFS comparison panels. Target B is evaluation-only."""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def head_crop(image, bbox, side=256):
    from PIL import Image
    if bbox is None:
        return image.resize((side, side), Image.Resampling.LANCZOS)
    x0, y0, x1, y1 = bbox
    center_x, center_y = (x0+x1)/2, (y0+y1)/2
    extent = max(x1-x0, y1-y0) * 2.0
    left = max(0, min(image.width - extent, center_x - extent/2))
    top = max(0, min(image.height - extent, center_y - extent/2))
    return image.crop((round(left), round(top), round(left+extent), round(top+extent))).resize((side, side), Image.Resampling.LANCZOS)


def number(value):
    return '--' if value is None else f'{value:.4f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('/home/shangguanrz/nihaochen/qwen21_full/bfs_compare_30_20261004'))
    args = p.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    from eval_iper_four_arm import IdentityMetric
    from scripts.eval.bfs_compare import face_detection
    run = args.run
    records = json.loads((run / 'records.json').read_text())
    result = json.loads((run / 'results.json').read_text())
    (run / 'panels').mkdir(exist_ok=True)
    (run / 'appearance_sheets').mkdir(exist_ok=True)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    identity = IdentityMetric()
    labels = ('Source A', 'Target B (eval only)', 'Body C (512 upscale)', 'Base repair', 'BFS V1.1', 'BFS alternative')
    variants = ('body', 'nolora', 'bfs', 'alternative')
    pages, groups, candidates = [], {}, []
    for record in records:
        stem = record['stem']
        source = Image.open(record['source_path']).convert('RGB')
        target = Image.open(record['target_path']).convert('RGB')
        target_bbox, _ = face_detection(identity, target)
        images = [source, target]
        boxes = [record['source_bbox'], target_bbox]
        for variant in variants:
            path = run / 'images' / f'{stem}_{variant}.png'
            images.append(Image.open(path).convert('RGB') if path.exists() else Image.new('RGB', (768, 768), (60, 60, 60)))
            boxes.append(record.get('body_head_bbox_for_eval'))
        canvas = Image.new('RGB', (1536, 608), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        zoom = Image.new('RGB', (1536, 304), (245, 245, 245))
        zd = ImageDraw.Draw(zoom)
        for col, (image, bbox, label) in enumerate(zip(images, boxes, labels)):
            draw.text((col*256+7, 5), label, font=font, fill=(15, 15, 15))
            canvas.paste(image.resize((256, 256), Image.Resampling.LANCZOS), (col*256, 32))
            crop = head_crop(image, bbox)
            canvas.paste(crop, (col*256, 320))
            zd.text((col*256+7, 5), label, font=font, fill=(15, 15, 15))
            zoom.paste(crop, (col*256, 32))
            if col >= 2:
                metrics = record['metrics'][variants[col-2]]
                text = f'ID={number(metrics.get("identity_A"))} PCK={number(metrics.get("pose_pck_002"))}'
                draw.text((col*256+5, 290), text, font=font, fill=(15, 15, 15))
                draw.text((col*256+5, 580), f'Outside MAE={number(metrics.get("outside_head_mae_255"))}', font=font, fill=(15, 15, 15))
        panel = f'panels/{stem}.jpg'
        canvas.save(run / panel, quality=94, subsampling=0)
        groups.setdefault(record['appearance'], []).append(zoom)
        bfs, base = record['metrics']['bfs'].get('identity_A'), record['metrics']['nolora'].get('identity_A')
        delta = bfs-base if bfs is not None and base is not None else None
        if delta is not None:
            candidates.append({'stem': stem, 'appearance': record['appearance'], 'bfs_minus_nolora_identity': delta, 'panel': panel})
        pages.append(f'<article id="{html.escape(stem)}"><h3>{html.escape(stem)}</h3><p>BFS−基干身份差值：{number(delta)}。上排整图，下排头部；生成结果使用同一个 C 头部框，便于检查偏移。</p><a href="{panel}"><img loading="lazy" src="{panel}" alt="六组整图与头部对照"></a></article>')
    for person, zooms in groups.items():
        sheet = Image.new('RGB', (1536, 304*len(zooms)), (245, 245, 245))
        for i, zoom in enumerate(zooms):
            sheet.paste(zoom, (0, i*304))
        sheet.save(run / 'appearance_sheets' / f'{person}.jpg', quality=94, subsampling=0)
    ordered = sorted(candidates, key=lambda row: row['bfs_minus_nolora_identity'])
    (run / 'visual_review_candidates.json').write_text(json.dumps({'worst_three': ordered[:3], 'best_three': ordered[-3:],
         'one_per_appearance': [next((r for r in candidates if r['appearance'] == person), None) for person in groups]}, indent=2), encoding='utf-8')
    rows = []
    for variant in variants:
        stats = result['summary'][variant]
        rows.append(f'<tr><td>{variant}</td><td>{stats["n_complete"]}/{stats["n_total"]}</td><td>{number(stats["identity_A_mean"])} ({stats["identity_A_n"]})</td><td>{number(stats["common_identity_mean"])} ({stats["common_identity_n"]})</td><td>{stats["n_unique_face_on_eligible"]}/{stats["n_source_face_eligible"]}</td><td>{number(stats["eligible_pose_pck_002_mean"])}</td><td>{number(stats["outside_head_mae_255_mean"])}</td></tr>')
    navigation = ' · '.join(f'<a href="appearance_sheets/{person}.jpg">{person} 衣着组头部对照</a>' for person in groups)
    output = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>BFS 30 对固定条件实验</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1536px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}table{border-collapse:collapse}th,td{padding:8px 16px;border:1px solid #ccc}img{width:100%;height:auto}article{margin:40px 0;border-top:1px solid #ccc}a{color:#125cab}</style><h1>BFS 扩大样本对照：30 对 / 8 人物 / 10 衣着组</h1><p>第一阶段简化 Body Adapter 生成 C；第二阶段是基干编辑、V1.1、alternative。三个第二阶段固定相同双图、提示词、768 分辨率、seed 42、30 步。Target B 仅用于评测与展示，未输入 BFS。</p><p>30 对中 24 对有唯一源参考脸；其余 6 对未运行第二阶段，作为保留 C 的回退案例。Body 对照是 512 生成结果的确定性放大。各自身份均值受检测成功样本组成影响，需结合共同四分支有效样本均值和检测覆盖。姿态列只在共同源脸有效样本上比较；身体 PCK 不能代替头部角度和表情检验。</p><table><tr><th>分支</th><th>完成</th><th>各自身份均值（n）</th><th>共同身份均值（n）</th><th>唯一脸 / 可修复</th><th>共同样本 PCK</th><th>头部区域外 MAE / 255</th></tr>' + ''.join(rows) + '</table><p>'+navigation+'</p><p><a href="results.json">汇总 JSON（按实际人物聚类）</a> · <a href="per_pair.csv">逐样本 CSV</a> · <a href="config.json">生成配置</a> · <a href="provenance.json">输入与评测指纹</a></p>' + ''.join(pages) + '</html>'
    (run / 'index.html').write_text(output, encoding='utf-8')
    print(f'PANELS {len(pages)}; appearance sheets {len(groups)}; index.html ready', flush=True)


if __name__ == '__main__':
    main()
