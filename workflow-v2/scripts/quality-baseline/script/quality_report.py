"""Paired metrics and browsable evidence. Execution never implies quality passed."""
from __future__ import annotations
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import statistics


def records(out):
    from campaign import paths
    result=[]
    for path in sorted((out/'jobs').glob('*.json')):
        item=json.loads(path.read_text())
        if item['status']!='complete':continue
        run,g,e=paths(out,item['job'])
        result.append(dict(**item,run=str(run),generation=json.loads((g/'generation.json').read_text()),
            evaluation=json.loads((e/'evaluation.json').read_text()),image=str(g/'final_canvas.png'),evaluation_dir=str(e)))
    return result


def metrics(e):
    def mean(values):
        values=[v for v in values if v is not None]
        return statistics.mean(values) if values else None
    persons=e['pose'].get('persons',[])
    return dict(pck_02=mean([p.get('pck@0.2') for p in persons]),
        pck_02_all_requested=e['pose'].get('coverage',{}).get('pck_02_all_requested'),
        body_detection_coverage=e['pose'].get('coverage',{}).get('coverage'),
        mpjpe_px=mean([p.get('mpjpe_px') for p in persons]),
        visible_body_points=sum(p.get('body_joints',0) for p in persons),
        arcface=mean(e.get('identity',{}).get('assigned_cosine',[])),
        appearance=mean(e.get('appearance',{}).get('assigned_cosine',[])),
        background_mae=e.get('background',{}).get('mae'),
        pelvis_error_px=(e.get('relative_pelvis',{}).get('value') or {}).get('error_px'),
        contact_proxy_error_px=e.get('contact_2d_proxy',{}).get('mean_relative_error_px'))


def paired(rows):
    baselines={(r['job']['case'],r['job']['seed']):r for r in rows if r['job']['phase']=='formal' and r['job']['mode']=='qwen_baseline'}
    pairs=[]
    for row in rows:
        j=row['job']
        if j['phase']!='formal' or j['mode']=='qwen_baseline':continue
        baseline=baselines.get((j['case'],j['seed']))
        if baseline is None:continue
        a,b=metrics(row['evaluation']),metrics(baseline['evaluation'])
        pairs.append(dict(case=j['case'],seed=j['seed'],variant=j['variant'],native=a,baseline=b,
            delta={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in a},
            assignment_uncertain=row['evaluation']['identity'].get('assignment_uncertain',True) or baseline['evaluation']['identity'].get('assignment_uncertain',True)))
    return pairs


def ablation_pairs(rows):
    base={(r['job']['case'],r['job']['seed'],r['job']['mode'],r['job']['variant']):r
        for r in rows if r['job']['phase']=='formal'}
    pairs=[]
    for row in rows:
        j=row['job']
        if j['phase'] not in ('formal','diagnose'):continue
        if j['phase']=='formal' and (j['variant']=='full' or j['mode']=='qwen_baseline'):continue
        baseline_variant='body_interaction' if j['variant']=='body_only' else 'full'
        other=base.get((j['case'],j['seed'],j['mode'],baseline_variant))
        if other is None:continue
        a,b=metrics(row['evaluation']),metrics(other['evaluation'])
        pairs.append(dict(case=j['case'],seed=j['seed'],comparison=j['label']+'_vs_'+
            ('qwen_only' if j['mode']=='qwen_baseline' else baseline_variant),
            image=row['image'],control_image=other['image'],
            delta={k:None if a[k] is None or b[k] is None else a[k]-b[k] for k in a}))
    return pairs


def audit(out,*,write=True):
    import hashlib
    import numpy as np
    import torch
    from abc_runner import sha256
    from abc_quality import geometry_consistency
    from src.pose_edit.common import atomic_json
    rows=records(out);groups=defaultdict(list);errors=[];geometry={}
    for row in rows:
        j=row['job'];g=row['generation']
        if row['run'] not in geometry:
            scene=json.loads((Path(row['run'])/'geometry/scene.json').read_text())
            with np.load(scene['geometry_arrays'],allow_pickle=False) as arrays:geometry[row['run']]=geometry_consistency(arrays)
            for check in scene['diagnostics'].get('source_fit_checks',[]):
                if check['mesh_max_error_m']>1e-4 or check['projection_max_error_px']>.25:errors.append('Source reconstruction tolerance failed')
            if not row['evaluation']['body_shape']['betas_preserved']:errors.append('Source betas changed')
        frozen_sources=torch.load(g['qwen_source64']['path'],map_location='cpu',weights_only=True)
        for index,name in enumerate(('source_A','source_B')):
            value=frozen_sources['sources'][name].contiguous()
            actual=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
            if actual!=g['qwen_source64']['sources'][index]['latent_sha256']:errors.append('Stored Source64 numerical SHA mismatch')
        if g.get('target_rgb_loaded') is not False:errors.append('Target RGB evidence missing: '+row['image'])
        if len(g['reference_images'])!=5:errors.append('Five-reference contract failed')
        if j['mode']=='native_v5':
            if not(g['adapter_loaded'] and g['native_forward_executed']):errors.append('Native did not execute')
            if g['checkpoint_sha256']!='6a340c0a9bde9fbea943bfb26cbfce425a893ac2201d1b23651e0508e4bd20e2':errors.append('Checkpoint changed')
            observation=g.get('residual_observation')
            if not observation or sha256(observation['path'])!=observation['sha256']:errors.append('Residual observation missing or changed')
            else:
                summary=json.loads(Path(observation['path']).read_text())['branch_summary']
                if not summary.get('body',{}).get('nonzero',0):errors.append('No measured Body residual execution')
                disabled=('face','hand') if j['variant'] in ('body_interaction','body_only') else (('hand',) if j['variant']=='no_hand' else (('face',) if j['variant']=='no_face' else ()))
                if j['variant']=='body_only':disabled+=('interaction',)
                if j['variant'] in ('body_interaction','body_only','no_source_detail'):disabled+=('detail',)
                for branch in disabled:
                    if summary.get(branch,{}).get('nonzero',0):errors.append('Disabled branch contributed: '+branch)
        elif g['adapter_loaded']:errors.append('Baseline loaded Adapter')
        if j['phase'] in ('smoke','formal'):groups[(j['case'],j['resolution'],j['seed'])].append(g)
    for group,items in groups.items():
        for key,getter in (
            ('prompt',lambda g:g['prompt_sha256']),('source64',lambda g:[s['latent_sha256'] for s in g['qwen_source64']['sources']]),
            ('noise',lambda g:g['sampling']['initial_noise_sha256']),('scheduler',lambda g:g['sampling']['timesteps']),
            ('five_images',lambda g:[r['input_file']['sha256'] for r in g['reference_images']])):
            if len({json.dumps(getter(g),sort_keys=True) for g in items})>1:errors.append(str(group)+' mismatched '+key)
    if errors:raise RuntimeError('\n'.join(errors))
    result=dict(schema='abc_quality_audit_v1',checked_jobs=len(rows),expected_jobs=68,
        execution_complete=len(rows)==68,evidence_checks_passed=True,quality_passed=None,
        no_training=True,unpaired_output_has_no_ground_truth_rgb=True)
    result['geometry']=geometry
    if write:atomic_json(out/'audit.json',result)
    return result


def build_report(out):
    from src.pose_edit.common import atomic_json
    from PIL import Image,ImageDraw,ImageOps
    rows=records(out);pairs=paired(rows)
    atomic_json(out/'paired_metrics.json',pairs)
    atomic_json(out/'ablation_metrics.json',ablation_pairs(rows))
    by_variant=defaultdict(list)
    for pair in pairs:by_variant[pair['variant']].append(pair)
    aggregate={}
    for variant,items in by_variant.items():
        aggregate[variant]={}
        for key in ('pck_02','pck_02_all_requested','body_detection_coverage','mpjpe_px','arcface','appearance','background_mae','pelvis_error_px'):
            values=[p['delta'][key] for p in items if p['delta'][key] is not None]
            higher=key in ('pck_02','pck_02_all_requested','body_detection_coverage','arcface','appearance')
            aggregate[variant][key]=dict(comparable=len(values),median_delta=statistics.median(values) if values else None,
                favorable=sum(v>0 if higher else v<0 for v in values),unfavorable=sum(v<0 if higher else v>0 for v in values))
    atomic_json(out/'summary.json',dict(expected=68,complete=len(rows),formal_complete=sum(r['job']['phase']=='formal' for r in rows),
        quality_passed=None,aggregate=aggregate,interpretation='Paired differences on three cases and three seeds; not a generalization or significance claim.'))
    with (out/'metrics.csv').open('w',newline='') as stream:
        columns=['phase','case','seed','label',*metrics(rows[0]['evaluation']).keys()] if rows else ['phase','case','seed','label']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in rows:writer.writerow({**{k:r['job'][k] for k in ('phase','case','seed','label')},**metrics(r['evaluation'])})
    galleries=defaultdict(list)
    for r in rows:galleries[(r['job']['phase'],r['job']['case'],r['job']['seed'])].append(r)
    sections=[]
    for (phase,case,seed),items in sorted(galleries.items()):
        cards=[]
        for r in items:
            image=Path(r['image']);rel=image.relative_to(out)
            cards.append(f'<figure><figcaption>{html.escape(r["job"]["label"])}</figcaption><a href="../{rel}"><img loading="lazy" src="../{rel}"></a></figure>')
        sections.append(f'<h2>{phase} / {case} / seed {seed}</h2><div class="row">'+''.join(cards)+'</div>')
        if phase=='formal':
            panel=Image.new('RGB',(320*len(items),430),'#222222');draw=ImageDraw.Draw(panel)
            for i,r in enumerate(items):
                with Image.open(r['image']) as im:thumb=ImageOps.contain(im.convert('RGB'),(320,400))
                panel.paste(thumb,(320*i+(320-thumb.width)//2,30+(400-thumb.height)//2))
                draw.text((320*i+6,8),r['job']['label'],fill='white')
            filename=f'{case}_s{seed}_comparison.png';path=out/'report'/filename
            # Reports may be rebuilt; generated photos themselves are immutable.
            if not path.exists():panel.save(path)
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A+B+C 质量基线</title>
<style>body{font:16px system-ui;background:#151515;color:#eee;margin:24px}.row{display:flex;gap:12px;overflow:auto}figure{margin:0;min-width:240px}img{width:280px;height:380px;object-fit:contain}figcaption{padding:8px}</style>
<h1>A+B+C 质量基线</h1><p>执行成功与质量结论分别记录。点击图片查看原图；脸手局部图位于各 evaluation 目录。</p>'''+''.join(sections)
    (out/'report/index.html').write_text(document,encoding='utf8')
    text=['# A+B+C Workflow 阶段 1 质量基线','',f'已完成 {len(rows)}/68；正式主矩阵 {sum(r["job"]["phase"]=="formal" for r in rows)}/36。',
        '', '完整对照见 [图片浏览页](index.html)，数值见上级 `paired_metrics.json`、`metrics.csv`、`summary.json`。',
        '', '握手下肢是估计几何，Hi4D 官方 SMPL 不包含细手指姿态。缺失检测不算姿态正确。ArcFace 不可测时保留空值。',
        '', '本批次没有配对输出真值 D；背景指标比较 A 的有效背景区域。接触二维代理不能代替三维接触。',
        '', '## 成对指标', '', '| Variant | 指标 | 可比数 | 中位差值 | 有利/不利 |','|---|---|---:|---:|---:|']
    for variant,values in aggregate.items():
        for key,v in values.items():text.append(f'| {variant} | {key} | {v["comparable"]} | {v["median_delta"]} | {v["favorable"]}/{v["unfavorable"]} |')
    text+=['','质量结论由实际图片复查与成对指标共同形成；不把运行通过或单一指标变化当成质量达标。']
    (out/'report/报告.md').write_text('\n'.join(text),encoding='utf8')
    audit(out)
