"""Evidence-derived selection and memory report; never invent human scores."""
import argparse
import json
import math
from pathlib import Path
from .common import write_json

def records(root):
    for p in sorted((root/'runs').glob('*/summary.json')):
        yield json.loads((p.parent/'config.json').read_text()),json.loads(p.read_text())

def select(root):
    scores=json.loads((root/'scores.json').read_text())
    candidates={}
    for c,s in records(root):
        if c.get('kind')!='quality' or c.get('task') not in ('S2','D1') or c.get('group') not in ('Q0','Q2','Q3') or c.get('preprocess')!='letterbox':continue
        key=(c['cfg'],c['steps'],c['negative_prompt'])
        candidates.setdefault(key,[]).append((c,s,scores.get(c['id'])))
    ranking=[];ineligible=[]
    for key,items in candidates.items():
        if {(c['task'],c['seed']) for c,_,_ in items}!={(t,s) for t in ('S2','D1') for s in (42,1024)}:
            raise ValueError('Planned comparison attempts are missing')
        if any(s['status']!='PASS' for _,s,_ in items):
            ineligible.append(dict(cfg=key[0],steps=key[1],negative_prompt=key[2],reason='At least one attempt did not PASS',
                                   statuses={c['id']:s['status'] for c,s,_ in items}));continue
        if any(score is None for _,_,score in items):
            raise ValueError('Human/visual scores missing; do not select a configuration without all four images')
        for c,_,score in items:
            for k in ('action','identity','clothing','background','anatomy'):
                if k not in score or (score[k] is None and k!='identity') or (score[k] is not None and (type(score[k]) is not int or score[k] not in (0,1,2))):
                    raise ValueError(f'Invalid or missing score {c["id"]}.{k}')
            if any(type(score.get(k)) is not bool for k in ('added_person','missing_person','wrong_person')):
                raise ValueError(f'Missing hard-failure flags for {c["id"]}')
        hard=sum(any(score[k] for k in ('added_person','missing_person','wrong_person')) for _,_,score in items)
        totals=[sum(score[k] or 0 for _,_,score in items) for k in ('action','identity','clothing','background','anatomy')]
        rank=(hard,*[-v for v in totals],key[1],key[0],key[2])
        ranking.append((rank,key,[c['id'] for c,_,_ in items]))
    if len(candidates)!=7:raise ValueError(f'Expected attempts for 7 configurations, got {len(candidates)}')
    if not ranking:raise ValueError('No eligible complete scored configuration; Q4 must be NOT_RUN')
    ranking.sort()
    rank,key,ids=ranking[0]
    selection=dict(parameters=dict(cfg=key[0],steps=key[1],negative_prompt=key[2]),
                   policy='min hard-failure images, then max action/identity/clothing/background/anatomy sums; fewer steps/lower CFG ties',
                   identity_na_handling='0 contribution; inspect unjudgeable identity counts in scores, not identity failure rate',
                   source_ids=ids,ineligible=ineligible,ranking=[dict(rank=list(r),cfg=k[0],steps=k[1],negative_prompt=k[2],ids=rows) for r,k,rows in ranking])
    write_json(root/'quality_selection.json',selection)
    return selection

def capacity(root):
    rows=list(records(root));results=[]
    text=['# DeepGen 实测显存结果','',
          '所有数值来自本目录运行记录。单位 GiB（2³⁰ 字节）。训练任务为真实图像重建＋模拟控制，不是姿态迁移效果验证。', '',
          '| 分辨率 | 稳定训练峰值 NVML | 稳态 PyTorch allocated / reserved | 训练步耗时 P50 | 全流程最大占用 | 含余量建议容量 |',
          '|---|---:|---:|---:|---:|---:|']
    for res in (512,768,1024):
        stable=[(c,s) for c,s in rows if c.get('group')=='stable' and c.get('resolution')==res and s['status']=='PASS' and s.get('validation_passed') and not s.get('nvml_error')]
        lifecycle=[s for c,s in rows if c.get('resolution')==res and s['status']=='PASS' and (c.get('kind') in ('cache','adapter_infer') or c.get('group')=='stable')]
        infer_tasks={c.get('task') for c,s in rows if c.get('kind')=='adapter_infer' and c.get('resolution')==res and s['status']=='PASS' and not s.get('nvml_error') and (s.get('peak_nvml_gib') or 0)>0}
        cache_tasks={c.get('task') for c,s in rows if c.get('kind')=='cache' and c.get('resolution')==res and s['status']=='PASS' and s.get('cache_roundtrip_exact')}
        if len(stable)!=3 or any(s.get('train_steps',0)<100 for _,s in stable) or infer_tasks!={'S0','D0'} or cache_tasks!={'S0','D0'}:
            text.append(f'| {res} | 未完成稳定复测/独立推理 | — | — | — | 不给出容量结论 |');continue
        peaks=[s['peak_nvml_gib'] for _,s in stable];variation=(max(peaks)-min(peaks))/min(peaks)
        if variation>.05:
            text.append(f'| {res} | 复测差异 {variation:.1%}，需调查 | — | — | — | 暂不验收 |');continue
        stages=[p for _,s in stable for p in s['stages'] if p['stage']=='steady_train']
        alloc=max(p['peak_allocated_gib'] for p in stages);reserved=max(p['peak_reserved_gib'] for p in stages)
        peak=max(s['peak_nvml_gib'] for s in lifecycle)
        recommendation=math.ceil(peak+max(2,peak*.1))
        seconds=max(s['step_p50_seconds'] for _,s in stable)
        r=dict(resolution=res,stable_peak_nvml_gib=max(peaks),steady_peak_allocated_gib=alloc,steady_peak_reserved_gib=reserved,
               lifecycle_peak_nvml_gib=peak,suggested_capacity_gib=recommendation,repeat_variation=variation,
               encoding=stable[0][0]['encoding'],checkpointing=stable[0][0]['checkpointing'],step_p50_seconds=seconds)
        results.append(r)
        text.append(f'| {res} | {max(peaks):.2f} | {alloc:.2f} / {reserved:.2f} | {seconds:.3f}s | {peak:.2f} | {recommendation} GiB |')
    text+=['','建议容量 = ceil(全流程最大设备占用 + max(2 GiB, 10%))；不是对某款低显存显卡的实测认证。',
           '全流程包含缓存生成、稳定训练（含加载及检查）和独立加载适配器推理。三种显存口径不相加。NVML 50ms 采样可能漏过瞬时尖峰。','',
           '## 所有配置','', '| ID | 状态 | NVML 峰值 GiB | PyTorch allocated / reserved GiB |', '|---|---|---:|---:|']
    for c,s in rows:
        def fmt(k):return '—' if s.get(k) is None else f'{s[k]:.2f}'
        text.append(f'| {c["id"]} | {s["status"]} | {fmt("peak_nvml_gib")} | {fmt("peak_allocated_gib")} / {fmt("peak_reserved_gib")} |')
    (root/'MEMORY_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write_json(root/'capacity.json',results)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','capacity']);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(json.dumps(select(args.root) if args.command=='select' else capacity(args.root),ensure_ascii=False))
if __name__=='__main__':main()
