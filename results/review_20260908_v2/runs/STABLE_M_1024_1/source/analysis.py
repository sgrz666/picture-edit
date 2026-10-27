"""Evidence-derived selection and memory report; never invent human scores."""
import argparse
import json
import math
from pathlib import Path
from .common import write_json,sha256

def records(root):
    for p in sorted((root/'runs').glob('*/summary.json')):
        yield json.loads((p.parent/'config.json').read_text(encoding='utf-8')),json.loads(p.read_text(encoding='utf-8'))


def evidence_index(root):
    """Add a review index only after byte-matching saved training inputs to cache inputs."""
    rows=list(records(root));indexed=[]
    for c,s in rows:
        if c.get('group')!='stable' or s['status']!='PASS':continue
        for task,relative in c.get('task_evidence',{}).items():
            directory=root/'runs'/c['id']/relative
            matches=[(cc,ss) for cc,ss in rows if cc.get('kind')=='cache' and cc.get('task')==task and cc.get('resolution')==c['resolution'] and cc.get('preprocess')==c['preprocess'] and ss['status']=='PASS']
            assert len(matches)==1, f'Ambiguous cache source for {c["id"]}/{task}'
            cc,ss=matches[0];cache_dir=root/'runs'/cc['id']
            for name in ('input_original.png','input_model.png'):
                assert sha256(directory/name)==sha256(cache_dir/name), f'Training/cache saved image mismatch: {c["id"]}/{task}/{name}'
            fields={k:cc[k] for k in ('input_sha256','input_path','original_size','content_box','model_revision','pipeline_sha256','resolution','preprocess')}
            fields.update(task=task,source_cache_run=cc['id'],cache_key=ss['cache_key'],
                          input_original_png_sha256=sha256(directory/'input_original.png'),input_model_png_sha256=sha256(directory/'input_model.png'),
                          provenance='Post-run review index: original and model-input PNG bytes exactly matched to the verified cache run; original run configuration is unchanged.')
            write_json(directory/'config.json',fields);indexed.append(c['id']+'/'+task)
    write_json(root/'evidence_index.json',dict(indexed=indexed,scope='Post-run provenance index, not a new GPU measurement'))
    return indexed

def select(root):
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8'))
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
           '## 固定实验条件','',
           '单卡 RTX 6000D，micro-batch=1，梯度累积=1。冻结基座 BF16；外挂适配器参数、梯度和 AdamW 状态 FP32，前向 BF16 autocast。主配置 130,277,376 个可训练参数，DiT/VLM/SCB/VAE 均不更新。',
           'AdamW：lr=1e-4，betas=(0.9,0.999)，eps=1e-8，weight_decay=0.01，foreach=False，fused=False；梯度范数裁剪 1.0。每例只有一张参考图，完整真实语义条件不截短。',
           '缓存模式训练时仅 DiT 和适配器驻留 GPU；缓存生成单独计量。在线模式冻结编码器驻留 GPU，每步重新编码。检查点配置同时覆盖 DiT 和适配器中间层，已验证实际重计算。','',
           '## 主矩阵：同一配置两张照片的最大值','',
           '| 分辨率 | 编码 | 检查点 | 有效照片数 | NVML 峰值 GiB | 稳态单步 P50 秒（两例较大值） |',
           '|---|---|---|---:|---:|---:|']
    for res in (512,768,1024):
        for enc in ('cache','online'):
            for ckpt in (True,False):
                pair=[s for c,s in rows if c.get('group')=='main' and c.get('resolution')==res and c.get('encoding')==enc and c.get('checkpointing')==ckpt and s['status']=='PASS']
                peak=f'{max(s["peak_nvml_gib"] for s in pair):.2f}' if pair else '—'
                step=f'{max(s["step_p50_seconds"] for s in pair):.3f}' if pair else '—'
                text.append(f'| {res} | {enc} | {"开启" if ckpt else "关闭"} | {len(pair)}/2 | {peak} | {step} |')
    text+=['','## 适配器规模敏感性（缓存＋检查点）','',
           '| 规模 | 实测可训练参数 | 分辨率 | 有效照片数 | NVML 峰值 GiB |','|---|---:|---:|---:|---:|']
    for size in ('S','M','L'):
        for res in (512,768,1024):
            pair=[s for c,s in rows if c.get('group') in ('main','sensitivity') and c.get('adapter_size')==size and c.get('resolution')==res and c.get('encoding')=='cache' and c.get('checkpointing') and s['status']=='PASS']
            params=str(pair[0]['adapter_parameters']) if pair else '—'
            peak=f'{max(s["peak_nvml_gib"] for s in pair):.2f}' if pair else '—'
            text.append(f'| {size} | {params} | {res} | {len(pair)}/2 | {peak} |')
    text+=['','## 稳定性复测','',
           '| 分辨率 | 独立运行 | 实际更新数 | NVML 峰值 GiB |','|---|---|---:|---:|']
    for c,s in rows:
        if c.get('group')=='stable':
            peak=f'{s["peak_nvml_gib"]:.2f}' if s.get('peak_nvml_gib') else '—'
            text.append(f'| {c["resolution"]} | {c["id"]} | {s.get("train_steps","—")} | {peak} |')
    text+=['','## 所有配置','', '| ID | 状态 | NVML 峰值 GiB | PyTorch allocated / reserved GiB |', '|---|---|---:|---:|']
    for c,s in rows:
        def fmt(k):return '—' if s.get(k) is None else f'{s[k]:.2f}'
        text.append(f'| {c["id"]} | {s["status"]} | {fmt("peak_nvml_gib")} | {fmt("peak_allocated_gib")} / {fmt("peak_reserved_gib")} |')
    (root/'MEMORY_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    write_json(root/'capacity.json',results)
    return results

def quality_report(root):
    scores=json.loads((root/'scores.json').read_text(encoding='utf-8')) if (root/'scores.json').exists() else {}
    rows=[(c,s) for c,s in records(root) if c.get('kind')=='quality' and c.get('group') in ('Q0','Q1','Q2','Q3','Q4')]
    originals=[root/'runs'/name/'output.png' for name in ('Q0_S0_42_stretch_35_4.5','REPRO_Q0_S0_42')]
    repro={str(p.relative_to(root)):sha256(p) for p in originals if p.exists()}
    repro_ok=len(repro)==2 and len(set(repro.values()))==1
    write_json(root/'reproducibility.json',dict(output_hashes=repro,exact_match=repro_ok,scope='same fixed software, hardware, preprocessing and RNG policy'))
    lines=['# 原生编辑效果复核','',
           '评分为 Codex 对实际输出的 AI 视觉初评，待人工复核，不是人工偏好实验或通用 benchmark。全部结果均保留，不挑选最好种子。', '',
           '## 目前能确认的结论','',
           '- 本轮原生编辑没有注入骨架、深度、法线或接触条件。',
           '- 独立进程同配置复现：'+('输出 PNG 哈希完全一致。' if repro_ok else '尚未通过两次输出哈希一致检查。'),
           '- 本轮招手案例中可观察到墨镜、帽子保留，同时仍有衣物或雪具变化；未做该提示词成分的单独消融，不能据此作因果结论或外推身份保持能力。',
           '- 多人场景中的新增人物、原坐姿人物未起身、衣物替换是独立失败项；动作画面看似合理不能抵消这些错误。',
           '- 灰色补边有时被模型扩展成场景；去补边图使用输入记录的 content_box，完整输出也保留供检查。', '',
           '| 分组 | 实际完成输出 | 已评分 | 观察到新增/丢失/错配人物的图像 |', '|---|---:|---:|---:|']
    for group in ('Q0','Q1','Q2','Q3','Q4'):
        eligible=[(c,s) for c,s in rows if c['group']==group and s['status']=='PASS']
        reviewed=[scores[c['id']] for c,_ in eligible if c['id'] in scores]
        hard=sum(any(s.get(k) is True for k in ('added_person','missing_person','wrong_person')) for s in reviewed)
        lines.append(f'| {group} | {len(eligible)} | {len(reviewed)} | {hard} |')
    lines+=['','以上为本轮有限样本计数，分组任务构成不同，不能直接当作胜率对比。身份 null 表示无法判断，不计为保持成功。', '',
            '## 逐图初评','', '| ID | 动作 / 身份 / 服装 / 背景 / 肢体 | 观察依据 |','|---|---|---|']
    for c,s in rows:
        score=scores.get(c['id'])
        if not score:
            lines.append(f'| {c["id"]} | 未评（{s["status"]}） | — |');continue
        values=' / '.join('N/A' if score.get(k) is None else str(score[k]) for k in ('action','identity','clothing','background','anatomy'))
        lines.append(f'| {c["id"]} | {values} | {score.get("note","").replace("|","/")} |')
    selection=root/'quality_selection.json'
    if selection.exists():
        selected=json.loads(selection.read_text(encoding='utf-8'))
        lines+=['','## 分辨率对照参数','',f'Q4 使用 `{json.dumps(selected["parameters"],ensure_ascii=False)}`。完整排序与依据见 quality_selection.json；选出的是本轮对照参数，不是质量达标认证。']
    lines+=['','## 适配器实验的边界','',
            '训练数据是两张照片的自身重建目标，控制数据为不与照片对齐的人工几何。适配器输出只证明已保存、可加载并能注入推理；不能说明学会了姿态迁移。',
            '后续效果研究应先建立身份明确、源图/目标图/目标控制一致的少量真实配对样本，并加入人物实例对应信息，再评估结构条件带来的增益。当前预算回答的是固定单参考图、固定适配器规模与精度的资源需求。']
    (root/'QUALITY_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(outputs=len(rows),reviewed=sum(c['id'] in scores for c,_ in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['select','capacity','quality','evidence']);p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(json.dumps({'select':select,'capacity':capacity,'quality':quality_report,'evidence':evidence_index}[args.command](args.root),ensure_ascii=False))
if __name__=='__main__':main()
