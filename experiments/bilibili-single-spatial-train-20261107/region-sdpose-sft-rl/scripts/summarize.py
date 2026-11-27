"""Verify paired artifacts and publish a pilot readout plus preserved image panels."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))


def read(path):return json.loads(Path(path).read_text())


LABELS=['initial','uniform_S3','regional_S3','continue_sft','grpo']


def load_evaluations(root,split,pairs,expected_shas):
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    expected={key for key,row in pairs.items() if row['split']==split};reports={};images={}
    for label in LABELS:
        report_label=label if split=='val' else 'test_'+label
        report=read(root/f'report/evaluation/{report_label}/metrics.json');reports[label]=report
        records=report['records']
        if len(records)!=len(expected) or {r['pair_id'] for r in records}!=expected:
            raise ValueError('Evaluation does not cover the exact locked split: '+split)
        images[label]={r['pair_id']:r for r in records}
        for row in records:
            if row['split']!=split:raise ValueError('Evaluation split declaration differs')
            if row['model_sha256']!=expected_shas[label]:raise ValueError('Evaluation checkpoint differs: '+label)
            if file_sha(row['path'])!=row['sha256'] or row['target_rgb_used_as_condition'] is not False:
                raise ValueError('Generated image identity or condition declaration differs')
    return reports,images


def trace(reference,phase):
    from src.pose_control.qwen21_full.native_v62.training.checkpointing import load_committed
    rows={};seen=set()
    while reference is not None:
        if reference['path'] in seen:raise ValueError('Checkpoint lineage cycle')
        seen.add(reference['path'])
        saved,_,_=load_committed(reference,schema='native_v7_training_v1')
        state=saved['training_state']
        if state['config']['stage']['phase']!=phase:break
        with Path(state['journal']['path']).open() as f:
            for i,line in enumerate(f):
                if i>=state['journal']['sequence']:break
                record=json.loads(line)['record']
                if not record.get('optimizer_updated'):continue
                value={'window':[(x['route'],x['source'],x['index'],x['sigma']) for x in record['window']],
                       'learning_rates':record['learning_rates']}
                key=record['global_update']
                if key in rows and rows[key]!=value:raise ValueError('Conflicting committed update traces')
                rows[key]=value
        reference=state.get('parent');del saved
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--split',choices=['val','test'],default='val');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new readout directory')
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    from src.data.canvas_contract import CanvasSpec
    from PIL import Image,ImageDraw,ImageOps
    complete=read(a.root/'complete.json');pairs={r['pair_id']:r for r in read(a.root/'data/evaluation.json')}
    prepared=read(a.root/'prepared.json');dataset=prepared.get('dataset_label','Bilibili')
    ntrain,nval=prepared['train_pairs'],prepared['val_pairs']
    labels=LABELS
    rl=complete['grpo']
    guarded='ode_validation' in rl
    rl_caption='GRPO (SFT fallback)' if guarded and rl['selected_update']==0 else 'grpo'
    expected_shas={'initial':prepared['initialization']['sha256'],'uniform_S3':complete['sft']['uniform']['sha256'],
        'regional_S3':complete['sft']['regional']['sha256'],'continue_sft':complete['continue_sft']['sha256'],
        'grpo':complete['grpo']['checkpoint']['sha256']}
    reports,images=load_evaluations(a.root,a.split,pairs,expected_shas)
    alignment={}
    for phase,expected in prepared['budgets'].items():
        u=trace(read(a.root/f'report/uniform_{phase}-training.json')['checkpoint'],phase)
        r=trace(read(a.root/f'report/regional_{phase}-training.json')['checkpoint'],phase)
        if u!=r or len(u)!=expected:raise ValueError('SFT route/source/index/sigma/LR traces are not paired: '+phase)
        alignment[phase]={'updates':len(u),'microsteps':sum(len(x['window']) for x in u.values()),'identical':True}
    table=[]
    for label in labels:
        s=reports[label]['summary']
        table.append({'label':label,'images':s['images'],'person_detected_rate':s['person_detected_rate'],
            'body_pck002':s['pose']['body']['pck_002'],'body_l2_all':s['pose']['body']['l2_px_all'],
            'hand_pck002':s['pose']['hands']['pck_002'],'hand_l2_all':s['pose']['hands']['l2_px_all'],
            'hand_recall':s['pose']['hands']['recall'],'id_cos_src':s['identity']['id_cos_src']['mean'],
            'background_mse':s['pixel']['background_mse']})
    a.output.mkdir(parents=True);(a.output/'panels').mkdir()
    strips=[]
    for pair_id in sorted(images['initial']):
        pair=pairs[pair_id];source=CanvasSpec.from_record(pair['source_canvas']);target=CanvasSpec.from_record(pair['target_canvas'])
        with Image.open(pair['source_rgb']['path']) as image:src=source.image(image.convert('RGB'))
        with Image.open(pair['target_rgb']['path']) as image:gt=target.image(image.convert('RGB'))
        condition=read(a.root/f'conditions/{pair_id}.json')
        with Image.open(condition['images'][1]['file']['path']) as image:pose=image.convert('RGB')
        cells=[('Source',src),('SDPose target',pose),('GT',gt)]
        for label in labels:
            with Image.open(images[label][pair_id]['path']) as image:cells.append((label,image.convert('RGB')))
        strip=Image.new('RGB',(256*len(cells),560),'#171c22');draw=ImageDraw.Draw(strip)
        for col,(title,img) in enumerate(cells):
            tile=ImageOps.contain(img,(252,512));strip.paste(tile,(256*col+(256-tile.width)//2,25))
            draw.text((256*col+5,5),rl_caption if title=='grpo' else title,fill='white')
            if title in images:
                metrics=images[title][pair_id]['pose'];body=metrics['body']['pck_002'];hand=metrics['hands']['pck_002']
                draw.text((256*col+5,540),f'B@.02={body:.3f} H@.02={hand:.3f}' if hand is not None else f'B@.02={body:.3f} H=n/a',fill='white')
        with (a.output/'panels'/f'{pair_id}.png').open('xb') as f:strip.save(f,format='PNG')
        strips.append(ImageOps.contain(strip,(1280,400)))
    overview=Image.new('RGB',(1280,sum(x.height for x in strips)));offset=0
    for strip in strips:overview.paste(strip,(0,offset));offset+=strip.height
    with (a.output/'overview.png').open('xb') as f:overview.save(f,format='PNG')
    result={'schema':'bilibili_region_sdpose_rl_comparison_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_split':a.split,'evaluated_pairs':len(images['initial']),
        'paired_sft_traces':alignment,'metrics':table,'checkpoints':complete,
        'limitations':[f"{ntrain} training pairs, {nval} validation pairs, {prepared.get('test_pairs',0)} test pairs; one training seed",
            'SDPose supplies both weak pose annotations and evaluation; manual panels remain necessary',
            'RL rollout compute, wall time and pair exposure are tracked separately from continued SFT',
            'GRPO checkpoint follows the held-out ODE gate' if guarded else 'All checkpoints are fixed-budget endpoints']}
    with (a.output/'comparison.json').open('x') as f:json.dump(result,f,indent=2)
    unit=prepared.get('group_unit','video')
    lines=[f'# {dataset} 区域 FM / SDPose / GRPO：{a.split}','',
        f"训练 {ntrain} 对／{prepared['train_videos']} 个 {unit}；验证 {nval} 对／{prepared['val_videos']} 个 {unit}。单 seed 的有限预算集成 pilot。",'',
        f"本报告实际评价 {a.split} 的 {len(images['initial'])} 对图片；测试评价在所有优化完成后执行。",'',
        '| 权重 | Body PCK@.02 | Hand PCK@.02 | Body L2（含失败） | Hand L2（含失败） | Source ID cosine | BG MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'n/a' if x is None else f'{x:.5f}'
    for row in table:lines.append('| '+row['label']+' | '+' | '.join(fmt(row[k]) for k in
        ('body_pck002','hand_pck002','body_l2_all','hand_l2_all','id_cos_src','background_mse'))+' |')
    updates=sum(x['updates'] for x in alignment.values());microsteps=sum(x['microsteps'] for x in alignment.values())
    budget = (f"RL 请求 {rl['requested_updates']} 个更新，实际完成 {rl['completed_updates']} 个，"
              f"出图使用门禁批准的第 {rl['selected_update']} 个更新权重（0 为 SFT），停止原因为 `{rl['stop']}`。"
              if guarded else 'RL 和继续 SFT 使用相同的 optimizer-update 预算，RL 额外 rollout 成本另计。')
    lines+=['',f'两个 SFT 臂四阶段的 route/source/index/sigma/LR 轨迹完全一致（各 {updates} updates／{microsteps} microsteps）。',
        'RL 和继续 SFT 使用同一 RN SFT 父权重。'+budget,'',
        '![保存的逐对生成图对比](overview.png)','','单对大图在 `panels/`；全部原始生成图与 RL 候选图保留在 run 的 output 中。']
    with (a.output/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(a.output),'models':len(table),'evaluation_split':a.split,
                      'paired_images':len(images['initial']),'sft_trace_alignment':True}))


if __name__=='__main__':main()
