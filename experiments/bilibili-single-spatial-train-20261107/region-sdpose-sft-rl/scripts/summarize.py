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
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new readout directory')
    from src.pose_control.qwen21_full.native_v62.training.io import file_sha
    from src.data.canvas_contract import CanvasSpec
    from PIL import Image,ImageDraw,ImageOps
    complete=read(a.root/'complete.json');pairs={r['pair_id']:r for r in read(a.root/'data/evaluation.json')}
    labels=['initial','uniform_S3','regional_S3','continue_sft','grpo'];reports={};images={}
    for label in labels:
        report=read(a.root/f'report/evaluation/{label}/metrics.json');reports[label]=report
        images[label]={r['pair_id']:r for r in report['records']}
        for r in report['records']:
            if file_sha(r['path'])!=r['sha256'] or r['target_rgb_used_as_condition'] is not False:
                raise ValueError('Generated image identity or condition declaration differs')
    if len({frozenset(x) for x in images.values()})!=1:raise ValueError('Evaluation pairs do not match across branches')
    alignment={}
    for phase,expected in (('S0',100),('S1',50),('S2',100),('S3',50)):
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
        for label in labels[1:]:
            with Image.open(images[label][pair_id]['path']) as image:cells.append((label,image.convert('RGB')))
        strip=Image.new('RGB',(256*len(cells),560),'#171c22');draw=ImageDraw.Draw(strip)
        for col,(title,img) in enumerate(cells):
            tile=ImageOps.contain(img,(252,512));strip.paste(tile,(256*col+(256-tile.width)//2,25))
            draw.text((256*col+5,5),title,fill='white')
            if title in images:
                metrics=images[title][pair_id]['pose'];body=metrics['body']['pck_002'];hand=metrics['hands']['pck_002']
                draw.text((256*col+5,540),f'B@.02={body:.3f} H@.02={hand:.3f}' if hand is not None else f'B@.02={body:.3f} H=n/a',fill='white')
        with (a.output/'panels'/f'{pair_id}.png').open('xb') as f:strip.save(f,format='PNG')
        strips.append(ImageOps.contain(strip,(1280,400)))
    overview=Image.new('RGB',(1280,sum(x.height for x in strips)));offset=0
    for strip in strips:overview.paste(strip,(0,offset));offset+=strip.height
    with (a.output/'overview.png').open('xb') as f:overview.save(f,format='PNG')
    result={'schema':'bilibili_region_sdpose_rl_comparison_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'paired_sft_traces':alignment,'metrics':table,'checkpoints':complete,
        'limitations':['78 training pairs, 10 validation pairs, one training seed; integration pilot only',
            'SDPose supplies both weak pose annotations and evaluation; manual panels remain necessary',
            'RL and continued SFT match optimizer updates, not rollout compute, wall time or pair exposure',
            'All checkpoints are fixed-budget endpoints; no best-val checkpoint selection']}
    with (a.output/'comparison.json').open('x') as f:json.dump(result,f,indent=2)
    lines=['# Bilibili Regional FM / SDPose / GRPO pilot','',
        '78 train pairs / 39 videos; 10 val pairs / 5 videos. One seed; bounded integration pilot.','',
        '| Model | Body PCK@.02 | Hand PCK@.02 | Body L2 incl failures | Hand L2 incl failures | Source ID cosine | BG MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'n/a' if x is None else f'{x:.5f}'
    for row in table:lines.append('| '+row['label']+' | '+' | '.join(fmt(row[k]) for k in
        ('body_pck002','hand_pck002','body_l2_all','hand_l2_all','id_cos_src','background_mse'))+' |')
    lines+=['','SFT route/source/index/sigma/LR traces match in all four stages (300 updates / 1200 microsteps per arm).',
        'RL uses the same RN SFT parent as the continued-SFT control; additional rollout cost is not called equal compute.','',
        '![Preserved paired generated panels](overview.png)','','Detailed fixed-pair images are in `panels/`; original generations and all RL candidates remain under the run output tree.']
    with (a.output/'README.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(a.output),'models':len(table),'paired_validation_images':len(images['initial']),'sft_trace_alignment':True}))


if __name__=='__main__':main()
