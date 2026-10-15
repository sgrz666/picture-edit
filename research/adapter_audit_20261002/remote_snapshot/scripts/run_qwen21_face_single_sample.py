#!/usr/bin/env python3
"""Bounded single-pair diagnosis. Every GPU action uses the complete controller."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image, ImageDraw
from eval_iper_four_arm import PoseMetric, image_metrics, open_rgb
from scripts.evaluate_qwen21_weight_inheritance import cpu_identity_metric, face_detection
from src.pose_control.qwen21_full.data_runtime import ASSETS, pair_metadata, read_pairs
from src.pose_control.qwen21_full.execution import RUN_ROOT, json_write

OUT = RUN_ROOT / 'face_single_sample'
BODY = RUN_ROOT / 'body_512_100/checkpoint_last.pt'
FACE = RUN_ROOT / 'face_512_100/checkpoint_last.pt'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def command(label, args):
    log = OUT / f'{label}.log'
    print(json.dumps({'action': label, 'command': list(map(str,args)), 'log': str(log)}, ensure_ascii=False), flush=True)
    with log.open('a') as stream:
        subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)

def generate(label, pairs, checkpoint=BODY, *, mode='body', progress='sigma', strength=1., kv=False, resolution=512):
    directory = OUT / label
    args = [ROOT / 'eval_iper_qwen21_full.py', '--pairs', pairs,
            '--body-checkpoint', checkpoint, '--face-checkpoint', checkpoint,
            '--modes', mode, '--output', directory, '--resolution', resolution,
            '--progress-mode', progress, '--face-strength', strength,
            '--use-kv-cache' if kv else '--no-use-kv-cache']
    command(label.replace('/', '_'), args)
    return [json.loads(line) for line in (directory / 'per_sample.jsonl').read_text().splitlines()]

class Metrics:
    def __init__(self):
        torch.set_num_threads(4)
        self.pose, self.identity = PoseMetric(), cpu_identity_metric()
        self.features, self.conditions = {}, {}

    def feature(self, app):
        if app not in self.features:
            self.features[app] = torch.load(RUN_ROOT / 'face_cache' / f'{app}.pt', map_location='cpu', weights_only=False)
        return self.features[app]

    def box(self, app, stem):
        value = self.feature(app)
        return value['face_boxes'][value['stem_to_idx'][stem]].float().tolist()

    def points(self, app, stem):
        if app not in self.conditions:
            self.conditions[app] = torch.load(ASSETS / app / 'v6_conditions.pt', map_location='cpu', weights_only=False)
        value = self.conditions[app]
        return value['kps_25_coords'][value['stem_to_idx'][stem]].numpy()

    def evaluate(self, row, pair):
        meta = pair_metadata(pair)
        a,b,g = (open_rgb(Path(p)) for p in (meta['source_rgb'],meta['target_rgb'],row['image']))
        abox,bbox = self.box(meta['source_appearance'],meta['source_stem']), self.box(meta['target_appearance'],meta['target_stem'])
        ad, ar, av = face_detection(self.identity, a, abox)
        bd, br, bv = face_detection(self.identity, b, bbox)
        gd, gr, gv = face_detection(self.identity, g, bbox)
        metrics = self.pose.compare(self.pose.detect(g), self.points(meta['target_appearance'],meta['target_stem']))
        masks=[]
        for which in ('source','target'):
            app=meta[f'{which}_appearance']; record=pair[which]
            path=ASSETS/app/'mask_512'/record['role']/f'{record["stem"]}.png'
            masks.append(np.asarray(Image.open(path).convert('L').resize((512,512),Image.Resampling.NEAREST))/255.)
        metrics.update(image_metrics(g,b,a,masks[1],masks[0]))
        metrics.update(identity_cosine_raw=self.identity.cosine(ar,gr), identity_cosine_reliable=self.identity.cosine(av,gv),
                       identity_target_cosine_raw=self.identity.cosine(br,gr),source_target_identity_cosine_raw=self.identity.cosine(ar,br))
        detections=[d for d in gd['detections'] if d['score'] >= .5 and d['short_side_at_512'] >=32]
        closest=max(detections,key=lambda d:d['head_iou']) if detections else None
        gate=bool((metrics['pose_pck_002'] or 0)>=.70 and closest is not None and closest['head_iou']>=.20 and closest['center_in_expected_head'])
        return {**row,'metrics':metrics,'face_detection':{'source':ad,'target':bd,'generated':gd},
                'face_boxes_normalized':{'source':abox,'target':bbox},'closest_face':closest,'body_gate_passed':gate}

def overlay(row, path):
    image=Image.open(row['image']).convert('RGB'); draw=ImageDraw.Draw(image)
    size=np.array([image.width,image.height,image.width,image.height])
    target=np.asarray(row['face_boxes_normalized']['target'])*size
    draw.rectangle(tuple(target),outline='red',width=2)
    for detection in row['face_detection']['generated']['detections']:
        box=np.asarray(detection['box'])*image.width/512
        draw.rectangle(tuple(box),outline='lime',width=2)
    image.save(path)

def select():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists(): return
    metric=Metrics(); original=read_pairs(RUN_ROOT/'train_pairs.jsonl')
    excluded=[]; candidates=[]
    final=read_pairs(ROOT/'datasets/iPER/iper_sampled_6src64tgt/splits/baseline_50_eval_pairs.jsonl')
    final_ids={(m['source_appearance'],m['source_stem'],m['target_stem']) for p in final for m in [pair_metadata(p)]}
    for index,pair in enumerate(original):
        meta=pair_metadata(pair)
        if (meta['source_appearance'],meta['source_stem'],meta['target_stem']) in final_ids:
            raise AssertionError('Final test sample in training candidates')
        if pair['appearance']=='001_10':
            excluded.append({'index':index,'reason':'adjacent near-identical frames'});continue
        a,b=(open_rgb(Path(meta[f'{w}_rgb'])) for w in ('source','target'))
        ad,_,_=face_detection(metric.identity,a,metric.box(meta['source_appearance'],meta['source_stem']))
        bd,_,_=face_detection(metric.identity,b,metric.box(meta['target_appearance'],meta['target_stem']))
        source_pose=metric.pose.compare(metric.pose.detect(a),metric.points(meta['target_appearance'],meta['target_stem']))
        if not ad['reliable'] or not bd['reliable'] or (source_pose['pose_pck_002'] or 0)>=.9:
            excluded.append({'index':index,'reason':'unreliable reference face or near-identical pose',
                             'source_detection':ad,'target_detection':bd,'source_pose':source_pose});continue
        candidates.append({**pair,'diagnostic_original_index':index})
    if not candidates: raise RuntimeError('No eligible original training candidate')
    pairs=OUT/'candidate_pairs.jsonl';pairs.write_text(''.join(json.dumps(p)+'\n' for p in candidates))
    rows=generate('selection_body',pairs)
    measured=[metric.evaluate(row,candidates[row['pair_index']]) for row in rows]
    ranked=sorted(measured,key=lambda r:(-(r['closest_face']['head_iou'] if r['closest_face'] else -1),
                                       -(r['metrics']['pose_pck_002'] or 0),candidates[r['pair_index']]['diagnostic_original_index']))
    selected=candidates[ranked[0]['pair_index']]
    (OUT/'single_pair.jsonl').write_text(json.dumps(selected)+'\n')
    json_write(OUT/'selection.json',{'selected':selected,'selected_baseline':ranked[0],'ranked':ranked,'excluded':excluded,
               'final_test_set_used':False,'sorting':['head_iou descending','PCK descending','original index ascending']})
    print(json.dumps({'selected':selected['appearance'],'initial_body_gate':ranked[0]['body_gate_passed'],
                      'pck':ranked[0]['metrics']['pose_pck_002']},ensure_ascii=False),flush=True)

def diagnose():
    select()
    if (OUT/'diagnosis_metrics.json').exists():return
    pairs=OUT/'single_pair.jsonl'; pair=read_pairs(pairs)[0]; metric=Metrics(); rows=[]
    cases=[('body', 'body','step',1.,True),('face_current','body_face','step',1.,True),
           ('face_quarter','body_face','step',.25,True),('face_sigma','body_face','sigma',1.,True),
           ('face_no_cache','body_face','step',1.,False),('face_sigma_no_cache','body_face','sigma',1.,False)]
    for label,mode,progress,strength,kv in cases:
        row=generate('diagnosis/'+label,pairs,FACE,mode=mode,progress=progress,strength=strength,kv=kv)[0]
        result=metric.evaluate(row,pair);result['case']=label
        overlay(result,OUT/'diagnosis'/f'{label}_overlay.png'); rows.append(result)
    json_write(OUT/'diagnosis_metrics.json',rows)

def train_args(branch,steps,checkpoint,resume=False):
    directory=OUT/f'{branch}_train'
    saves=[0,100,300,500,1000] if branch=='body' else [0,100,300,500,1000,2000]
    return [ROOT/'train_iper_qwen21_full.py','--branch',branch,'--train-pairs',OUT/'single_pair.jsonl',
            '--val-pairs',RUN_ROOT/'val_pairs.jsonl','--output',directory,'--max-steps',steps,
            '--resume' if resume else '--init-checkpoint',checkpoint,
            '--save-every',100,'--save-steps',*[s for s in saves if s<=steps],
            '--diagnostic-probes','--verify-resume','--require-active-gradients']

def train():
    diagnose()
    if (OUT/'training_decision.json').exists():return
    pairs=OUT/'single_pair.jsonl';pair=read_pairs(pairs)[0]; metric=Metrics();body_checkpoint=BODY; stage_rows=[]
    initial=generate('body_stages/step_000000',pairs,BODY)[0]
    initial=metric.evaluate(initial,pair);initial['added_step']=0;stage_rows.append(initial)
    passed=initial['body_gate_passed']
    if not passed:
        for maximum in (500,1000):
            resumed=maximum==1000
            source=OUT/'body_train/checkpoint_last.pt' if resumed else BODY
            command(f'body_train_to_{maximum}',train_args('body',maximum,source,resume=resumed))
            for step in ([100,300,500] if maximum==500 else [1000]):
                checkpoint=OUT/'body_train'/f'checkpoint_step_{step:06d}.pt'
                row=generate(f'body_stages/step_{step:06d}',pairs,checkpoint)[0]
                measured=metric.evaluate(row,pair);measured['added_step']=step;stage_rows.append(measured)
            passed=stage_rows[-1]['body_gate_passed']
            body_checkpoint=OUT/'body_train'/f'checkpoint_step_{maximum:06d}.pt'
            json_write(OUT/'body_stage_metrics.json',stage_rows)
            if passed: break
    json_write(OUT/'body_stage_metrics.json',stage_rows)
    decision={'body_gate_passed':passed,'body_checkpoint':str(body_checkpoint),
              'body_added_steps':stage_rows[-1]['added_step'],'face_started':False,'final_test_set_used':False}
    if passed:
        decision['face_started']=True
        json_write(OUT/'training_running.json',decision)
        command('face_train_to_2000',train_args('face',2000,body_checkpoint))
        face_rows=[]
        for step in (0,100,300,500,1000,2000):
            row=generate(f'face_stages/step_{step:06d}',pairs,OUT/'face_train'/f'checkpoint_step_{step:06d}.pt',mode='body_face')[0]
            measured=metric.evaluate(row,pair);measured['added_step']=step;face_rows.append(measured)
        json_write(OUT/'face_stage_metrics.json',face_rows)
    else:
        decision['reason']='Body position gate unmet after allowed 1000 updates; Face long training not started'
    json_write(OUT/'training_decision.json',decision)
    print(json.dumps(decision,ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('select','diagnose','train'),default='train')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'original_protection.json').exists():
        json_write(OUT/'original_protection.json',{str(p):sha(p) for p in (BODY,FACE)})
    {'select':select,'diagnose':diagnose,'train':train}[args.phase]()
    for path,expected in json.loads((OUT/'original_protection.json').read_text()).items():
        if sha(path)!=expected:raise AssertionError('Original checkpoint changed')

if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        OUT.mkdir(parents=True,exist_ok=True)
        json_write(OUT/'failure.json',{'error':str(error),'traceback':traceback.format_exc()})
        raise
