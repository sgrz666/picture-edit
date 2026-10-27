#!/usr/bin/env python3
"""Faithful contact sheet of existing inputs/baselines/results; no inference."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def main():
    from PIL import Image,ImageOps,ImageDraw
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=('tiktok','hi4d'),required=True)
    p.add_argument('--update',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ablations',action='store_true',help='Existing final branch-off images, without running inference')
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():raise FileExistsError('Never overwrite an existing panel')
    if not a.output.resolve().is_relative_to('/data/shangguanrz'):raise ValueError('New reports belong on the persistent data volume')
    spec=importlib.util.spec_from_file_location('panel_prep',ROOT/'scripts/data/prepare_qwen21_fixedpair_v5.py')
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);root=h.TROOT if a.kind=='tiktok' else h.HROOT
    selection=json.loads((root/'data/selection.json').read_text())
    baseline_path=max((root/'report').glob('*/baseline_generation.json'),key=lambda p:p.stat().st_mtime_ns)
    baseline=json.loads(baseline_path.read_text())['records'];inputs=json.loads((root/'data/native_v5_v1/qwen_records.json').read_text())
    rows=[];provenance=[];controls=None
    if a.ablations:
        if a.update!=300:raise ValueError('Final controls are bound to update300')
        control_path=max((root/'report').glob('*/final_controls.json'),key=lambda p:p.stat().st_mtime_ns)
        controls=json.loads(control_path.read_text())['records']
        variants=['native','local_off','depth_off']+(['interaction_off'] if a.kind=='hi4d' else [])
    for case in selection['pairs']:
        row=next(r for r in inputs if r['pair_id']==case['pair_id'])
        condition=json.loads(Path(row['qwen_condition']['path']).read_text())
        v4=[json.loads(s) for s in Path(row['v4_manifest']['path']).read_text().splitlines()]
        old=next(r for r in v4 if r['pair_id']==case['pair_id'])
        qwen=next(r for r in baseline if r['pair_id']==case['pair_id'] and r['variant']=='qwen_only')
        native=next(r for r in baseline if r['pair_id']==case['pair_id'] and r['variant']=='native')
        learned=root/f'training/output/images/update_{a.update:06d}'/case['role']/case['pair_id']/'native_seed42.png'
        paths=[Path(old['source_rgb']['path']),Path(condition['images'][1]['file']['path']),Path(qwen['path']),Path(native['path']),learned,Path(old['target_rgb']['path'])]
        if controls is not None:
            paths=[Path(qwen['path'])]+[Path(next(r for r in controls if r['pair_id']==case['pair_id'] and r['variant']==v and r['seed']==42)['path']) for v in variants]+[Path(old['target_rgb']['path'])]
        rows.append(paths);provenance.append({'pair_id':case['pair_id'],'role':case['role'],'files':[h.ref(x) for x in paths]})
    cell_w,cell_h=210,376
    panel=Image.new('RGB',(len(rows[0])*cell_w,2*(cell_h+48)),(245,245,245));draw=ImageDraw.Draw(panel)
    labels=['Source','Pose guide','Qwen only','V5 initialization',f'V5 update {a.update}','Target RGB (GT)']
    if controls is not None:labels=['Qwen only','V5 full','Local off','Depth off']+(['Interaction off'] if a.kind=='hi4d' else [])+['Target RGB (GT)']
    for r,paths in enumerate(rows):
        for c,path in enumerate(paths):
            with Image.open(path) as image:im=ImageOps.contain(image.convert('RGB'),(cell_w,cell_h))
            x=c*cell_w;y=r*(cell_h+48)
            draw.text((x+5,y+4),labels[c],fill='black')
            if c==0:draw.text((x+5,y+20),selection['pairs'][r]['role'],fill='black')
            panel.paste(im,(x+(cell_w-im.width)//2,y+40))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:panel.save(f,format='PNG')
    h.write(a.output.with_suffix('.json'),{'kind':a.kind,'update':a.update,'seed':42,'sampling_steps':20,'inference_branch_off_comparison':a.ablations,
        'baseline_report':h.ref(baseline_path),'rows':provenance,'panel':h.ref(a.output)})
    print(json.dumps({'panel':str(a.output),'sha256':h.sha(a.output)}))


if __name__=='__main__':main()
