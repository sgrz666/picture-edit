"""Measure actual new generations and build baseline/new comparisons using CPU DWPose."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps

HERE=Path(__file__).resolve().parent
WORKFLOW=HERE.parents[1]
ROOT=WORKFLOW/'runs/scale_anchor_20261013'
OLD=WORKFLOW/'diagnostics/2026-10-13-scale'
FONT_PATH=next((p for p in ['C:/Windows/Fonts/arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()),None)
FONT=ImageFont.truetype(FONT_PATH,16) if FONT_PATH else ImageFont.load_default()


def comparison(names,path):
    columns=['Source A','Previous generation','New guide over A','New generation']
    w,h=290,420
    result=Image.new('RGB',(w*4,len(names)*(h+32)+32),'#eeeeee')
    d=ImageDraw.Draw(result)
    for i,title in enumerate(columns):d.text((i*w+5,6),title,fill='black',font=FONT)
    for row,name in enumerate(names):
        baseline=name.split('_pose_consistent')[0]
        y=32+row*(h+32)
        d.text((5,y+3),name,fill='black',font=FONT)
        files=[WORKFLOW/'runs'/baseline/'inputs/A_reference.png',
            WORKFLOW/'runs'/baseline/'generation/final_canvas.png',
            ROOT/name/'geometry/A_poseguide_alignment.png',ROOT/name/'generation/final_canvas.png']
        for col,file in enumerate(files):
            with Image.open(file) as im:small=ImageOps.contain(im.convert('RGB'),(w-8,h),Image.Resampling.LANCZOS)
            result.paste(small,(col*w+(w-small.width)//2,y+28))
    result.save(path)


def prompt_comparison():
    images=[WORKFLOW/'runs/hard_02/inputs/A_reference.png',
        ROOT/'hard_02/generation/final_canvas.png',
        ROOT/'hard_02_pose_consistent/generation/final_canvas.png']
    titles=['Source A','A scale: original conflicting prompt','Same guide: microphone clause fixed']
    w,h=370,530
    sheet=Image.new('RGB',(w*3,h+35),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for col,(file,title) in enumerate(zip(images,titles)):
        draw.text((col*w+5,5),title,fill='black',font=FONT)
        with Image.open(file) as im:small=ImageOps.contain(im.convert('RGB'),(w-8,h),Image.Resampling.LANCZOS)
        sheet.paste(small,(col*w+(w-small.width)//2,35))
    sheet.save(HERE/'microphone_prompt_comparison.jpg')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',default=['hard_01','hard_02','hard_04','hard_05','medium_02'])
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('source_probe',OLD/'probe_source_pose.py')
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    detector=probe.session('yolox_l.onnx');pose=probe.session('dw-ll_ucoco_384.onnx')
    old={r['run']:r for r in json.loads((OLD/'source_pose_measurements.json').read_text())}
    for name in args.cases:
        baseline=name.split('_pose_consistent')[0]
        r=ROOT/name
        c=json.loads((WORKFLOW/'runs'/baseline/'geometry/B_camera.json').read_text())
        m=c['source_letterbox']
        anchor=np.asarray(c['detections']['A']['bbox_xyxy'])*np.tile(m['scale_xy'],2)+np.tile(m['offset_xy'],2)
        g,p,scores=probe.measure(r/'generation/final_canvas.png',anchor,detector,pose)
        a=old[baseline]['source'];before=old[baseline]['generated']
        info={'case':name,'source':a,'baseline_generated':before,'new_generated':g,
            'baseline_output_to_A_torso':before['torso_px']/a['torso_px'],
            'new_output_to_A_torso':g['torso_px']/a['torso_px'],
            'baseline_face_height_to_A':before['face_contour_height_px']/a['face_contour_height_px'],
            'new_face_height_to_A':g['face_contour_height_px']/a['face_contour_height_px'],
            'limits':'Observed 2D lengths, affected by pose/viewpoint and landmark error; manual checks of extra arms/persons required'}
        directory=r/'evaluation_dwpose';directory.mkdir(exist_ok=True)
        (directory/'measurements.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
        np.savez_compressed(directory/'keypoints.npz',generated_raw133=p,generated_scores=scores)
        print(name,'torso/A',round(info['baseline_output_to_A_torso'],3),'->',round(info['new_output_to_A_torso'],3),
            'face height/A',round(info['baseline_face_height_to_A'],3),'->',round(info['new_face_height_to_A'],3),flush=True)
    records={p.parents[1].name:json.loads(p.read_text()) for p in ROOT.glob('*/evaluation_dwpose/measurements.json')}
    (HERE/'measurements_summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    comparison(args.cases,HERE/'comparison.jpg')
    groups=[['hard_01','hard_02'],['hard_04','hard_05'],['medium_02']]
    for i,group in enumerate(groups):
        available=[n for n in group if (ROOT/n/'generation/final_canvas.png').exists()]
        if available:comparison(available,HERE/f'comparison_{i+1}.jpg')
    if (ROOT/'hard_02_pose_consistent/generation/final_canvas.png').exists():
        prompt_comparison()
        if (ROOT/'hard_01/generation/final_canvas.png').exists():
            comparison(['hard_01','hard_02_pose_consistent'],HERE/'recommended_comparison_1.jpg')


if __name__=='__main__':main()
