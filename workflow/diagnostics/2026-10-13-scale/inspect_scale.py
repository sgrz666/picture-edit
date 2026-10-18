"""Build evidence views without changing any workflow code or existing run."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
CASES = ROOT if (ROOT / 'hard_01').is_dir() else ROOT.parents[1] / 'runs'
FONT_PATH = next((p for p in ('C:/Windows/Fonts/arial.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf') if Path(p).is_file()), None)
FONT = ImageFont.truetype(FONT_PATH, 17) if FONT_PATH else ImageFont.load_default()
SMALL = ImageFont.truetype(FONT_PATH, 14) if FONT_PATH else ImageFont.load_default()


def sheet(names, output):
    columns = [('A source', 'inputs/A_reference.png'), ('B pose source', 'inputs/B_rgb.png'),
        ('Guide over A', 'geometry/A_poseguide_alignment.png'),
        ('Generated (same canvas)', 'generation/final_canvas.png')]
    width, height = 275, 370
    canvas = Image.new('RGB', (len(columns) * width, len(names) * (height + 32) + 30), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for col, (title, _) in enumerate(columns):
        draw.text((col * width + 8, 5), title, fill='black', font=FONT)
    for row, name in enumerate(names):
        y = 30 + row * (height + 32)
        draw.text((8, y + 3), name, fill='black', font=FONT)
        for col, (_, rel) in enumerate(columns):
            with Image.open(CASES / name / rel) as image:
                scaled = ImageOps.contain(image.convert('RGB'), (width - 8, height), Image.Resampling.LANCZOS)
                canvas.paste(scaled, (col * width + (width - scaled.width) // 2, y + 30))
    canvas.save(ROOT / output)


def length(p):
    return float(np.linalg.norm(p[1] - np.nanmean(p[[8, 11]], axis=0)))


def metrics(name):
    root = CASES / name
    camera = json.loads((root / 'geometry/B_camera.json').read_text())
    meta = camera['source_letterbox']
    a = np.asarray(camera['detections']['A']['bbox_xyxy']) * np.tile(meta['scale_xy'], 2)
    a += np.tile(meta['offset_xy'], 2)
    b = np.asarray(camera['detections']['B']['bbox_xyxy'])
    aw, ah = a[2:] - a[:2]
    bw, bh = b[2:] - b[:2]
    guide = np.load(root / 'geometry/B_keypoints.npz')
    evaluation = np.load(root / 'evaluation/final_keypoints.npz')
    p = guide['keypoints_generation_px']
    g = evaluation['dwpose134_px']
    offset = np.asarray(meta['content_box'][:2])
    g_canvas = g + offset
    s = camera['B_to_generation']['scale']
    result = {'run': name, 'A_box_canvas': a.tolist(), 'A_box_height_px': float(ah),
        'A_box_width_px': float(aw), 'B_box_width_px': float(bw), 'B_box_height_px': float(bh),
        'fit_width_scale': float(aw / bw), 'fit_height_scale': float(ah / bh), 'used_scale': s,
        'limiting_side': 'width' if aw / bw < ah / bh else 'height',
        'guide_torso_px': length(p), 'output_torso_px': length(g),
        'output_to_guide_torso': length(g) / length(p),
        'guide_torso_to_A_box_height': length(p) / ah,
        'output_torso_to_A_box_height': length(g) / ah,
        'guide_hips_canvas': np.nanmean(p[[8, 11]], axis=0).tolist(),
        'output_hips_canvas': np.nanmean(g_canvas[[8, 11]], axis=0).tolist(),
        'guide_neck_canvas': p[1].tolist(), 'output_neck_canvas': g_canvas[1].tolist(),
        'eval': json.loads((root / 'evaluation/eval.json').read_text())['summary']}
    return result


def preview(names):
    """Illustrate one candidate geometry change; these are NOT regenerated results."""
    observations = {x['run']:x for x in json.loads((ROOT/'source_pose_measurements.json').read_text())}
    edges = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
        (1,11),(11,12),(12,13),(2,5),(8,11),(0,14),(14,15),(0,16),(16,17)]
    titles = ['A: measured neck / pelvis', 'Existing guide over A',
        'Output: measured neck / pelvis', 'A scale + pelvis (guide ONLY)']
    w, h = 275, 400
    output = Image.new('RGB',(w*4,(h+30)*len(names)+60),'#eeeeee')
    d = ImageDraw.Draw(output)
    for i,t in enumerate(titles):d.text((i*w+5,5),t,fill='black',font=SMALL)
    d.text((5,28),'Last column is a 2D diagnostic preview, not a fix or a generated image.',fill='black',font=SMALL)
    records=[]
    for row,name in enumerate(names):
        r=CASES/name
        a=observations[name]['source']; g=observations[name]['generated']
        p=np.load(r/'geometry/B_keypoints.npz')['keypoints_generation_px']
        anchor=np.nanmean(p[[8,11]],axis=0)
        factor=a['torso_px']/length(p)
        shifted=np.asarray(a['pelvis_canvas']) + factor*(p-anchor)
        assert abs(length(shifted)-a['torso_px'])<1e-6
        assert np.allclose(np.nanmean(shifted[[8,11]],axis=0),a['pelvis_canvas'])
        with Image.open(r/'inputs/A_reference.png') as im:source=im.convert('RGB')
        with Image.open(r/'geometry/A_poseguide_alignment.png') as im:old=im.convert('RGB')
        with Image.open(r/'generation/final_canvas.png') as im:generated=im.convert('RGB')
        assert source.size == old.size == generated.size
        for image,obs in [(source,a),(generated,g)]:
            draw=ImageDraw.Draw(image)
            n,pelvis=tuple(obs['neck_canvas']),tuple(obs['pelvis_canvas'])
            draw.line([n,pelvis],fill='yellow',width=4)
            for x,y in [n,pelvis]:draw.ellipse((x-6,y-6,x+6,y+6),fill='yellow')
        with Image.open(r/'inputs/A_reference.png') as im:candidate=im.convert('RGB')
        draw=ImageDraw.Draw(candidate)
        for left,right in edges:
            if np.isfinite(shifted[[left,right]]).all():
                # Draw full coordinates; Pillow clips raster lines rather than moving joints.
                draw.line([tuple(shifted[left]),tuple(shifted[right])],fill=(255,70,210),width=3)
        for x,y in shifted[:18]:
            if np.isfinite([x,y]).all() and 0<=x<candidate.width and 0<=y<candidate.height:
                draw.ellipse((x-4,y-4,x+4,y+4),fill=(255,70,210))
        y=60+row*(h+30)
        d.text((5,y),name,fill='black',font=FONT)
        for col,im in enumerate([source,old,generated,candidate]):
            small=ImageOps.contain(im,(w-8,h),Image.Resampling.LANCZOS)
            output.paste(small,(col*w+(w-small.width)//2,y+25))
        records.append({'run':name,'scale_multiplier':factor,'anchor_A_pelvis':a['pelvis_canvas'],
            'torso_matches_A_measurement':True,'pelvis_matches_A_measurement':True,
            'finite_body_joints_outside_canvas':int(np.sum(np.isfinite(shifted[:18]).all(axis=1)&
                ((shifted[:18,0]<0)|(shifted[:18,0]>=candidate.width)|
                 (shifted[:18,1]<0)|(shifted[:18,1]>=candidate.height)))),
            'limits':'Uniform 2D scale illustration only; keeps B bone proportions; no generator validation'})
    output.save(ROOT/'a_scale_anchor_preview.jpg')
    (ROOT/'counterfactual_geometry.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


def check_evidence(names):
    checks=[]
    for name in names:
        r=CASES/name
        c=json.loads((r/'geometry/B_camera.json').read_text())
        meta=c['source_letterbox']
        a=np.asarray(c['detections']['A']['bbox_xyxy'])*np.tile(meta['scale_xy'],2)+np.tile(meta['offset_xy'],2)
        b=np.asarray(c['detections']['B']['bbox_xyxy'])
        scale=c['B_to_generation']['scale']
        assert abs(scale-min((a[2:]-a[:2])/(b[2:]-b[:2])))<1e-6
        points=np.load(r/'geometry/B_keypoints.npz')
        raw=points['keypoints_B_px']; aligned=points['keypoints_generation_px']
        finite=np.isfinite(raw).all(axis=1)&np.isfinite(aligned).all(axis=1)
        residual=float(np.max(np.abs((raw*scale+c['B_to_generation']['offset_xy']-aligned)[finite])))
        assert residual<1e-6
        gen=json.loads((r/'generation/generation.json').read_text())
        paths={'A_reference':r/'inputs/A_reference.png','B_3d_dwpose':r/'geometry/B_3d_dwpose.png'}
        for ref in gen['reference_images']:
            assert hashlib.sha256(paths[ref['role']].read_bytes()).hexdigest()==ref['sha256']
        recorded=json.loads((r/'generation/stage.json').read_text())['outputs']['final_canvas']
        assert hashlib.sha256((r/'generation/final_canvas.png').read_bytes()).hexdigest()==recorded['sha256']
        checks.append({'run':name,'recorded_scale_equals_bbox_fit':True,
            'affine_residual_px':residual,'generation_reference_hashes_match':True,
            'generated_canvas_hash_matches_stage':True})
    (ROOT/'integrity_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(f'{len(checks)}/{len(checks)} saved runs: scale, coordinates and image hashes verified')


if __name__ == '__main__':
    bad = ['hard_01', 'hard_02', 'hard_04', 'hard_05']
    good = ['hard_03', 'medium_01', 'medium_02']
    extra = ['hard_05', 'hard_05_torso', 'hard_05_manual']
    check_evidence(bad+good)
    sheet(bad, 'bad_cases.jpg')
    sheet(good, 'good_cases.jpg')
    sheet(extra, 'hard05_variants.jpg')
    rows = [metrics(name) for name in ['hard_01','hard_02','hard_03','hard_04','hard_05',
        'medium_01','medium_02','hard_05_torso','hard_05_manual']]
    (ROOT / 'metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    if (ROOT/'source_pose_measurements.json').exists():preview(bad)
    for row in rows:
        print(row['run'], 'limit', row['limiting_side'], 's', round(row['used_scale'],3),
            'guide/output torso', round(row['guide_torso_px'],1), round(row['output_torso_px'],1),
            'output/guide', round(row['output_to_guide_torso'],3),
            'guide/A-height', round(row['guide_torso_to_A_box_height'],3),
            'PCK',round(row['eval']['body_pck@0.2'],3))
