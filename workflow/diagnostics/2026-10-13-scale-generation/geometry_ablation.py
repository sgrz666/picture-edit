"""A-measured uniform-scale/pelvis ablation; does not retarget individual bone lengths."""
from pathlib import Path
import sys

import numpy as np
from PIL import ImageDraw

WORKFLOW = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORKFLOW))
from pose_transfer.geometry import render_poseguide, valid_points


def fit_to_source_anchor(guide, source, scores, min_score=0.3):
    guide=np.asarray(guide,dtype=float)
    source=np.asarray(source,dtype=float)
    scores=np.asarray(scores,dtype=float)
    if guide.shape!=(134,2) or source.shape!=(133,2) or scores.shape!=(133,):
        raise ValueError('Expected guide DWPose-134 and source raw COCO wholebody-133')
    source_indices=[5,6,11,12]
    guide_indices=[2,5,8,11]
    if (not np.isfinite(source[source_indices]).all() or
        not np.isfinite(scores[source_indices]).all() or
        np.any(scores[source_indices]<min_score)):
        raise ValueError('A shoulder/hip anchors are missing or below confidence threshold')
    if not np.isfinite(guide[guide_indices]).all():
        raise ValueError('The saved guide has missing shoulder/hip anchors')
    a_shoulders=source[[5,6]].mean(0); a_pelvis=source[[11,12]].mean(0)
    b_shoulders=guide[[2,5]].mean(0); b_pelvis=guide[[8,11]].mean(0)
    a_torso=float(np.linalg.norm(a_shoulders-a_pelvis))
    b_torso=float(np.linalg.norm(b_shoulders-b_pelvis))
    if min(a_torso,b_torso)<=1 or not np.isfinite([a_torso,b_torso]).all():
        raise ValueError('Degenerate source or guide torso')
    scale=a_torso/b_torso
    aligned=a_pelvis+scale*(guide-b_pelvis)
    aligned[~np.isfinite(guide).all(1)]=np.nan
    return aligned,{'operation':'uniform scale about guide pelvis, then anchor at A pelvis',
        'torso_definition':'shoulder midpoint to hip midpoint on both sides',
        'scale_multiplier':scale,'A_torso_px':a_torso,'baseline_guide_torso_px':b_torso,
        'A_shoulder_midpoint':a_shoulders.tolist(),'A_pelvis':a_pelvis.tolist(),
        'baseline_guide_shoulder_midpoint':b_shoulders.tolist(),
        'baseline_guide_pelvis':b_pelvis.tolist(),
        'source_anchor_scores':scores[source_indices].tolist(),
        'confidence_threshold':min_score,
        'body_proportions':'B bone proportions retained; A torso scale and pelvis anchor only'}


def render_clipped_guide(points,size):
    points=np.asarray(points,dtype=float)
    image=render_poseguide(points,size)
    draw=ImageDraw.Draw(image)
    in_frame=valid_points(points,*size)
    edges=[(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),
        (1,11),(11,12),(12,13),(2,5),(8,11),(0,14),(14,15),(0,16),(16,17)]
    palette=[(255,50,50),(255,105,30),(255,165,0),(255,215,0),(70,220,90),
        (0,200,180),(30,144,255),(65,105,225),(138,43,226),(220,20,160),
        (255,70,130),(255,120,90)]

    def crossing(left,right,color,width):
        if np.isfinite(points[[left,right]]).all() and not (in_frame[left] and in_frame[right]):
            draw.line([tuple(points[left]),tuple(points[right])],fill=color,width=width)

    for i,(left,right) in enumerate(edges):crossing(left,right,palette[i%len(palette)],3)
    for left,right in [(10,21),(10,23),(13,18),(13,20),(18,19),(18,20),(21,22),(21,23)]:
        crossing(left,right,(255,190,40),2)
    finger_colors=[(255,70,70),(255,190,40),(70,230,100),(60,170,255),(205,90,255)]
    for wrist in (92,113):
        for digit in range(5):
            first=wrist+1+digit*4
            crossing(wrist,first,finger_colors[digit],2)
            for joint in range(3):crossing(first+joint,first+joint+1,finger_colors[digit],2)
    return image
