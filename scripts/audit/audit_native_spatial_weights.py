#!/usr/bin/env python3
"""Read-only full-cache, GT identity and Qwen64 numerical acceptance."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from PIL import Image
from src.data.canvas_contract import CanvasSpec
from src.pose_control.qwen21_full.native_v5.spatial_weights import read_spatial_index,read_spatial_components,spatial_token_weights,spatial_velocity_loss
from src.pose_control.qwen21_full.native_depth_runtime import native_velocity_loss
from scripts.data.build_native_spatial_weights import sha,digest,publish
from src.pose_control.qwen21_full.native_v5.spatial_masks import canvas_for


def audit(index_path,frames_path,*,allow_subset=False):
    torch.set_num_threads(4)
    reference=dict(path=str(Path(index_path).resolve()),sha256=sha(index_path))
    index,bound=read_spatial_index(reference)
    rows=[json.loads(line) for line in Path(frames_path).read_text().splitlines() if line.strip()]
    frames={r['frame_id']:r for r in rows}
    entries=list(index['entries'].values())
    cached_ids=[e['metadata']['frame_id'] for e in entries]
    if len(cached_ids)!=len(set(cached_ids)) or not set(cached_ids)<=set(frames):
        raise ValueError('Cache frame IDs duplicate or differ from source frames')
    if not allow_subset and set(cached_ids)!=set(frames):
        raise ValueError('Cache frame coverage differs from full manifest')
    result=dict(schema='native_spatial_audit_v1',timestamp_utc=datetime.now(timezone.utc).isoformat(),passed=True,
                frames=len(entries),source_frames=len(rows),index=bound,source_manifest_sha256=sha(frames_path),
                qwen_channels=64,padding_weight_max=0.,zero_coefficients_baseline_max_error=0.,
                hand_enabled_frames=0,hand_support_frames=0,weight_max=0.,loss_min=None,loss_max=None,
                all_gt_rgb_verified=True,all_component_sha_verified=True,all_canvas_verified=True,
                training_started=False,model_forward_executed=False)
    for i,entry in enumerate(entries):
        metadata=entry['metadata'];frame=frames[metadata['frame_id']]
        gt=sha(frame['rgb_path'])
        if gt!=frame['rgb_sha256'] or gt!=metadata['gt_rgb_sha256']:
            raise ValueError('GT RGB changed: '+frame['frame_id'])
        if metadata.get('source_frame_sha256')!=digest(frame):
            raise ValueError('Original source frame record changed: '+frame['frame_id'])
        maps=read_spatial_components(entry['file'],gt_rgb_sha256=gt,target_canvas_id=metadata['target_canvas_id'])
        if maps['metadata']!=metadata:
            raise ValueError('Index/NPZ metadata differs')
        spec=CanvasSpec.from_record(metadata['canvas'])
        with Image.open(frame['rgb_path']) as image:
            original_hw=(image.height,image.width)
        if spec.original_hw!=tuple(frame['original_hw']) or spec.original_hw!=original_hw:
            raise ValueError('Canvas original grid differs from frame/GT RGB: '+frame['frame_id'])
        generation=index.get('config',{})
        if 'canvas_long_side' in generation and spec.metadata()!=canvas_for(original_hw,generation['canvas_long_side']).metadata():
            raise ValueError('Canvas differs from declared generation policy: '+frame['frame_id'])
        components={k:maps[k] for k in ('body','joint','hand')};valid=maps['valid']
        enabled=any(frame.get('loss_mask',{}).get(k,False) and frame.get('hand_raw_pose_usable',{}).get(k,False) for k in ('hand_a','hand_b'))
        support=bool(maps['hand'].any())
        if support and not enabled:
            raise ValueError('Disabled hand received extra supervision: '+frame['frame_id'])
        result['hand_enabled_frames']+=int(enabled);result['hand_support_frames']+=int(support)
        weights=spatial_token_weights(components,valid,spec.token_hw,(.5,.5,1.))
        zero=spatial_token_weights(components,valid,spec.token_hw,(0.,0.,0.))
        coverage=F.interpolate(valid[:,None],size=spec.token_hw,mode='area').flatten(1)
        if not torch.equal(zero,coverage):
            raise ValueError('Zero spatial coefficients differ from actual Canvas coverage')
        if bool((weights[coverage==0]!=0).any()):
            raise ValueError('Padding has positive Flow loss weight')
        if weights.max()>2.500001:
            raise ValueError('Weights exceed configured cap')
        n=weights.shape[1]
        prediction=(torch.arange(n*64,dtype=torch.float32).reshape(1,n,64)%29)/29
        target=torch.zeros_like(prediction)
        baseline=native_velocity_loss(prediction,target,coverage)
        zero_loss=spatial_velocity_loss(prediction,target,zero)
        error=abs(float(baseline-zero_loss))
        if error>1e-6:
            raise ValueError('Qwen64 zero-weight baseline differs')
        loss=float(spatial_velocity_loss(prediction,target,weights))
        if not 0<=loss<1:
            raise ValueError('Nonfinite or invalid weighted Qwen64 acceptance loss')
        result['zero_coefficients_baseline_max_error']=max(result['zero_coefficients_baseline_max_error'],error)
        result['weight_max']=max(result['weight_max'],float(weights.max()))
        result['loss_min']=loss if result['loss_min'] is None else min(result['loss_min'],loss)
        result['loss_max']=loss if result['loss_max'] is None else max(result['loss_max'],loss)
        if (i+1)%100==0:
            print(json.dumps(dict(audited=i+1,total=len(entries))),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index',type=Path,required=True);p.add_argument('--frames',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-subset',action='store_true')
    args=p.parse_args()
    result=audit(args.index,args.frames,allow_subset=args.allow_subset)
    publish(args.output,result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
