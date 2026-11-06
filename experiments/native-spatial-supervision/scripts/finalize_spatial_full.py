"""Verify full v2 equals the human-reviewed settings and export a pinned binding."""
from collections import Counter
import argparse
import json
from pathlib import Path
import sys

ROOT=next(p for p in (Path.cwd(),*Path(__file__).resolve().parents) if (p/'src/data/canvas_contract.py').exists())
sys.path.insert(0,str(ROOT))
import torch
from src.pose_control.qwen21_full.native_v5.spatial_weights import read_spatial_components
from scripts.data.build_native_spatial_weights import sha,publish


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))
def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def validate_full_scope(baseline,approved,candidate,audit,baseline_ref,candidate_ref):
    if baseline_ref!=approved.get('config',{}).get('body_cache_index'):
        raise ValueError('Baseline must be the exact approved complete donor index')
    if set(baseline['entries'])!=set(candidate['entries']):
        raise ValueError('Full GT/Canvas coverage differs from approved donor')
    source=candidate['config']['frames']
    if source!=baseline['config']['frames']:
        raise ValueError('Full source manifest differs from approved complete donor')
    count=len(candidate['entries'])
    if (not audit.get('passed') or audit.get('index')!=candidate_ref or audit.get('frames')!=count or
            audit.get('source_frames')!=count or audit.get('source_manifest_sha256')!=source['sha256']):
        raise ValueError('Audit does not bind the complete source manifest and candidate')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline-index','approved-index','candidate-index','audit','approval','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4)
    old,approved,new=map(load,(args.baseline_index,args.approved_index,args.candidate_index))
    new_ref=ref(args.candidate_index);approval=load(args.approval);audit=load(args.audit)
    if approval.get('reviewed_index')!=ref(args.approved_index) or approval.get('source')!='direct_user_message':
        raise ValueError('Human-reviewed source binding differs')
    validate_full_scope(old,approved,new,audit,ref(args.baseline_index),new_ref)
    if sha(new['config']['frames']['path'])!=new['config']['frames']['sha256']:
        raise ValueError('Full source manifest changed after generation')
    if audit['padding_weight_max']!=0 or audit['zero_coefficients_baseline_max_error']>1e-6:
        raise ValueError('Full numerical acceptance failed')
    excluded={'frames','selected_frame_ids'}
    reviewed_settings={k:v for k,v in approved['config'].items() if k not in excluded}
    full_settings={k:v for k,v in new['config'].items() if k not in excluded}
    if reviewed_settings!=full_settings or full_settings.get('hand_policy')!='visible_region_v2':
        raise ValueError('Full generation changed human-reviewed settings')
    totals=Counter();reasons=Counter();videos=set()
    for i,(key,entry) in enumerate(new['entries'].items()):
        prior=old['entries'][key];meta=entry['metadata']
        before=read_spatial_components(prior['file'],gt_rgb_sha256=meta['gt_rgb_sha256'],target_canvas_id=meta['target_canvas_id'])
        after=read_spatial_components(entry['file'],gt_rgb_sha256=meta['gt_rgb_sha256'],target_canvas_id=meta['target_canvas_id'])
        if before['metadata']!=prior['metadata'] or after['metadata']!=meta:raise ValueError('Index metadata differs from NPZ')
        for name in ('body','joint','valid'):
            if not torch.equal(before[name],after[name]):raise ValueError('Non-hand component changed: '+meta['frame_id']+'/'+name)
        if key in approved['entries']:
            accepted=read_spatial_components(approved['entries'][key]['file'],gt_rgb_sha256=meta['gt_rgb_sha256'],target_canvas_id=meta['target_canvas_id'])
            if any(not torch.equal(accepted[n],after[n]) for n in ('body','joint','hand','valid')):
                raise ValueError('Full output differs from manually approved sample: '+meta['frame_id'])
            totals['approved_samples_exact']+=1
        old_hands=prior['metadata']['region_status']['hands'];new_hands=meta['region_status']['hands']
        totals['old_frames']+=int(any(h['enabled'] for h in old_hands.values()))
        totals['new_frames']+=int(any(h['enabled'] for h in new_hands.values()))
        for side,h in new_hands.items():
            totals['old_hands']+=int(old_hands[side]['enabled']);totals['new_hands']+=int(h['enabled'])
            totals['coarse_hands']+=int(h['coarse']['enabled']);totals['fine_hands']+=int(h['fine']['enabled'])
            if not h['coarse']['enabled']:reasons[h['coarse']['reason']]+=1
            if h['coarse']['reason'] in ('explicit_occluded_or_out_of_frame','hand_not_present'):
                totals['visibility_negative_hands']+=1
                if h['enabled']:raise ValueError('Visibility negative received hand enhancement')
        videos.add(meta['frame_id'].rsplit('_',1)[0])
        if (i+1)%200==0:print(json.dumps(dict(compared=i+1,total=len(new['entries']))),flush=True)
    if totals['approved_samples_exact']!=len(approved['entries']):raise ValueError('Approved sample coverage incomplete')
    report=dict(schema='native_spatial_full_v2_comparison_v1',baseline=ref(args.baseline_index),approved=ref(args.approved_index),
                candidate=new_ref,audit=ref(args.audit),approval=ref(args.approval),frames=len(new['entries']),videos=len(videos),
                counts=dict(totals),coarse_rejection_reasons=dict(reasons),non_hand_components_identical=True,
                human_reviewed_settings_identical=True,approved_samples_bit_exact=True,training_started=False)
    publish(args.output_root/'comparison_summary.json',report)
    binding=dict(schema='native_spatial_training_binding_v1',spatial_weights=new_ref,
                 loss=dict(velocity=1.,face=0.,hand=0.,spatial_body=.5,spatial_joint=.5,spatial_hand=1.,spatial_max=2.5),
                 usage='将空间索引绑定至所选训练与验证 Source，保留其余资产与划分配置；重新执行匹配的 preflight，再启动训练。',
                 human_review_approval=ref(args.approval),full_audit=ref(args.audit),training_started=False)
    publish(args.output_root/'training_binding.json',binding)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
