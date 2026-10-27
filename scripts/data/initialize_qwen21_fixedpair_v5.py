#!/usr/bin/env python3
"""Strict, explicit common V5 initialization; no optimizer or training."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))


def adapt_face72_order(state):
    import torch
    prefix='legacy.face_preparer.'
    order=torch.tensor([*range(4),*range(55,72),*range(4,55)])
    embedding=prefix+'geometry_tokenizer.landmark_index.weight';stem=prefix+'spatial_encoder.stem.weight'
    if state[embedding].shape!=(72,32) or state[stem].shape!=(16,73,3,3):
        raise ValueError('Audited legacy Face72 tensor shape differs')
    result=dict(state);result[embedding]=state[embedding][order].clone()
    result[stem]=torch.cat((state[stem][:,order],state[stem][:,72:]),1).clone()
    return result,{'source_layout':'SMPLest-X137[65:137]:4 extras+51 interior+17 contour',
        'target_layout':'4 extras+DWPose68:17 contour+51 interior','permutation_new_to_old':order.tolist(),
        'adapted_tensors':[embedding,stem],
        'source_evidence':'scripts/precompute_tiktok_face_features_v65.py; original SMPLest-X joint_idx',
        'operation':'embedding row permutation; first72 heatmap input channels permutation; feather channel unchanged'}


def main():
    import torch
    from src.pose_control.qwen21_full.native_full_assets import read_record,sha256
    from src.pose_control.qwen21_full.native_full_migration import migrate_v3_to_v4
    from src.pose_control.qwen21_full.native_v5.migration import migrate_v4_to_v5
    from src.pose_control.qwen21_full.native_v5.runtime import NativeFullRuntimeV5
    from src.pose_control.qwen21_full.native_v5.contracts import NativeConfigV5
    from src.pose_control.qwen21_full.native_local_control import NativeLocalConfigV4
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--complete-ledger',action='store_true',help='New metadata-complete file; preserve existing initialization')
    args=p.parse_args();root=args.root;out=root/'initialization';torch.set_num_threads(4);torch.manual_seed(42)
    if args.complete_ledger:
        previous=out/'native_v5_init.pt';destination=out/'native_v5_init_v2.pt'
        if destination.exists():raise FileExistsError(destination)
        saved=torch.load(previous,map_location='cpu',weights_only=True,mmap=True)
        ledger=json.loads((out/'migration_report.json').read_text());ledger.pop('destination',None)
        ledger['source_v4_checkpoint']={'path':str(out/'full_v4.pt'),'sha256':sha256(out/'full_v4.pt')}
        ledger['previous_initialization']={'path':str(previous),'sha256':sha256(previous)}
        saved['metadata']['inheritance']=ledger
        with destination.open('xb') as f:torch.save(saved,f)
        rebuilt=NativeFullRuntimeV5.from_checkpoint(torch.load(destination,map_location='cpu',weights_only=True,mmap=True))
        if any(not torch.equal(v,rebuilt.state_dict()[k]) for k,v in saved['state_dict'].items() if torch.is_tensor(v)):
            raise ValueError('Metadata completion changed a weight')
        ledger['destination']={'path':str(destination),'sha256':sha256(destination)}
        with (out/'migration_report_v2.json').open('x') as f:json.dump(ledger,f,indent=2)
        print(json.dumps({'checkpoint':ledger['destination'],'weights_unchanged':True,'strict_roundtrip':True}));return
    if (out/'native_v5_init.pt').exists():raise FileExistsError('Never overwrite common initialization')
    face_path=ROOT.parent.parent/'nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_last.pt'
    # ROOT=/home/shangguanrz/project/pic-edit; nihaochen belongs to the user home.
    face_path=Path('/home/shangguanrz/nihaochen/qwen21_full/tiktok_face_v1_20261004/training/checkpoint_last.pt')
    original=torch.load(face_path,map_location='cpu',weights_only=False,mmap=True)
    if original.get('architecture')!='qwen21_full_v1' or original.get('step')!=3000:raise ValueError('Face candidate history differs')
    state,permutation=adapt_face72_order(original['adapter_state_dict'])
    local_file=root/'data/body_local_v4_v1/0374_local.npz'
    local=read_record({'path':str(local_file),'sha256':sha256(local_file),'schema':'native_local_condition_v4'},schema='native_local_condition_v4')
    keys=('basis_sha256','seed','coordinate','normalization','encoding','sides','mirror')
    bps={k:local.metadata[0]['bps'][0][0][k] for k in keys}
    basis={'expression_basis_sha256':local.metadata[0]['expression_basis_sha256'][0],
        'num_expression_coeffs':10,'jaw_convention':'local_axis_angle','face72_layout':'4extras+dwpose68'}
    evidence={'step':3000,'checkpoint_sha256':sha256(face_path),
        'training_log_sha256':sha256(face_path.parent/'summary_003000.json')}
    semantics={'policy':'encoder-only','target_bps_convention':bps,'target_face_parameter_basis':basis,
        'face_training_evidence':evidence,'face_landmark_adaptation':permutation}
    saved=torch.load(out/'dual_v3.pt',map_location='cpu',weights_only=True,mmap=True)
    v4=migrate_v3_to_v4(saved,{'state_dict':state},face_prefix='legacy.face_preparer.',allow_fresh_hand=True,
        semantics=semantics,config=NativeLocalConfigV4(),extra_discard_prefixes=('legacy.','latent_bridge.','text_bridge.','pooled_bridge.'))
    report=v4.inheritance_report
    report['source_face_checkpoint']={'path':str(face_path),'sha256':evidence['checkpoint_sha256'],'step':3000}
    report['explicit_face72_permutation']=permutation
    payload=v4.checkpoint_payload(metadata={'inheritance':report,'module_training_evidence':{'face':evidence},'acceptance':'not_run'})
    with (out/'full_v4.pt').open('xb') as f:torch.save(payload,f)
    v5=migrate_v4_to_v5(payload,config=NativeConfigV5(),allow_fresh_hand=True,allow_fresh_detail=True,warm_start_native_local=False)
    ledger={**v5.inheritance_report,'face72_permutation':permutation,'v4_migration':report,
        'same_initialization_for_tiktok_hi4d':True,'face_training_evidence':evidence}
    v5.inheritance_report=ledger
    final=v5.checkpoint_payload(metadata={'inheritance':ledger,'module_training_evidence':{'face':evidence},'seed':42,
        'runtime_acceptance':'not_run','same_initialization_for_both_experiments':True})
    destination=out/'native_v5_init.pt'
    with destination.open('xb') as f:torch.save(final,f)
    rebuilt=NativeFullRuntimeV5.from_checkpoint(torch.load(destination,map_location='cpu',weights_only=True,mmap=True))
    old=v5.state_dict();new=rebuilt.state_dict()
    if old.keys()!=new.keys() or any(not torch.equal(v,new[k]) for k,v in old.items() if torch.is_tensor(v)):
        raise ValueError('V5 strict checkpoint roundtrip differs')
    ledger.update(destination={'path':str(destination),'sha256':sha256(destination)},strict_roundtrip=True,
        optimizer_resumed=False,training_started=False)
    with (out/'migration_report.json').open('x') as f:json.dump(ledger,f,indent=2)
    with (out/'semantics.json').open('x') as f:json.dump(semantics,f,indent=2)
    print(json.dumps({'checkpoint':ledger['destination'],'strict_roundtrip':True,'face_step':3000,
        'hand':'fresh_unverified','source_detail':'fresh_unverified','training_started':False}))


if __name__=='__main__':main()
