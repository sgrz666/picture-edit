"""Read-only checks of migrated tensors, frame geometry and frozen model inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from prepare import read,ref,sha,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.repo))
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from src.pose_control.qwen21_full.native_v7.runtime import NativeFullRuntimeV7
    receipt=read(a.root/'data/initialization.json');selection=read(a.root/'data/selection.json');geometry=read(a.root/'data/geometry.json')
    for name in ('source','initialization'):
        if ref(receipt[name]['path'])!=receipt[name]:raise ValueError('Pinned weight SHA differs')
    old=torch.load(receipt['source']['path'],map_location='cpu',mmap=True,weights_only=True)
    new=torch.load(receipt['initialization']['path'],map_location='cpu',mmap=True,weights_only=True)
    runtime=NativeFullRuntimeV7.from_checkpoint(new); del runtime
    inherited=0;numel=0
    for key,value in old['state_dict'].items():
        if torch.is_tensor(value):
            if key not in new['state_dict'] or not torch.equal(value,new['state_dict'][key]):
                raise ValueError('Trained Bilibili tensor changed during migration: '+key)
            inherited+=1;numel+=value.numel()
    if new['architecture']!='native_full_v7':raise ValueError('Latest Native V7 architecture required')
    old_steps=old['training_state']['global_update']
    if old_steps!=20000:raise ValueError('Source is not the selected Bilibili 20k endpoint')
    del old
    minimum_z=float('inf');max_score=0.;rgb_counts=0
    for fid,frame in selection['frames'].items():
        if sha(frame['rgb_path'])!=frame['rgb_sha256']:raise ValueError('Selected original RGB changed')
        raw=torch.load(geometry['frames'][fid]['parameters']['path'],map_location='cpu',weights_only=True)
        z=raw['mesh_cam'][...,2]
        if not torch.isfinite(z).all() or (z<=0).any():raise ValueError('Mesh leaves positive camera space: '+fid)
        minimum_z=min(minimum_z,float(z.min()));max_score=max(max_score,float(np.nanmax(read(frame['sdpose']['path'])['scores134_raw'])))
        rgb_counts+=1
    header=new['runtime_config'];model_repo=Path('/home/shangguanrz/project/pic-edit')
    module_path=Path(__file__).resolve().parents[1]/'source/native-builders/hia_training.py'
    if not module_path.exists():module_path=a.repo/'experiments/xiaohongshu-region-sdpose-sft-rl/source/native-builders/hia_training.py'
    spec=importlib.util.spec_from_file_location('xhs_import_audit',module_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.ROOT=model_repo
    importer=helper.Importer(model_repo/'add/tools/SMPLest-X/human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz',
        a.output.parent/'model-metadata',ref(__file__))
    expected=(header.get('face_parameter_basis') or {}).get('expression_basis_sha256')
    if expected is not None and expected!=importer.basis_sha:raise ValueError('Data/model face expression basis differs')
    frame=next(iter(selection['frames'].values()));directory=a.output.parent/('import-'+a.output.stem);directory.mkdir()
    asset,_=importer.convert(geometry['frames'][frame['frame_id']]['parameters'],{'scope':'actual one-frame import/FK audit'},frame,directory)
    result={'schema':'xhs_bilibili20k_native_v7_acceptance_v1','passed':True,'source':receipt['source'],
        'initialization':receipt['initialization'],'source_updates':old_steps,'architecture':new['architecture'],
        'inherited_tensors_exact':inherited,'inherited_tensor_elements_exact':numel,'raw_rgb_and_geometry_frames':rgb_counts,
        'minimum_camera_mesh_z':minimum_z,'maximum_raw_sdpose_score':max_score,'one_frame_import_fk_passed':True,
        'expression_basis_sha256':importer.basis_sha,'runtime_pose_config':header['pose_config'],
        'optimizer_inherited':False,'new_architecture_updates':0,'training_started':False}
    write(a.output,result)
    print(json.dumps({k:result[k] for k in ('passed','architecture','inherited_tensors_exact','raw_rgb_and_geometry_frames','maximum_raw_sdpose_score')}),flush=True)


if __name__=='__main__':main()
