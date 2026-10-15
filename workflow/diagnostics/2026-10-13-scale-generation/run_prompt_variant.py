"""One extra hard_02 run: keep source-anchor geometry, resolve microphone/arm conflict."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

from run_experiment import (HERE,ROOT,WORKFLOW,atomic_json,file_lock,prepare,read_json,
    sha256_file,stage_outputs,stage_valid,workflow_cli)

OLD_CLAUSE=('Keep the microphone gripped in her raised hand exactly as in <image1>, '
    'held at mouth height with the same finger grip and upward orientation while the '
    'other arm crosses the torso toward the opposite side.')
NEW_CLAUSE=('Keep exactly one microphone from <image1>, gripped in one of her two crossed '
    'hands near the opposite upper arm. Move the microphone with that hand into the pose '
    'shown in <image2>; it is held near chest height rather than at her mouth. '
    'The woman has exactly two arms and two hands.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    with file_lock(WORKFLOW/'.workflow.lock'):
        source,settings,source_request=prepare('hard_02',verify=True)
        target=ROOT/'hard_02_pose_consistent'
        original=(source/'prompt/generation_prompt.txt').read_text()
        if original.count(OLD_CLAUSE)!=1:raise RuntimeError('Original conflict clause is not unique')
        modified=original.replace(OLD_CLAUSE,NEW_CLAUSE)
        files=['inputs/A_reference.png','inputs/source_preprocess.json','geometry/B_3d_dwpose.png',
            'geometry/B_keypoints.npz','geometry/A_poseguide_alignment.png','geometry/operation.json']
        if not args.verify:
            for rel in files:
                (target/rel).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source/rel,target/rel)
            (target/'prompt').mkdir(exist_ok=True)
            (target/'prompt/generation_prompt.txt').write_text(modified,encoding='utf-8')
            atomic_json(target/'prompt/generation_prompt.json',{'source':'single_clause_manual_ablation',
                'source_path':str(source/'prompt/generation_prompt.txt'),
                'source_sha256':sha256_file(source/'prompt/generation_prompt.txt'),
                'sha256':sha256_file(target/'prompt/generation_prompt.txt'),
                'removed_clause':OLD_CLAUSE,'replacement_clause':NEW_CLAUSE,'PE_rerun':False})
        for rel in files:
            if sha256_file(source/rel)!=sha256_file(target/rel):raise RuntimeError(f'Geometry/A changed: {rel}')
        if (target/'prompt/generation_prompt.txt').read_text()!=modified:
            raise RuntimeError('Recorded prompt variant is not the single-clause change')
        paths={'run_dir':str(target),'A_reference':str(target/'inputs/A_reference.png'),
            'B_3d_dwpose':str(target/'geometry/B_3d_dwpose.png'),
            'source_preprocess':str(target/'inputs/source_preprocess.json'),
            'generation_prompt':str(target/'prompt/generation_prompt.txt'),
            'generation_prompt_provenance':str(target/'prompt/generation_prompt.json')}
        request=workflow_cli.stage_request('generate',settings,paths,target)
        request['identity']['experiment_geometry']=source_request['identity']['experiment_geometry']
        request['identity']['prompt_variant']={'source_prompt_sha256':sha256_file(source/'prompt/generation_prompt.txt'),
            'script_sha256':sha256_file(__file__),'operation':'replace only the contradictory microphone-arm clause'}
        record=target/'generation/stage.json'
        if not stage_valid(record,request,target):
            if args.verify:raise RuntimeError('Prompt-variant generation absent or stale')
            (target/'generation').mkdir(exist_ok=True)
            atomic_json(target/'generation/request.json',request)
            atomic_json(target/'experiment.json',{'schema':'source_anchor_prompt_ablation_v1',
                'status':'generating','source_geometry_case':str(source),
                'only_change_vs_source_geometry_case':'microphone-arm conflict clause','started_at_unix':time.time()})
            print('GENERATE hard_02_pose_consistent',flush=True)
            with (target/'generation/run.log').open('w') as log:
                done=subprocess.run([settings['qwen_python'],'-u',str(WORKFLOW/'generate_worker.py'),
                    '--request',str(target/'generation/request.json')],cwd=WORKFLOW,stdout=log,stderr=subprocess.STDOUT)
            if done.returncode:
                raise RuntimeError((target/'generation/run.log').read_text(errors='replace')[-5000:])
            if not stage_valid(record,request,target):raise RuntimeError('Generation artifact evidence failed validation')
        outputs=stage_outputs(record,target)
        if not args.verify:
            atomic_json(target/'experiment.json',{'schema':'source_anchor_prompt_ablation_v1',
                'status':'complete','source_geometry_case':str(source),'baseline_run':str(WORKFLOW/'runs/hard_02'),
                'only_change_vs_source_geometry_case':'microphone-arm conflict clause',
                'outputs':outputs,'finished_at_unix':time.time()})
        print(('VERIFIED ' if args.verify else 'COMPLETE ')+outputs['final_image'],flush=True)


if __name__=='__main__':main()
