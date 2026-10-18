"""Source-only inference: no supervision or target RGB command-line options."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.hi4d_smplx_dataset import Hi4DSMPLXDataset,file_sha256
from src.pose_control.qwen21_full.dual.checkpoint import load_checkpoint
from src.pose_control.qwen21_full.dual.conditions import ControlStrengths
from src.pose_control.qwen21_full.dual.precision import precision_policy
from src.pose_control.qwen21_full.dual.runtime import DETAILED_HUG_PROMPT,generate,load_native


def checkpoint_record(path,metadata,payload):
    if path is None:
        return dict(checkpoint=None,checkpoint_step=None,checkpoint_sha256=None)
    if payload is None or type(payload.get('step')) is not int:
        raise ValueError('checkpoint payload must declare its actual step')
    return dict(checkpoint=str(path),checkpoint_step=payload['step'],checkpoint_sha256=file_sha256(path),
                checkpoint_architecture=metadata.get('architecture_version'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--generation',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('/home/shangguanrz/project/pic-edit/models/Qwen-Image-2.1'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=40)
    p.add_argument('--detailed-prompt',action='store_true')
    p.add_argument('--no-kv-cache',action='store_true')
    p.add_argument('--cfg',type=float,default=1.)
    p.add_argument('--geometry-strength',type=float,default=1.)
    p.add_argument('--appearance-strength',type=float,default=1.)
    p.add_argument('--contact-strength',type=float,default=1.)
    p.add_argument('--max-ratio',type=float)
    a=p.parse_args()
    generation=Hi4DSMPLXDataset(a.generation).generation(prompt=DETAILED_HUG_PROMPT if a.detailed_prompt else None,seed=a.seed)
    pipe=load_native(a.model)
    adapter=None;metadata=None;payload=None
    if a.checkpoint:
        adapter,metadata,payload=load_checkpoint(a.checkpoint,device='cuda');adapter.eval()
        if metadata['model_id']!=str(a.model.resolve()):raise ValueError('checkpoint native model differs')
    strengths=ControlStrengths(a.geometry_strength,a.appearance_strength,a.contact_strength,a.max_ratio)
    output=generate(pipe,generation,adapter,strengths,a.steps,a.cfg,not a.no_kv_cache)
    a.output.parent.mkdir(parents=True,exist_ok=True);output.save(a.output)
    record=dict(generation_manifest=str(a.generation),generation_sha256=file_sha256(a.generation),
        geometry_sha256=generation.geometry.geometry_hash,
        precision_policy=getattr(pipe,'last_dual_v8_precision_policy',None) or precision_policy(),
        **checkpoint_record(a.checkpoint,metadata,payload),
        target_geometry_known=adapter is not None and not strengths.disabled,prompt=generation.prompts[0],seed=a.seed,steps=a.steps,cfg=a.cfg,
        use_kv_cache=not a.no_kv_cache,strengths=vars(strengths),output_sha256=file_sha256(a.output),
        diagnostics=None if adapter is None else adapter.last_diagnostics,
        cfg_diagnostics=None if adapter is None else getattr(adapter,'last_inference_diagnostics',None))
    a.output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
