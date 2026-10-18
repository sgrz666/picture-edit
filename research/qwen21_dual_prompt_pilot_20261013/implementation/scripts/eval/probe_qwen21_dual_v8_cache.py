"""Diagnose native BF16 cache numerics; not a pilot quality evaluation."""
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.data.hi4d_smplx_dataset import Hi4DSMPLXDataset
from src.pose_control.qwen21_full.dual import DualV8Adapter
from src.pose_control.qwen21_full.dual.runtime import load_native,native_inputs,source_cache,flow_sample,prepared_for_qwen
from scripts.train.train_qwen21_dual_v8_pilot import cache_probe


def main():
    root=Path('/home/shangguanrz/project/pic-edit')
    data=root/'research/qwen21_dual_upgrade_20261011/pilot_controls_v2/generation.json'
    generation=Hi4DSMPLXDataset(data).generation().to('cuda',torch.float32)
    pipe=load_native(root/'models/Qwen-Image-2.1')
    cache_dir=root/'experiments/qwen21_dual_v8/gates_pair14_v4/native_cache'
    _,source=source_cache(pipe,generation,cache_dir)
    clean=torch.load(next(cache_dir.glob('target_*.pt')),map_location='cuda',weights_only=True)
    pipe.text_encoder.to('cpu');pipe.vae.to('cpu');gc.collect();torch.cuda.empty_cache()
    torch.manual_seed(14062)
    adapter=DualV8Adapter().to('cuda',torch.float32).eval()
    with torch.no_grad():
        for branches in adapter.output_heads.values():
            for head in branches.values():head[-1].weight.normal_(std=.001)
    with torch.no_grad():state=prepared_for_qwen(adapter,generation)
    sigma=torch.tensor([.65],device='cuda')
    rng=torch.Generator(device='cuda').manual_seed(42)
    noisy,_=flow_sample(clean,torch.randn(clean.shape,device=clean.device,dtype=clean.dtype,generator=rng),sigma)
    inputs=native_inputs(source,noisy,sigma)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    original=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    records=[]
    try:
        for name,reduced,math_only in [('default',original,False),('bf16_full_reduction',False,False),
                                      ('math_sdpa',False,True)]:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=reduced
            backend=sdpa_kernel([SDPBackend.MATH]) if math_only else nullcontext()
            with backend,torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                record=cache_probe(pipe.transformer,adapter,state,inputs,sigma,require_close=False)
            record.update(name=name,bf16_reduced_precision=reduced,math_sdpa=math_only,
                processor=type(pipe.transformer.transformer_blocks[0].attn.processor).__name__)
            records.append(record);print(json.dumps(record),flush=True)
    finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=original
    path=root/'experiments/qwen21_dual_v8/cache_precision_probe_v1.json'
    path.write_text(json.dumps(dict(task='native_cache_precision_diagnosis_random_adapter_heads_not_quality',records=records),indent=2)+'\n')


if __name__=='__main__':main()
