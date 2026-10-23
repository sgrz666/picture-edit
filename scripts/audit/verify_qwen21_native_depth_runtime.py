#!/usr/bin/env python3
"""Bounded real-Qwen7B depth audit with paired geometry; synthetic latent/text only.

No image generation, optimizer update or checkpoint write. Only new adapter
heads are perturbed to probe an active depth path; base/Full weights stay frozen.
"""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import inspect
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--full-checkpoint", type=Path, required=True)
    p.add_argument("--champ-checkpoint", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--depth-manifest", type=Path, required=True)
    p.add_argument("--max-pixels", type=int, default=512*512)
    p.add_argument("--min-free-gib", type=float, default=40.)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from torch.nn import functional as F
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
    from src.pose_control.qwen21_full.data_runtime import QwenFullDataset
    from src.data.geometry_pair_dataset_v2 import assert_pair_contract
    from src.pose_control.qwen21_full.native_geometry import Qwen21NativeGeometry
    from src.pose_control.qwen21_full.native_body import Qwen21NativeBody
    from src.pose_control.qwen21_full.native_depth import Qwen21NativeDepth
    from src.pose_control.qwen21_full.native_depth_runtime import NativeDepthRecords, NativeDepthRuntime
    from src.pose_control.qwen21_full.native_integration import install_native_body, bind_native_body
    from scripts.audit.audit_qwen21_native_geometry import _sha256

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the 7B audit")
    free, total = torch.cuda.mem_get_info()
    if args.min_free_gib <= 0 or free / 1024**3 < args.min_free_gib:
        raise RuntimeError(f"Need {args.min_free_gib} GiB free, have {free/1024**3:.2f}")
    started = time.perf_counter()
    torch.manual_seed(23)
    torch.set_num_threads(4)
    dataset = QwenFullDataset(args.pairs, resolution="native", dataset_type="tiktok_v3",
                             align_mode="aspect_preserving", max_pixels=args.max_pixels)
    sample = dataset[0]
    assert_pair_contract(sample, where="7B depth audit")
    records = NativeDepthRecords(args.depth_manifest, mode="relative_png")
    condition = records.condition(sample).to("cuda")
    grid = tuple(sample["target_token_hw"])
    n = grid[0] * grid[1]
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else [v] for k,v in sample.items()}
    coverage = F.avg_pool2d(batch["valid_image_mask"].cuda().float(), 16).flatten(1)
    before_stat = args.full_checkpoint.stat()
    source = torch.load(args.full_checkpoint, map_location="cpu", mmap=True, weights_only=False)
    geometry = Qwen21NativeGeometry.from_full_checkpoint(source)
    del source
    depth = Qwen21NativeDepth()
    inheritance = depth.load_champ_state_dict(torch.load(args.champ_checkpoint, map_location="cpu", weights_only=True))
    system = NativeDepthRuntime(geometry, Qwen21NativeBody(), depth).cuda().eval()
    with torch.no_grad():
        plain = system.geometry(system.geometry.condition_injector(
            normal_a=batch["normal"].cuda(), pose_heatmap_a=batch["pose_heatmap"].cuda(),
            part_onehot_a=batch["part_onehot"].cuda(), human_mask_a=batch["human_mask"].cuda(),
            smplx_global_a=batch["smplx_global"].cuda(), task_id=batch["task_id"].cuda()), target_token_hw=grid)
        initial_features = depth(plain, condition)
    geometry_zero_diff = float((initial_features.geometry-plain.geometry).abs().max())
    torch.cuda.reset_peak_memory_stats()
    model = QwenImage21Transformer2DModel.from_pretrained(str(args.base_model), subfolder="transformer",
            torch_dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True).eval().cuda()
    base_parameters = dict(model.named_parameters())
    versions = {k: v._version for k,v in base_parameters.items()}
    ref_hw = (16,16)
    ref_n = ref_hw[0]*ref_hw[1]
    text_length = ref_n//4+4
    img_mask = torch.zeros(1,text_length+n//4,device="cuda",dtype=torch.bool)
    img_mask[:,2:2+ref_n//4] = True
    img_mask[:,text_length:] = True
    values = {"hidden_states":torch.randn(1,ref_n+n,64,device="cuda",dtype=torch.bfloat16),
              "encoder_hidden_states":torch.randn(1,text_length,4096,device="cuda",dtype=torch.bfloat16),
              "encoder_hidden_states_mask":torch.ones(1,text_length,device="cuda",dtype=torch.bool),
              "timestep":torch.tensor([.5],device="cuda"),
              "img_shapes":[[(1,*ref_hw),(1,*grid)]],"img_mask":img_mask,"return_dict":False}
    with torch.no_grad():
        base = model(**values)[0]
    install_native_body(model, system.body)
    with torch.no_grad(), bind_native_body(model,initial_features,valid_mask=coverage):
        zero = model(**values)[0]
    zero_diff = float((zero.float()-base.float()).abs().max())
    # Perturb only fresh adapter parameters, never the inherited or base checkpoint.
    for head in system.body.output_heads:
        torch.nn.init.normal_(head.weight,std=.002)
    with torch.no_grad(), bind_native_body(model,plain,valid_mask=coverage):
        nodepth = model(**values)[0]
    with torch.no_grad(), bind_native_body(model,initial_features,valid_mask=coverage):
        depth_initial = model(**values)[0]
    zero_with_active_body = float((nodepth.float()-depth_initial.float()).abs().max())
    torch.nn.init.normal_(depth.projection.weight,std=.01)
    features = depth(plain,condition)
    model.enable_gradient_checkpointing()
    with bind_native_body(model,features,valid_mask=coverage):
        active = model(**values)[0]
    prefix_diff = float((active[:,:ref_n].detach().float()-nodepth[:,:ref_n].float()).abs().max())
    target_diff = float((active[:,ref_n:].detach().float()-nodepth[:,ref_n:].float()).abs().max())
    active[:,ref_n:].float().square().mean().backward()
    grads = {"depth_conv":float(depth.encoder.conv_in.weight.grad.norm()),
             "depth_attention":float(depth.encoder.attention.proj_in.weight.grad.norm()),
             "depth_projection":float(depth.projection.weight.grad.norm())}
    body_grads = [float(head.weight.grad.norm()) for head in system.body.output_heads]
    frozen = all(not v.requires_grad and v.grad is None and v._version==versions[k] for k,v in base_parameters.items())
    source_unchanged = before_stat.st_size == args.full_checkpoint.stat().st_size and before_stat.st_mtime_ns == args.full_checkpoint.stat().st_mtime_ns
    clean = model._native_body_execution.get() is None and model._native_body_binding.get() is None
    finite = bool(torch.isfinite(active).all())
    passed = (zero_diff==0 and geometry_zero_diff==0 and zero_with_active_body==0 and prefix_diff==0
              and target_diff>0 and frozen and source_unchanged and clean and finite
              and all(v>0 and torch.isfinite(torch.tensor(v)) for v in [*grads.values(),*body_grads]))
    torch.cuda.synchronize()
    root = Path(__file__).resolve().parents[2]
    result = {"schema":"qwen21_native_depth_7b_audit_v1","passed":bool(passed),
              "pair_id":sample["pair_metadata"]["pair_id"],"target_token_hw":list(grid),"target_tokens":n,
              "canvas":sample["target_canvas"],"depth_metadata":condition.metadata[0],
              "base_model":str(args.base_model),"layers":len(model.transformer_blocks),"hidden_dim":model.inner_dim,
              "zero_body_max_abs":zero_diff,"zero_depth_geometry_max_abs":geometry_zero_diff,
              "zero_depth_with_active_body_max_abs":zero_with_active_body,
              "active_prefix_max_abs":prefix_diff,"active_target_max_abs":target_diff,
              "depth_gradients":grads,"body_head_gradient_norms":body_grads,"base_frozen_unchanged":frozen,
              "full_checkpoint_stat_unchanged":source_unchanged,"context_clean":clean,"finite":finite,
              "free_gib_before":free/1024**3,"total_gib":total/1024**3,
              "peak_allocated_gib":torch.cuda.max_memory_allocated()/1024**3,"seconds":time.perf_counter()-started,
              "depth_parameters":sum(v.numel() for v in depth.parameters()),
              "champ_inheritance":inheritance,"torch":torch.__version__,
              "qwen_source_sha256":_sha256(Path(inspect.getfile(QwenImage21Transformer2DModel))),
              "source_sha256":{name:_sha256(root/name) for name in (
                  "src/pose_control/qwen21_full/native_depth.py","src/pose_control/qwen21_full/depth_condition.py",
                  "src/pose_control/qwen21_full/native_body.py","src/pose_control/qwen21_full/native_integration.py",
                  "third_party/champ_guidance/spatial_attention.py")},
              "limits":"Real paired geometry/depth, synthetic text/latents. New heads perturbed for gradients. No optimizer, sampling or image-quality claim."}
    with args.output.open("x",encoding="utf8") as stream:
        json.dump(result,stream,indent=2,default=str)
        stream.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("champ_inheritance","depth_metadata","canvas","source_sha256")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
