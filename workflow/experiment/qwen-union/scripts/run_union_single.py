#!/usr/bin/env python3
"""加载官方 Qwen-Image-2.1-Fun-Controlnet-Union 模型并执行单样本六组正式推理。

六组实验定义：
| 组别 | 模式 | A 图 | Mask | Pose 条件 | Prompt |
| T1  | Control Only    | 无 | 无                  | B 原生布局        | 统一 PE |
| T2a | Control Only    | 无 | 无                  | source_person 对齐 | 统一 PE |
| T2b | Control Only    | 无 | 无                  | source_torso 对齐  | 统一 PE |
| T3  | Inpaint+Control | A  | mask_native         | B 原生布局        | 统一 PE |
| T4a | Inpaint+Control | A  | mask_source_person  | source_person 对齐 | 统一 PE |
| T4b | Inpaint+Control | A  | mask_source_torso   | source_torso 对齐  | 统一 PE |

参数全组固定：
- 采样步数：40
- 随机种子：42
- true_cfg_scale：1.0
- control_context_scale：1.0
- use_kv_cache：False
- 画布尺寸：(640, 928)
- 精度：bfloat16
- 显存调度：model_group_offload
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image
import torch

# 挂载 VideoX-Fun 与项目依赖
EXP_ROOT = Path(__file__).resolve().parents[1]
VIDEOX_DIR = Path("/home/shangguanrz/project/pic-edit/experiments/6.qwen-controlnet union/source/VideoX-Fun")
BASE_MODEL_DIR = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1")
UNION_CKPT_PATH = Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1/models/Qwen-Image-2.1-Fun-Controlnet-Union/Qwen-Image-2.1-Fun-Controlnet-Union.safetensors")
CONFIG_PATH = VIDEOX_DIR / "config/qwenimage21/qwenimage21_control.yaml"


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stub_unrelated_optional():
    """屏蔽无关音频/文本依赖 librosa, ftfy，防止 VideoX-Fun 全量导入时崩溃。"""
    import importlib.util
    import types
    from importlib.machinery import ModuleSpec

    for name in ("librosa", "ftfy", "soxr"):
        if importlib.util.find_spec(name) is not None:
            continue
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__file__ = f"<stub:{name}>"
        mod.__spec__ = ModuleSpec(name, loader=None, is_package=True)

        def _dummy(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], str):
                return args[0]
            return None

        def _getattr(attr, _n=name):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _dummy

        mod.__getattr__ = _getattr
        sys.modules[name] = mod


def load_pipeline():
    """加载官方 VideoX-Fun QwenImage21ControlPipeline 并严格审计权重。"""
    stub_unrelated_optional()
    if str(VIDEOX_DIR) not in sys.path:
        sys.path.insert(0, str(VIDEOX_DIR))

    from diffusers import FlowMatchEulerDiscreteScheduler
    import yaml
    from safetensors.torch import load_file
    from videox_fun.models import (AutoencoderKLQwenImage21,
                                   Qwen3VLForConditionalGeneration,
                                   Qwen3VLProcessor,
                                   QwenImage21ControlTransformer2DModel)
    from videox_fun.pipeline.pipeline_qwenimage21_control import QwenImage21ControlPipeline
    from videox_fun.utils import apply_gpu_memory_mode

    dtype = torch.bfloat16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f">> [Model Load] 加载 Base Transformer: {BASE_MODEL_DIR / 'transformer'}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    kwargs = cfg.get("transformer_additional_kwargs", {})
    transformer = QwenImage21ControlTransformer2DModel.from_pretrained(
        str(BASE_MODEL_DIR), subfolder="transformer", low_cpu_mem_usage=True,
        torch_dtype=dtype, transformer_additional_kwargs=kwargs,
    ).to(dtype)

    print(f">> [Model Load] 加载 Union 权重: {UNION_CKPT_PATH}")
    state = load_file(str(UNION_CKPT_PATH))
    state = state["state_dict"] if "state_dict" in state else state

    missing, unexpected = transformer.load_state_dict(state, strict=False)

    named_params = dict(transformer.named_parameters())
    ctrl_keys_ckpt = sorted(k for k in state if k.startswith("control_blocks.") or k.startswith("control_img_in."))
    print(f"   Checkpoint 中控制层权重数: {len(ctrl_keys_ckpt)}")
    print(f"   Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    # 验证全部预期控制层权重均已加载且非零
    ctrl_nonzero = sum(
        1 for k in ctrl_keys_ckpt
        if k in named_params and float(named_params[k].detach().float().abs().sum()) > 0.0
    )
    print(f"   已恢复且非零的控制参数数: {ctrl_nonzero} / {len(ctrl_keys_ckpt)}")
    if ctrl_nonzero != len(ctrl_keys_ckpt) or ctrl_nonzero != 180:
        raise RuntimeError(f"ControlNet 权重加载不完整或存在全零参数！有效数: {ctrl_nonzero}, 预期: 180")

    print(f">> [Model Load] 加载 VAE, Processor, Text Encoder...")
    vae = AutoencoderKLQwenImage21.from_pretrained(str(BASE_MODEL_DIR), subfolder="vae").to(dtype)
    processor = Qwen3VLProcessor.from_pretrained(str(BASE_MODEL_DIR), subfolder="processor")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        str(BASE_MODEL_DIR), subfolder="text_encoder", torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(str(BASE_MODEL_DIR), subfolder="scheduler")

    print(f">> [Model Load] 组装官方 QwenImage21ControlPipeline 并应用 model_group_offload...")
    pipeline = QwenImage21ControlPipeline(
        vae=vae, text_encoder=text_encoder, processor=processor,
        transformer=transformer, scheduler=scheduler
    )

    apply_gpu_memory_mode(
        pipeline, "model_group_offload", device, dtype,
        exclude_module_name=["img_in", "txt_in", "time_text_embed", "modulation"]
    )
    audit_info = {
        "base_model": str(BASE_MODEL_DIR),
        "union_checkpoint": str(UNION_CKPT_PATH),
        "union_sha256": sha256_file(UNION_CKPT_PATH),
        "control_keys_count": len(ctrl_keys_ckpt),
        "control_nonzero_count": ctrl_nonzero,
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected)
    }
    return pipeline, audit_info


def main():
    parser = argparse.ArgumentParser(description="Run Official Qwen-Image-2.1-Fun-Controlnet-Union Inference")
    parser.add_argument("--arms", default="T1,T2a,T2b,T3,T4a,T4b",
                        help="运行的实验组列表 (逗号分隔)")
    parser.add_argument("--steps", type=int, default=40, help="推理步数 (固定40)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (固定42)")
    parser.add_argument("--true-cfg", type=float, default=1.0, help="CFG 强度 (固定1.0)")
    parser.add_argument("--control-scale", type=float, default=1.0, help="控制分支强度 (固定1.0)")
    parser.add_argument("--controls-dir", type=Path, default=EXP_ROOT / "data/prepared_controls",
                        help="预处理控制图目录")
    parser.add_argument("--prompt-file", type=Path, default=EXP_ROOT / "input/prompt_pe_union.txt",
                        help="编译冻结后的 PE Prompt 文件")
    parser.add_argument("--source-image", type=Path,
                        default=Path("/home/shangguanrz/project/pic-edit/workflow/runs/test/hard_05/inputs/A_reference.png"),
                        help="A 源图路径")
    parser.add_argument("--output", type=Path, default=EXP_ROOT / "output",
                        help="输出目录")
    parser.add_argument("--dry-run", action="store_true", help="演练模式，仅校验配置不跑扩散")
    parser.add_argument("--resume", action="store_true", help="若已有有效产物且校验通过则跳过")
    args = parser.parse_args()

    arms_requested = [a.strip() for a in args.arms.split(",") if a.strip()]
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 检查并读取控制条件与 Prompt
    if not args.prompt_file.is_file():
        raise FileNotFoundError(f"未找到冻结 Prompt: {args.prompt_file}")
    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_sha256 = sha256_file(args.prompt_file)

    manifest_path = args.controls_dir / "prep_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"未找到控制条件 manifest: {manifest_path}，请先运行 prepare_union_controls.py")
    prep_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    w_canvas, h_canvas = prep_manifest["canvas_size"]

    # 定义六组实验配置字典
    arm_specs = {
        "T1": {
            "mode": "control_only",
            "layout": "native",
            "pose_path": args.controls_dir / "pose_native.png",
            "mask_path": None,
            "image_path": None,
            "desc": "Control Only + B 原生布局"
        },
        "T2a": {
            "mode": "control_only",
            "layout": "source_person",
            "pose_path": args.controls_dir / "pose_source_person.png",
            "mask_path": None,
            "image_path": None,
            "desc": "Control Only + source_person 框对齐"
        },
        "T2b": {
            "mode": "control_only",
            "layout": "source_torso",
            "pose_path": args.controls_dir / "pose_source_torso.png",
            "mask_path": None,
            "image_path": None,
            "desc": "Control Only + source_torso 躯干对齐"
        },
        "T3": {
            "mode": "inpaint_control",
            "layout": "native",
            "pose_path": args.controls_dir / "pose_native.png",
            "mask_path": args.controls_dir / "mask_native.png",
            "image_path": args.source_image,
            "desc": "Inpaint + Control + B 原生布局"
        },
        "T4a": {
            "mode": "inpaint_control",
            "layout": "source_person",
            "pose_path": args.controls_dir / "pose_source_person.png",
            "mask_path": args.controls_dir / "mask_source_person.png",
            "image_path": args.source_image,
            "desc": "Inpaint + Control + source_person 框对齐"
        },
        "T4b": {
            "mode": "inpaint_control",
            "layout": "source_torso",
            "pose_path": args.controls_dir / "pose_source_torso.png",
            "mask_path": args.controls_dir / "mask_source_torso.png",
            "image_path": args.source_image,
            "desc": "Inpaint + Control + source_torso 躯干对齐"
        }
    }

    print("==================================================================")
    print("      Qwen-Image-2.1-Fun-Controlnet-Union 正式实验执行矩阵")
    print("==================================================================")
    print(f"Canvas 尺寸: {w_canvas}x{h_canvas} | 步数: {args.steps} | Seed: {args.seed}")
    print(f"Prompt SHA256: {prompt_sha256}")
    print(f"请求执行组别: {arms_requested}")

    # 预检查待执行组的输入文件完整性
    for arm in arms_requested:
        if arm not in arm_specs:
            raise ValueError(f"未知组别: {arm}")
        spec = arm_specs[arm]
        if not spec["pose_path"].is_file():
            raise FileNotFoundError(f"[{arm}] 缺少 Pose 图: {spec['pose_path']}")
        if spec["mask_path"] and not spec["mask_path"].is_file():
            raise FileNotFoundError(f"[{arm}] 缺少 Mask 图: {spec['mask_path']}")
        if spec["image_path"] and not spec["image_path"].is_file():
            raise FileNotFoundError(f"[{arm}] 缺少 Inpaint 源图: {spec['image_path']}")

    if args.dry_run:
        print("✅ [Dry Run] 所有输入图路径及配置检查通过，退出。")
        return

    # 2. 加载模型
    t_start_load = time.time()
    pipe, audit_info = load_pipeline()
    load_duration = time.time() - t_start_load
    print(f"✅ 模型全部加载就绪，耗时: {load_duration:.2f}s\n")

    summary_records = {}

    # 3. 逐组推理
    for idx, arm in enumerate(arms_requested, start=1):
        spec = arm_specs[arm]
        arm_dir = out_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        final_img_path = arm_dir / "final_image.png"
        record_json_path = arm_dir / "run_record.json"

        if args.resume and final_img_path.is_file() and record_json_path.is_file():
            print(f"[{idx}/{len(arms_requested)}] 跳过已存在的组别 {arm} ({spec['desc']})")
            summary_records[arm] = json.loads(record_json_path.read_text(encoding="utf-8"))
            continue

        print(f"------------------------------------------------------------------")
        print(f"[{idx}/{len(arms_requested)}] 开始执行 {arm}: {spec['desc']}")
        print(f"------------------------------------------------------------------")

        pose_im = Image.open(spec["pose_path"]).convert("RGB")
        inpaint_im = Image.open(spec["image_path"]).convert("RGB") if spec["image_path"] else None
        mask_im = Image.open(spec["mask_path"]).convert("L") if spec["mask_path"] else None

        # 校验尺寸一致性
        if pose_im.size != (w_canvas, h_canvas):
            raise ValueError(f"[{arm}] Pose 尺寸 {pose_im.size} 与 Canvas ({w_canvas}, {h_canvas}) 不一致！")
        if inpaint_im and inpaint_im.size != (w_canvas, h_canvas):
            raise ValueError(f"[{arm}] Inpaint 图尺寸 {inpaint_im.size} 与 Canvas 不一致！")
        if mask_im and mask_im.size != (w_canvas, h_canvas):
            raise ValueError(f"[{arm}] Mask 尺寸 {mask_im.size} 与 Canvas 不一致！")

        # 统一设置相同的种子
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device=pipe._execution_device).manual_seed(args.seed)

        t0 = time.time()
        with torch.no_grad():
            output_sample = pipe(
                prompt=prompt_text,
                image=inpaint_im,
                mask_image=mask_im,
                control_image=pose_im,
                height=h_canvas,
                width=w_canvas,
                num_inference_steps=args.steps,
                true_cfg_scale=args.true_cfg,
                control_context_scale=args.control_scale,
                use_kv_cache=False,
                generator=generator,
                output_type="pil"
            )

        gen_seconds = round(time.time() - t0, 2)
        peak_vram_gib = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

        out_img = output_sample.images[0]
        out_img.save(final_img_path)
        out_sha256 = sha256_file(final_img_path)

        rec = {
            "arm": arm,
            "status": "completed",
            "description": spec["desc"],
            "mode": spec["mode"],
            "layout": spec["layout"],
            "parameters": {
                "steps": args.steps,
                "seed": args.seed,
                "true_cfg_scale": args.true_cfg,
                "control_context_scale": args.control_scale,
                "use_kv_cache": False,
                "canvas_hw": [h_canvas, w_canvas]
            },
            "inputs": {
                "prompt_sha256": prompt_sha256,
                "pose_path": str(spec["pose_path"]),
                "pose_sha256": sha256_file(spec["pose_path"]),
                "mask_path": str(spec["mask_path"]) if spec["mask_path"] else None,
                "mask_sha256": sha256_file(spec["mask_path"]) if spec["mask_path"] else None,
                "inpaint_source": str(spec["image_path"]) if spec["image_path"] else None,
                "inpaint_sha256": sha256_file(spec["image_path"]) if spec["image_path"] else None
            },
            "output": {
                "path": str(final_img_path),
                "sha256": out_sha256,
                "size_wh": [out_img.width, out_img.height]
            },
            "performance": {
                "generation_seconds": gen_seconds,
                "peak_gpu_vram_gib": peak_vram_gib
            }
        }
        record_json_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_records[arm] = rec

        print(f"✅ {arm} 完成: 耗时 {gen_seconds}s | 显存峰值: {peak_vram_gib} GiB | 产物 SHA256: {out_sha256[:12]}...")

    # 保存整体执行摘要
    exec_summary_path = out_dir / "execution_summary.json"
    exec_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit": audit_info,
        "arms": summary_records
    }
    exec_summary_path.write_text(json.dumps(exec_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n🎉 全部六组推理执行完毕！执行摘要写入: {exec_summary_path}")

    # 强制退出，避免 Inductor 线程池导致脚本阻塞
    os._exit(0)


if __name__ == "__main__":
    main()
