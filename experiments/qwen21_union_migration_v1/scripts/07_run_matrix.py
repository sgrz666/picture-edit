#!/usr/bin/env python
"""六组 × 四配置 × 两 seed = 48 张正式结果。

配置：native（当前原生 Qwen 工作流基线）、union 接入版 scale 0 / 0.5 / 1.0。
同一样本四组使用完全一致的参考图、控制图、提示词、画布与 seed；目标 RGB 不进入生成器。
统一 40 步、CFG 1、BF16、batch 1、关闭 KV cache、官方 group offload。
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time

import torch
from PIL import Image

from union_env import WORK, build_pipeline, verify_videox_checkout  # noqa: F401

EXP = pathlib.Path(__file__).resolve().parent.parent
CONFIGS = [("native", None), ("union_s0", 0.0), ("union_s05", 0.5), ("union_s10", 1.0)]
SEEDS = [42, 43]
# 门禁只看空闲显存，不改实验变量：本矩阵实测峰值 3.9 GiB（官方 group offload + batch 1），
# 留 12 GiB 余量既能跑起来，也不会把同卡的其他任务挤爆。24 GiB 是首轮过保守的取值。
MIN_FREE_GIB = 12.0


def wait_for_gpu(log, need=MIN_FREE_GIB, max_wait_s=6 * 3600):
    t0 = time.time()
    while True:
        free, total = torch.cuda.mem_get_info()
        if free / 2**30 >= need:
            return {"waited_s": round(time.time() - t0, 1), "free_GiB": round(free / 2**30, 2)}
        if time.time() - t0 > max_wait_s:
            raise RuntimeError(f"显存等待超过 {max_wait_s/3600:.0f} 小时仍不足 {need} GiB")
        time.sleep(120)


def completed_keys(records_path: pathlib.Path):
    """从既有 records.jsonl 里取出已成功产出的 (sample, config, seed)，用于续跑时跳过。"""
    if not records_path.exists():
        return set()
    keys = set()
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("out"):
            keys.add((r["sample"], r["config"], r["seed"]))
    return keys


def sample_inputs(entry: dict):
    """返回 (参考图列表, 控制图, 提示词, 画布 h/w)。目标 RGB 不在其中。"""
    files = entry["files"]
    if entry["kind"] == "single":
        refs = [Image.open(files["source"]["path"]).convert("RGB"),
                Image.open(files["pose_reference"]["path"]).convert("RGB")]
    else:
        refs = [Image.open(files["source_A"]["path"]).convert("RGB"),
                Image.open(files["source_B"]["path"]).convert("RGB")]
        per = [p for k, p in sorted(files.items()) if k.startswith("pose_reference_")]
        if per:
            refs += [Image.open(p["path"]).convert("RGB") for p in per]
    ctrl = Image.open(files["control"]["path"]).convert("RGB")
    wh = files["control"]["wh"]
    return refs, ctrl, entry["prompt"], int(wh[1]), int(wh[0])


def to_tensor(img: Image.Image, h: int, w: int):
    from videox_fun.utils import get_image_latent
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        t = get_image_latent(f.name, sample_size=[h, w])[:, :, 0]
    pathlib.Path(f.name).unlink(missing_ok=True)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--only", default=None, help="只跑部分样本，逗号分隔 sample 名")
    ap.add_argument("--configs", default="native,union_s0,union_s05,union_s10")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--min-free-gib", dest="min_free_gib", type=float, default=MIN_FREE_GIB,
                    help="空闲显存门禁（GiB）。实测峰值 3.9 GiB，默认 12 GiB 已留三倍余量")
    args = ap.parse_args()

    manifest = json.loads((EXP / "input_manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.only:
        keep = set(args.only.split(","))
        samples = [s for s in samples if s["sample"] in keep]
    want_cfgs = set(args.configs.split(","))
    seeds = [int(s) for s in args.seeds.split(",")]

    name = args.name or f"matrix_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = WORK / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    already = completed_keys(records_path)

    print(f"[matrix] run={out_dir} samples={len(samples)} configs={sorted(want_cfgs)} seeds={seeds} "
          f"steps={args.steps} expected={len(samples)*len([c for c in CONFIGS if c[0] in want_cfgs])*len(seeds)} "
          f"已在 records 里完成={len(already)}（续跑会跳过这些组合，不重复产出也不覆盖既有图片）",
          flush=True)

    done, gpu_waits, skipped = 0, [], 0

    def record(rec):
        nonlocal done
        done += 1
        with open(records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{done:02d}] " + json.dumps(rec, ensure_ascii=False), flush=True)

    def pending(cfg_name):
        return [(e, s) for e in samples for s in seeds if (e["sample"], cfg_name, s) not in already]

    # ---- 第一遍：native 基线 ----
    if "native" in want_cfgs and pending("native"):
        pipe, _ = build_pipeline(kind="native", memory_mode="model_group_offload")
        for entry in samples:
            refs, _ctrl, prompt, h, w = sample_inputs(entry)
            for seed in seeds:
                if (entry["sample"], "native", seed) in already:
                    skipped += 1
                    continue
                gate = wait_for_gpu(out_dir, need=args.min_free_gib); gpu_waits.append(gate)
                t0 = time.time()
                torch.cuda.reset_peak_memory_stats()
                try:
                    img = pipe(prompt=prompt, image=refs, height=h, width=w, num_inference_steps=args.steps,
                               true_cfg_scale=1.0, use_kv_cache=False,
                               generator=torch.Generator(device=pipe._execution_device).manual_seed(seed)
                               ).images[0]
                    p = out_dir / f"{entry['kind']}__{entry['sample']}__native__seed{seed}.png"
                    img.save(p)
                    record({"sample": entry["sample"], "kind": entry["kind"], "config": "native", "seed": seed,
                            "canvas_hw": [h, w], "seconds": round(time.time() - t0, 1),
                            "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                            "out": str(p), "prompt_sha256": entry["prompt_sha256"], "error": None})
                except Exception as e:
                    record({"sample": entry["sample"], "kind": entry["kind"], "config": "native", "seed": seed,
                            "canvas_hw": [h, w], "seconds": round(time.time() - t0, 1),
                            "out": None, "error": f"{type(e).__name__}: {e}"})
        del pipe
        gc.collect(); torch.cuda.empty_cache()

    # ---- 第二遍：Union 接入版三种强度 ----
    union_cfgs = [c for c in CONFIGS if c[0] in want_cfgs and c[1] is not None and pending(c[0])]
    if union_cfgs:
        pipe, audit = build_pipeline(kind="reference", memory_mode="model_group_offload")
        (out_dir / "weight_audit.json").write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
        for entry in samples:
            refs, ctrl, prompt, h, w = sample_inputs(entry)
            ctrl_t = to_tensor(ctrl, h, w)
            for cfg_name, scale in union_cfgs:
                for seed in seeds:
                    if (entry["sample"], cfg_name, seed) in already:
                        skipped += 1
                        continue
                    gate = wait_for_gpu(out_dir, need=args.min_free_gib); gpu_waits.append(gate)
                    t0 = time.time()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        img = pipe(prompt=prompt, reference_images=refs, control_image=ctrl_t,
                                   control_context_scale=scale, height=h, width=w,
                                   num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=False,
                                   generator=torch.Generator(device=pipe._execution_device).manual_seed(seed)
                                   ).images[0]
                        p = out_dir / f"{entry['kind']}__{entry['sample']}__{cfg_name}__seed{seed}.png"
                        img.save(p)
                        record({"sample": entry["sample"], "kind": entry["kind"], "config": cfg_name,
                                "scale": scale, "seed": seed, "canvas_hw": [h, w],
                                "seconds": round(time.time() - t0, 1),
                                "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                "out": str(p), "prompt_sha256": entry["prompt_sha256"],
                                "n_refs": len(refs), "error": None})
                    except Exception as e:
                        record({"sample": entry["sample"], "kind": entry["kind"], "config": cfg_name,
                                "scale": scale, "seed": seed, "canvas_hw": [h, w],
                                "seconds": round(time.time() - t0, 1), "out": None,
                                "error": f"{type(e).__name__}: {e}"})
        del pipe
        gc.collect(); torch.cuda.empty_cache()

    expected = len(samples) * len([c for c in CONFIGS if c[0] in want_cfgs]) * len(seeds)
    total_done = len(completed_keys(records_path))      # 续跑后以 records 里的实际产出为准
    summary = {"run": str(out_dir), "steps": args.steps, "seeds": seeds, "configs": sorted(want_cfgs),
               "expected_images": expected,
               "produced_this_invocation": done, "produced_total": total_done,
               "skipped_already_done": skipped, "min_free_gib": args.min_free_gib,
               "videox_commit": verify_videox_checkout(),
               "kv_cache": False, "cfg": 1.0, "dtype": "bfloat16",
               "gpu_waits": [g for g in gpu_waits if g["waited_s"] > 1],
               "complete": total_done >= expected}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
