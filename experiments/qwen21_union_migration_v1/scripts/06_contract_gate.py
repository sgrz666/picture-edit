#!/usr/bin/env python
"""接入关（第 2 关）：证明适配层没有偷改官方行为，并且参考图/控制序列真的对齐了。

比较对象是**单步 transformer 预测**（noise_pred），不是走完 scheduler 之后的 latent：
1 步 schedule（sigmas=[1.0]）下 FlowMatchEuler 的末位 sigma 会让 latent 变 NaN，
拿它做数值对照会把调度器边界当成模型差异。8 步 smoke 出图正常，可反证这一点。

分两个进程跑，因为 group offload 的 hook 在 pipeline 调用结束时会释放，
两个 wrapper 共用一套组件会互相拆掉对方的 hook：
  --phase official  只用原版官方 pipeline，把单步预测存盘
  --phase adapter   用 ReferenceUnionPipeline 跑同样输入并与上一步对照（A），再跑 B/C/D

判定口径按研究文档第 8.1 节：同一后端 BF16 记录 max/mean 差异并给容差，不把逐位相同当唯一条件。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from union_env import WORK, build_pipeline, verify_videox_checkout  # noqa: F401

PRED_OFFICIAL = WORK / "cache/gate_official_pred.npy"
BF16_ATOL = 2e-2


def capture_first_pred(pipe):
    """返回 (storage, undo)。storage["pred"] 是第一次 transformer 前向的输出。"""
    store = {}
    original = pipe.transformer.forward

    def spy(*a, **kw):
        out = original(*a, **kw)
        if "pred" not in store:
            t = out[0] if isinstance(out, (tuple, list)) else out.sample
            store["pred"] = t.detach().float().cpu().numpy()
            cc = kw.get("control_context")
            if cc is not None:
                store["control_context_shape"] = list(cc.shape)
                store["hidden_states_shape"] = list(kw["hidden_states"].shape)
                store["img_shapes"] = [list(map(int, s)) for s in kw.get("img_shapes", [[]])[0]]
                store["img_mask_slot_count"] = int(kw["img_mask"].shape[1])
        return out

    pipe.transformer.forward = spy
    return store, lambda: setattr(pipe.transformer, "forward", original)


def finite_report(name, arr):
    return {f"{name}_shape": list(arr.shape), f"{name}_nan_count": int(np.isnan(arr).sum()),
            f"{name}_inf_count": int(np.isinf(arr).sum())}


def compare(a, b):
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "comparable": False}
    d = np.abs(a - b)
    return {"shape": list(a.shape), "comparable": True,
            "max_abs": float(np.nanmax(d)) if d.size else None,
            "mean_abs": float(np.nanmean(d)) if d.size else None,
            "bitwise_equal": bool(np.array_equal(a, b)),
            "within_bf16_atol": bool(np.nanmax(d) <= BF16_ATOL) if d.size else False,
            **finite_report("a", a), **finite_report("b", b)}


def adapter_pred(pipe, *, prompt, ctrl, refs, scale, h, w, steps, seed):
    g = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    store, undo = capture_first_pred(pipe)
    try:
        pipe(prompt=prompt, reference_images=refs, control_image=ctrl, control_context_scale=scale,
             height=h, width=w, num_inference_steps=steps, true_cfg_scale=1.0, use_kv_cache=False,
             generator=g, output_type="latent", return_dict=False)
    finally:
        undo()
    return store


def official_pred(pipe, *, prompt, ctrl, scale, h, w, steps, seed):
    g = torch.Generator(device=pipe._execution_device).manual_seed(seed)
    store, undo = capture_first_pred(pipe)
    try:
        pipe(prompt=prompt, height=h, width=w, num_inference_steps=steps, true_cfg_scale=1.0,
             control_image=ctrl, control_context_scale=scale, use_kv_cache=False,
             generator=g, output_type="latent", return_dict=False)
    finally:
        undo()
    return store


def load_assets(h, w):
    from PIL import Image
    from videox_fun.utils import get_image_latent
    UNION = WORK / "models/Qwen-Image-2.1-Fun-Controlnet-Union"
    ctrl = get_image_latent(str(UNION / "asset/control_4_00000931.png"), sample_size=[h, w])[:, :, 0]
    refs = [Image.open(UNION / "asset/inpaint_source.png").convert("RGB"),
            Image.open(UNION / "asset/control_5_00005389.png").convert("RGB")]
    return ctrl, refs


PROMPT = "一位年轻女子站在海边，亮紫色长发被海风吹起，身穿纯白色无袖连衣裙。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["official", "adapter"], default="adapter")
    ap.add_argument("--canvas", default="512,512")
    ap.add_argument("--steps", type=int, default=2, help=">=2 步，避开 1 步 schedule 的 NaN 边界")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    h, w = [int(x) for x in args.canvas.split(",")]
    name = args.name or f"contract_gate_{args.phase}_{time.strftime('%Y%m%dT%H%M%S')}"
    out_dir = WORK / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    PRED_OFFICIAL.parent.mkdir(parents=True, exist_ok=True)

    if args.phase == "official":
        pipe, _ = build_pipeline(kind="official", memory_mode="model_group_offload")
        ctrl, _ = load_assets(h, w)
        store = official_pred(pipe, prompt=PROMPT, ctrl=ctrl, scale=1.0, h=h, w=w,
                              steps=args.steps, seed=args.seed)
        np.save(PRED_OFFICIAL, store["pred"])
        print(json.dumps({"phase": "official", "saved": str(PRED_OFFICIAL),
                          **finite_report("pred", store["pred"]),
                          "control_context_shape": store.get("control_context_shape"),
                          "hidden_states_shape": store.get("hidden_states_shape")}, ensure_ascii=False))
        return 0

    pipe, audit = build_pipeline(kind="reference", memory_mode="model_group_offload")
    ctrl, refs = load_assets(h, w)
    results = {"videox_commit": verify_videox_checkout(), "weight_gate": audit.get("gate_pass"),
               "canvas": [h, w], "steps": args.steps, "seed": args.seed, "bf16_atol": BF16_ATOL,
               "compared_quantity": "single-step transformer prediction (noise_pred)"}

    # A. 不传参考图 vs 原版官方（跨进程单步预测对照）
    sA = adapter_pred(pipe, prompt=PROMPT, ctrl=ctrl, refs=None, scale=1.0,
                      h=h, w=w, steps=args.steps, seed=args.seed)
    np.save(WORK / "cache/gate_adapter_noref_pred.npy", sA["pred"])
    if PRED_OFFICIAL.exists():
        results["A_no_reference_matches_official"] = compare(np.load(PRED_OFFICIAL), sA["pred"])
    else:
        results["A_no_reference_matches_official"] = {"skipped": "官方单步预测未落盘，先跑 --phase official"}

    # B. 控制强度 0 == 完全不传控制图（同一实例、同进程）
    sB0 = adapter_pred(pipe, prompt=PROMPT, ctrl=ctrl, refs=None, scale=0.0,
                       h=h, w=w, steps=args.steps, seed=args.seed)
    sB1 = adapter_pred(pipe, prompt=PROMPT, ctrl=None, refs=None, scale=0.0,
                       h=h, w=w, steps=args.steps, seed=args.seed)
    results["B_control_scale_zero_matches_no_control"] = compare(sB0["pred"], sB1["pred"])

    # C. 参考图接入后的序列对齐
    sC = adapter_pred(pipe, prompt=PROMPT, ctrl=ctrl, refs=refs, scale=1.0,
                      h=h, w=w, steps=args.steps, seed=args.seed)
    tgt = (h // 16) * (w // 16)
    info = {"control_context_shape": sC.get("control_context_shape"),
            "hidden_states_shape": sC.get("hidden_states_shape"),
            "img_shapes": sC.get("img_shapes"),
            "img_mask_slot_count": sC.get("img_mask_slot_count")}
    n_total = (info["hidden_states_shape"] or [0, 0])[1]
    lead_rows = max(0, n_total - tgt)
    results["C_sequence_alignment"] = {
        **info, "target_tokens_expected": tgt, "ref_tokens_derived": n_total - tgt,
        "control_len_equals_ref_plus_target": (info["control_context_shape"] or [0, 0])[1] == n_total,
        "control_dim_is_129": (info["control_context_shape"] or [0, 0, 0])[2] == 129,
        "img_shapes_covers_two_refs_and_target": len(info["img_shapes"] or []) == 3,
        "pred_is_finite": bool(np.isfinite(sC["pred"]).all()),
    }

    # 输出裁剪：像素尺寸等于画布
    img = pipe(prompt=PROMPT, reference_images=refs, control_image=ctrl, control_context_scale=1.0,
               height=h, width=w, num_inference_steps=args.steps, true_cfg_scale=1.0, use_kv_cache=False,
               generator=torch.Generator(device=pipe._execution_device).manual_seed(42)).images[0]
    results["C_output_canvas"] = {"returned_wh": list(img.size), "expected_wh": [w, h],
                                  "ok": list(img.size) == [w, h]}
    img.save(out_dir / "sample_two_refs.png")

    # D. 换参考顺序：token 数不变，结果改变
    sD = adapter_pred(pipe, prompt=PROMPT, ctrl=ctrl, refs=[refs[1], refs[0]], scale=1.0,
                      h=h, w=w, steps=args.steps, seed=args.seed)
    cmpD = compare(sC["pred"], sD["pred"])
    results["D_reference_order_matters"] = {
        "same_token_count": sD.get("hidden_states_shape") == info["hidden_states_shape"],
        "max_abs_latent_diff_vs_unswapped": cmpD.get("max_abs"),
        "outputs_differ": bool((cmpD.get("max_abs") or 0.0) > 1e-3),
        **{k: v for k, v in cmpD.items() if k.endswith(("nan_count", "inf_count"))}}

    a, b = results["A_no_reference_matches_official"], results["B_control_scale_zero_matches_no_control"]
    results["gate_pass"] = bool(
        a.get("comparable") and a.get("within_bf16_atol")
        and b.get("comparable") and b.get("within_bf16_atol")
        and results["C_sequence_alignment"]["control_len_equals_ref_plus_target"]
        and results["C_sequence_alignment"]["control_dim_is_129"]
        and results["C_sequence_alignment"]["img_shapes_covers_two_refs_and_target"]
        and results["C_sequence_alignment"]["pred_is_finite"]
        and results["C_output_canvas"]["ok"]
        and results["D_reference_order_matters"]["same_token_count"]
        and results["D_reference_order_matters"]["outputs_differ"])

    (out_dir / "contract_gate.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=1, ensure_ascii=False))
    return 0 if results["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
