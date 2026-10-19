#!/usr/bin/env python
"""指标报告：把 metrics.jsonl / 分组 CSV / 三关结果汇总成 report/METRICS.md。

报告只讲磁盘上量得到的数字：每个样本每个配置的严格 PCK、误差、检测覆盖、身份分、耗时与峰值显存，
并按固定规则给出「直接使用 / 需要输入适配 / 需要另立训练实验」三选一的初步判定。
判定规则写在报告里，谁都能复核；规则之外的人工判断（遮挡场景的面板检查）单独列出，不混进数字。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

EXP = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path("/data/shangguanrz/pic-edit-experiments/qwen21_union_migration_v1")
CONFIGS = ["native", "union_s0", "union_s05", "union_s10"]
IMPROVE_EPS = 0.02      # 严格 PCK 相对 native 高出这么多才算「改善」
IDENTITY_DROP = 0.05    # 身份分掉这么多才算「身份退化」


def latest(prefix: str):
    cands = sorted(p for p in (WORK / "runs").glob(f"{prefix}*") if (p / "contract_gate.json").exists()
                   or (p / "summary.json").exists())
    return cands[-1] if cands else None


def fnum(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    return f"{float(v):.{nd}f}"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="WORK/runs 下的矩阵运行名")
    args = ap.parse_args()

    run_dir = WORK / "runs" / args.run
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    gate_dir = latest("contract_gate_adapter")
    gate = json.loads((gate_dir / "contract_gate.json").read_text(encoding="utf-8")) if gate_dir else {}
    off_dir = latest("official_formal")
    off = json.loads((off_dir / "summary.json").read_text(encoding="utf-8")) if off_dir else {}
    audit = json.loads((run_dir / "weight_audit.json").read_text(encoding="utf-8")) if (run_dir / "weight_audit.json").exists() else {}
    manifest = json.loads((EXP / "input_manifest.json").read_text(encoding="utf-8"))
    prov = json.loads((EXP / "input_provenance.json").read_text(encoding="utf-8")) if (EXP / "input_provenance.json").exists() else {}

    by = {}
    for r in rows:
        by.setdefault((r["sample"], r["config"]), []).append(r)
    samples = [s["sample"] for s in manifest["samples"]]
    kinds = {s["sample"]: s["kind"] for s in manifest["samples"]}

    # 逐样本逐配置的均值（两 seed）
    metric_rows = []
    deltas = {}
    for sample in samples:
        base = by.get((sample, "native"), [])
        def mean(rs, field, ap=None):
            vals = []
            for r in rs:
                v = r.get(field)
                if v is None and ap:
                    v = (r.get("appearance") or {}).get(field)
                if v is not None:
                    vals.append(v)
            return float(np.mean(vals)) if vals else None
        for cfg in CONFIGS:
            rs = by.get((sample, cfg), [])
            if not rs:
                continue
            pck = mean(rs, "pck@0.2_strict")
            ident = mean(rs, "identity_mean", ap=True)
            swap = [ (r.get("appearance") or {}).get("identity_swap_suspected") for r in rs]
            swap = [s for s in swap if s is not None]
            metric_rows.append([
                sample, kinds[sample], cfg, len(rs), fnum(pck), fnum(mean(rs, "pck@0.5_strict")),
                fnum(mean(rs, "mpjpe_norm")), fnum(mean(rs, "face_mpjpe_norm")), fnum(mean(rs, "hand_mpjpe_norm")),
                fnum(mean(rs, "detector_coverage")), fnum(mean(rs, "persons_detected"), 2),
                fnum(mean(rs, "person_count_ok") if kinds[sample] == "dual" else None),
                fnum(ident), (f"{sum(1 for s in swap if s)}/{len(swap)}" if swap else "—"),
                fnum(mean(rs, "seconds"), 1), fnum(mean(rs, "peak_GiB"), 2)])
            if pck is not None:
                deltas.setdefault(sample, {})[cfg] = pck

    # 规则判定
    verdict_lines, improved, regressed, identity_hits = [], 0, 0, 0
    for sample, d in deltas.items():
        best_cfg = max((c for c in d if c != "native"), key=lambda c: d[c], default=None)
        if best_cfg is None or "native" not in d or d["native"] is None:
            continue
        delta = d[best_cfg] - d["native"]
        tag = "改善" if delta > IMPROVE_EPS else ("退化" if delta < -IMPROVE_EPS else "持平")
        if tag == "改善":
            improved += 1
        elif tag == "退化":
            regressed += 1
        idd = {}
        for cfg in CONFIGS:
            rs = by.get((sample, cfg), [])
            vals = [(r.get("appearance") or {}).get("identity_mean") for r in rs]
            vals = [v for v in vals if v is not None]
            if vals:
                idd[cfg] = float(np.mean(vals))
        drop = (idd.get("native", 0) - idd.get(best_cfg, 0)) if idd.get("native") and idd.get(best_cfg) else None
        if drop is not None and drop > IDENTITY_DROP:
            identity_hits += 1
        verdict_lines.append(
            f"- {sample}（{kinds[sample]}）：native 严格 PCK@0.2 {fnum(d['native'])} → 最佳 Union 档 "
            f"{best_cfg} {fnum(d[best_cfg])}（{tag}，Δ={delta:+.3f}）；身份分 "
            f"{fnum(idd.get('native'))} → {fnum(idd.get(best_cfg))}"
            + (f"（掉 {drop:.3f}，超过阈值 {IDENTITY_DROP}）" if drop and drop > IDENTITY_DROP else ""))

    total = len(verdict_lines)
    if total and improved >= max(2, int(np.ceil(0.75 * total))) and identity_hits == 0:
        verdict = "可继续直接使用官方权重"
        why = f"{improved}/{total} 个样本在最佳 Union 档上严格 PCK 改善超过 {IMPROVE_EPS}，且没有样本身份分下降超过 {IDENTITY_DROP}"
    elif improved >= 1:
        verdict = "需要进一步输入适配"
        why = (f"只有 {improved}/{total} 个样本改善（{regressed} 个退化、身份退化 {identity_hits} 例），"
               "控制通道有效但覆盖面不足，退化点需要先定位在输入侧还是权重侧")
    else:
        verdict = "需要另立训练实验"
        why = f"没有任何样本在 Union 档上改善严格 PCK（改善 {improved}、退化 {regressed}）"

    missing = [(s, c) for s in samples for c in CONFIGS if (s, c) not in by]
    errors = [r for r in rows if not r.get("pose") and r.get("error")]
    undetected = [r for r in rows if r.get("detector_ok") is False]

    md = []
    md.append(f"# 指标报告：{args.run}\n")
    md.append(f"- 运行目录：`{run_dir}`")
    md.append(f"- 参数：{summary.get('steps','?')} 步、seed {summary.get('seeds')}、CFG {summary.get('cfg')}、"
              f"{summary.get('dtype')}、KV cache {summary.get('kv_cache')}、单卡串行 + 官方 group offload")
    md.append(f"- 应产出 {summary.get('expected_images','?')} 张，实际评估 {len(rows)} 张，"
              f"complete={summary.get('complete')}；缺 {len(missing)} 个样本×配置组合"
              + (f"：{missing}" if missing else ""))
    md.append(f"- 生成失败 {len(errors)} 次；DWPose 完全未检到人 {len(undetected)} 张（保留在结果里，不当成生成成功）")
    if summary.get("gpu_waits"):
        md.append(f"- 等显存 {len(summary['gpu_waits'])} 次（与人共用 GPU）")
    md.append("")

    md.append("## 三关状态\n")
    md.append(table([
        ["1 官方复现", off.get("gate_pass"), f"权重 {audit.get('control_keys_expected','180')} 键全部加载，"
         f"零初始化残留 {audit.get('still_zero_after_load','?')} 个" if audit else "见 official_formal 运行",
         str(off_dir)],
        ["2 接入契约", gate.get("gate_pass"), "无参考图与原版逐位相同；scale 0 与不加控制逐位相同；"
         "控制张量长度 = 参考+目标、通道 129；换参考图顺序输出会变" if gate else "见 contract_gate 运行",
         str(gate_dir)],
        [f"3 效果比较（{len(rows)}/48）", not missing and len(rows) > 0,
         f"改善 {improved} / 退化 {regressed} / 身份退化 {identity_hits}，共 {total} 个样本", str(run_dir)],
    ], ["关口", "是否通过", "依据", "产物目录"]) + "\n")

    md.append("## 逐样本逐配置指标\n")
    md.append("严格口径（strict）把漏检或未对齐的人记 0 分；`persons_detected` 与 `人数对齐` 用来说明"
              "低分到底是姿态错还是检测器没分开人。身份分是 DINOv2 余弦相似度，双人另有互换计数。\n")
    md.append(table(metric_rows, ["样本", "类型", "配置", "n", "PCK@0.2 strict", "PCK@0.5 strict",
                                  "MPJPE(归一)", "脸 MPJPE", "手 MPJPE", "检测覆盖", "检出人数",
                                  "人数对齐", "身份分", "疑似互换", "耗时(s)", "峰值(GiB)"]) + "\n")

    md.append("## 每个样本相对 native 的变化\n")
    md.extend(verdict_lines)
    md.append("")

    md.append("## 双人身份与对应关系细节\n")
    dual_lines = []
    for r in rows:
        if r["kind"] != "dual":
            continue
        a = r.get("appearance") or {}
        asg = r.get("assignment") or {}
        dual_lines.append(
            f"- {r['sample']} {r['config']} seed{r['seed']}：检出 {r['persons_detected']} 人，"
            f"髋部配对 {'歧义' if asg.get('ambiguous') else '明确'}（成本 {asg.get('cost_px','?')} px，"
            f"与次优差 {asg.get('cost_margin_to_next','?')} px），身份分 {fnum(a.get('identity_mean'))}，"
            f"疑似互换 {a.get('identity_swap_suspected')}（阈值 {a.get('swap_margin_threshold','?')}），"
            f"pair kind={prov.get(r['sample'], {}).get('pair_kind','?')}")
    md.extend(dual_lines or ["（本轮暂无双人结果）"])
    md.append("")

    md.append("## 判定\n")
    md.append(f"**{verdict}**。\n\n规则依据：{why}。\n")
    md.append("规则写在这里以便复核：把「最佳 Union 档相对 native 的严格 PCK@0.2」提高超过 "
              f"{IMPROVE_EPS} 记一个改善样本；身份分下降超过 {IDENTITY_DROP} 记一个身份退化样本；"
              f"改善样本数 ≥ 75% 且无身份退化 → 直接使用；有改善但未全覆盖 → 需要输入适配；全不改善 → 需要另立训练实验。")
    md.append("\n小样本结论只决定下一步方向，不自动替换现有工作流；遮挡场景请结合 `panels/` 人工面板看，"
              "不要把检测器漏检直接算成生成失败。")

    out = EXP / "report" / "METRICS.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "verdict": verdict, "improved": improved,
                      "regressed": regressed, "identity_regressions": identity_hits,
                      "samples_judged": total, "missing": len(missing)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
