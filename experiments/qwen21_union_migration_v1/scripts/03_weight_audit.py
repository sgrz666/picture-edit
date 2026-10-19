#!/usr/bin/env python
"""加载检查：Union 的 180 个控制权重键、shape 全量核验，并反证"加载后没有被再次置零"。

这一步只看权重，不看生成质量。gate_pass 为假就不要继续跑样例。
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from union_env import WORK, load_union_transformer  # noqa: E402


def main():
    import torch
    transformer, audit = load_union_transformer(device="cpu")

    named = dict(transformer.named_parameters())
    rows = []
    for name in sorted(n for n in named
                       if n.startswith("control_blocks.") or n.startswith("control_img_in.")):
        t = named[name]
        rows.append({
            "key": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "abs_sum": float(t.detach().float().abs().sum()),
            "all_zero": bool(float(t.detach().float().abs().sum()) == 0.0),
        })
    audit["per_key"] = rows
    audit["control_keys_listed"] = len(rows)
    audit["keys_all_zero"] = sum(1 for r in rows if r["all_zero"])

    out = WORK / "report/weight_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")

    summary = {k: v for k, v in audit.items() if k != "per_key"}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    zero_proj = [r["key"] for r in rows if r["all_zero"]]
    print("\n控制键数:", len(rows), " 其中全零:", len(zero_proj))
    if zero_proj:
        print("全零键样例:", zero_proj[:10])
    print("GATE:", "PASS" if audit["gate_pass"] else "FAIL")
    return 0 if audit["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
