"""Append narrow documentation and preserve the exact runtime/canvas source dependency set."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

MARKER = "## Native Depth / CHAMP (2026-10-23)"
NOTES = {
    "src/pose_control/qwen21_full/README.md": "独立Depth输入契约、CHAMP CNN/空间attention、512维残差与新checkpoint已接入Native。见 [NATIVE_DEPTH.md](NATIVE_DEPTH.md)。旧Full默认入口保留。",
    "scripts/train/README.md": "`train_qwen21_native_depth.py` 使用实际Native Geometry/Body与可选Depth；显式off/relative_png/camera_z、pair/canvas CPU预检、原生cache与flow训练。用法见 `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。",
    "scripts/audit/README.md": "`audit_qwen21_native_depth.py` 严格迁入官方active path并比较空间attention；`verify_qwen21_native_depth_runtime.py` 使用真实7B和矩形条件核对prefix、冻结基座与Depth/Body梯度，不更新优化器、不采样图片。",
    "tests/README.md": "`test_qwen21_depth_condition.py` / `test_qwen21_native_depth.py` / `test_qwen21_native_depth_runtime.py` 覆盖Depth方向/valid/canvas、CHAMP权重、512残差、pair绑定、资产漂移、checkpoint重建与原生32层反向。",
    "third_party/champ_guidance/README.md": "`spatial_attention.py` 在相同固定commit下移植官方有效空间Transformer，保留GN/proj、self-attention、LayerNorm/GEGLU与state key，改用PyTorch SDPA并支持可选valid-key mask。历史卷积mixer入口保持独立；新Native Depth可显式加载官方active权重，见 `src/pose_control/qwen21_full/NATIVE_DEPTH.md`。",
    "THIRD_PARTY_NOTICES.md": "`third_party/champ_guidance/spatial_attention.py` adapts `models/transformer_3d.py` and the active spatial path of `TemporalBasicTransformerBlock` in `models/attention.py` from CHAMP commit `4d0cad2ca23990a26a0c2f69d4ecb1b55f5df140` (MIT, license retained). It preserves the active parameter names and spatial math, uses PyTorch SDPA and an optional valid-key mask, and omits temporal/cross attention and unused modules. Native Depth can explicitly import the official active CNN/attention weights; no weight is bundled in Git.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    args = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source = args.run/"source"
    backup = source/f"docs_before_{stamp}"
    backup.mkdir(parents=True,exist_ok=False)
    changed = []
    for name,note in NOTES.items():
        path = args.project/name
        before = path.read_bytes()
        if MARKER.encode() in before:
            continue
        target = backup/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(before)
        text = before.decode("utf8")
        if name == "third_party/champ_guidance/README.md":
            text = text.replace("本项目不加载 CHAMP 预训练权重。", "历史guidance_encoder.py不加载CHAMP权重；新增空间attention供Native Depth显式初始化。")
        after = text.rstrip()+"\n\n"+MARKER+"\n\n"+note+"\n"
        if path.read_bytes()!=before:
            raise RuntimeError(f"Documentation changed concurrently: {path}")
        path.write_bytes(after.encode("utf8"))
        changed.append({"path":name,"before_sha256":hashlib.sha256(before).hexdigest(),
                        "after_sha256":hashlib.sha256(after.encode()).hexdigest()})
    tracked = subprocess.check_output(["git","diff","--name-only"],cwd=args.project,text=True).splitlines()
    others = subprocess.check_output(["git","ls-files","--others","--exclude-standard","src","scripts","tests","third_party"],
                                     cwd=args.project,text=True).splitlines()
    pinned = ["src/pose_control/qwen21_full/native_geometry.py","src/pose_control/qwen21_full/native_body.py",
              "src/pose_control/qwen21_full/native_integration.py","src/pose_control/qwen21_full/qwen_runtime.py",
              "src/pose_control/qwen21_full/data_runtime.py","third_party/champ_guidance/LICENSE"]
    files = sorted({n for n in [*tracked,*others,*pinned]
                    if (args.project/n).is_file() and (Path(n).suffix in (".py",".md",".json") or n.endswith("/LICENSE"))})
    hashes = {n:hashlib.sha256((args.project/n).read_bytes()).hexdigest() for n in files}
    archive = source/f"runtime_and_canvas_{stamp}.tar.gz"
    with tarfile.open(archive,"x:gz") as stream:
        for name in files:
            if not (args.project/name).resolve().is_relative_to(args.project.resolve()):
                raise ValueError("Source snapshot must stay inside project")
            stream.add(args.project/name,arcname=name,recursive=False)
    record = {"timestamp_utc":stamp,"head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=args.project,text=True).strip(),
              "source_archive":str(archive),"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
              "source_sha256":hashes,"documentation_changes":changed,
              "scope":"Text source only, including current uncommitted canvas prerequisites; no media/model/cache moved or included"}
    with (source/f"delivery_{stamp}.json").open("x",encoding="utf8") as stream:
        json.dump(record,stream,indent=2)
        stream.write("\n")
    print(json.dumps({"files":len(files),"archive":str(archive),"docs_appended":len(changed)}))


if __name__=="__main__":
    main()
