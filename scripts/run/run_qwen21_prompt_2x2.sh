#!/usr/bin/env bash
set -Ee

# 先激活已有 Qwen 编辑环境，再启用 nounset
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /home/shangguanrz/nihaochen/envs/qwen21
set -Eeuo pipefail

PY="/home/shangguanrz/nihaochen/envs/qwen21/bin/python"
EDIT_MODEL="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1"
PROJECT="$HOME/project/pic-edit"
REPO="$PROJECT/vendor/Qwen-Image-2.1"
PE_ENV="$HOME/.venvs/qwen-pe-i2i"

IDS=(single_0002 single_0007 single_0026 single_0042 single_0050)

CANDIDATES=(
  "$HOME/data/baidu_pose_collection/run_20261025_172757/images/10/single"
  "$HOME/data/baidu_pose_collection/qwen_review_9_selected_20261025_184010/images"
  "$HOME/data/baidu_pose_collection/qwen_review_9_selected_20261025_184010/images/single"
  "$HOME/data/baidu_pose_collection/qwen_prompt_exp5_cn_20261026_210448/inputs"
)

find_image() {
  local dir="$1" id="$2"
  find "$dir" -maxdepth 1 -type f \
    \( -iname "${id}.jpg" -o -iname "${id}.jpeg" \
       -o -iname "${id}.png" -o -iname "${id}.webp" \) \
    -print -quit
}

SRC_DIR=""
for d in "${CANDIDATES[@]}"; do
  [[ -d "$d" ]] || continue
  ok=1
  for id in "${IDS[@]}"; do
    f="$(find_image "$d" "$id")"
    [[ -n "$f" ]] || { ok=0; break; }
  done
  if (( ok )); then
    SRC_DIR="$d"
    break
  fi
done

if [[ -z "$SRC_DIR" ]]; then
  echo "[停止] 没有找到全部五张原图。"
  printf '已检查目录：\n%s\n' "${CANDIDATES[@]}"
  exit 2
fi

[[ -x "$PY" ]] || { echo "[停止] Qwen Python 环境不存在"; exit 2; }
[[ -f "$EDIT_MODEL/model_index.json" ]] || {
  echo "[停止] Qwen-Image-2.1 模型目录不正确：$EDIT_MODEL"
  exit 2
}

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$HOME/data/baidu_pose_collection/qwen_prompt_2x2_$STAMP"
mkdir -p "$OUT"/{inputs,metadata,A_short_raw,B_short_pe,C_detailed_raw,D_detailed_pe}

exec > >(tee -a "$OUT/run.log") 2>&1

echo "========== 实验配置 =========="
echo "原图目录：$SRC_DIR"
echo "编辑模型：$EDIT_MODEL"
echo "输出目录：$OUT"

for id in "${IDS[@]}"; do
  f="$(find_image "$SRC_DIR" "$id")"
  ext="${f##*.}"
  cp "$f" "$OUT/inputs/${id}.${ext,,}"
done

# 官方 PE-I2I 源码：只克隆到独立目录，不修改现有项目。
mkdir -p "$PROJECT/vendor"
if [[ ! -f "$REPO/prompt_rewrite/run_transformers.py" ]]; then
  if [[ -e "$REPO" ]]; then
    echo "[停止] 已存在不完整的官方源码目录：$REPO"
    exit 2
  fi
  git clone --depth 1 https://github.com/QwenLM/Qwen-Image-2.1.git "$REPO"
fi

# 查找已有 PE-I2I 权重，避免不必要的重复下载。
PE_CKPT=""
LOCAL_CKPTS=(
  "/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I"
  "$HOME/models/Qwen-Image-2.1-PE-I2I"
)

for p in "${LOCAL_CKPTS[@]}"; do
  if [[ -f "$p/config.json" && -f "$p/system_prompt.txt" ]]; then
    PE_CKPT="$p"
    break
  fi
done

for cache in \
  "$HOME/.cache/huggingface/hub" \
  "/data/shangguanrz/.cache/huggingface/hub" \
  "/data/shangguanrz/hf-cache/hub"; do
  [[ -n "$PE_CKPT" ]] && break
  for p in "$cache"/models--Qwen--Qwen-Image-2.1-PE-I2I/snapshots/*; do
    if [[ -f "$p/config.json" && -f "$p/system_prompt.txt" ]]; then
      PE_CKPT="$p"
      break
    fi
  done
done

if [[ -z "$PE_CKPT" ]]; then
  # 官方模型约 20GB。使用数据盘缓存，避免挤占系统盘。
  export HF_HOME="/data/shangguanrz/hf-cache"
  mkdir -p "$HF_HOME"
  FREE_GB="$(df -BG "$HF_HOME" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')"
  if [[ -z "$FREE_GB" ]] || (( FREE_GB < 25 )); then
    echo "[停止] 数据盘可用空间不足 25GB，不自动下载 PE-I2I 权重。"
    df -h "$HF_HOME"
    exit 2
  fi
  PE_CKPT="Qwen/Qwen-Image-2.1-PE-I2I"
  echo "[提示] 本地没有发现 PE-I2I 权重；将尝试从 Hugging Face 下载。"
else
  echo "[PE-I2I] 使用已有权重：$PE_CKPT"
fi

# 为 PE-I2I 建立独立 Python 环境，避免升级原 Qwen 编辑环境的依赖。
if [[ ! -x "$PE_ENV/bin/python" ]]; then
  "$PY" -m venv --system-site-packages "$PE_ENV"
fi
PE_PY="$PE_ENV/bin/python"

"$PE_PY" -m pip install \
  "transformers==5.4.0" \
  "tokenizers==0.22.1" \
  "huggingface_hub==1.27.0" \
  "accelerate==1.11.0" \
  "tqdm==4.67.1"

# 写入实验脚本
cat > "$OUT/experiment.py" <<'PY'
import os
import sys
import json
import gc
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(os.environ["POSE_OUT"])
INPUT = ROOT / "inputs"
IDS = [
    "single_0002", "single_0007", "single_0026",
    "single_0042", "single_0050"
]

# 两组共享同一条保持约束。
KEEP = "只改变人物姿势，尽量保持人物外观、服装、背景和原始构图不变。"

# 简洁版：只给出主要动作。
SHORT_ACTIONS = {
    "single_0002": "让人物自然地交叉步行，前腿迈出，身体保持平衡。",
    "single_0007": "让人物自然地单腿抬起，抬起腿膝盖弯曲，另一腿稳定支撑。",
    "single_0026": "让人物双臂向身体两侧平举，保持自然站姿。",
    "single_0042": "让人物做宽距半蹲，双膝弯曲，双臂向两侧展开。",
    "single_0050": "让人物一只手臂向斜上方举起，另一只手臂自然下垂。",
}

# 详细版：更明确地描述关节关系和动作完成方式。
DETAILED_ACTIONS = {
    "single_0002": (
        "将人物调整为自然的交叉步行姿态：一条腿向前迈出并从另一条腿前方通过，"
        "双脚落点自然，膝关节轻微弯曲，躯干直立，重心稳定。"
    ),
    "single_0007": (
        "将人物调整为自然的单腿屈膝抬起姿态：一条腿向前抬起，膝关节明显弯曲，"
        "小腿自然向下，抬起的脚离地；另一条腿作为支撑腿，脚掌平稳落地，"
        "膝盖微屈，骨盆稳定，躯干直立，重心落在支撑脚上。双手保持原来的位置。"
    ),
    "single_0026": (
        "将人物调整为自然的双臂侧平举站姿：双臂分别向身体两侧抬至接近肩部高度，"
        "肘部自然微屈，手掌放松，肩部不耸起，躯干直立，双腿自然站立。"
    ),
    "single_0042": (
        "将人物调整为自然的宽距半蹲姿态：双脚向左右分开，膝关节自然弯曲且方向"
        "与脚尖一致，臀部稍向后，躯干轻微前倾以保持平衡，双臂向两侧展开，"
        "双手和双脚完整留在画面内。"
    ),
    "single_0050": (
        "将人物调整为自然的单臂斜上举姿态：一只手臂向斜上方抬起，肘部自然伸展，"
        "手掌放松；另一只手臂沿身体侧面自然下垂，双腿自然站立，躯干和骨盆稳定。"
    ),
}

SHORT = {k: v + KEEP for k, v in SHORT_ACTIONS.items()}
DETAILED = {k: v + KEEP for k, v in DETAILED_ACTIONS.items()}

def save_json(path, data):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def find_input(stem):
    matches = list(INPUT.glob(stem + ".*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{stem}: 找到 {len(matches)} 张输入图片")
    return matches[0]

def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")

def prepare():
    variants = {
        stem: {"short": SHORT[stem], "detailed": DETAILED[stem]}
        for stem in IDS
    }
    save_json(ROOT / "metadata" / "prompt_variants.json", variants)

    # 官方 PE-I2I 输入：10 条指令、5 张原图各用两次。
    cases = []
    for stem in IDS:
        img = str(find_input(stem).resolve())
        cases.append({
            "id": f"short_{stem}",
            "prompt": SHORT[stem],
            "input_images": [img],
            "task_type": "basic_edit",
        })
        cases.append({
            "id": f"detailed_{stem}",
            "prompt": DETAILED[stem],
            "input_images": [img],
            "task_type": "basic_edit",
        })

    with (ROOT / "metadata" / "pe_input.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    print(f"[完成] 已准备 {len(cases)} 条 PE-I2I 输入。")

def make_sheet():
    groups = [
        ("SOURCE", None),
        ("A_SHORT_RAW", "A_short_raw"),
        ("B_SHORT_PE", "B_short_pe"),
        ("C_DETAIL_RAW", "C_detailed_raw"),
        ("D_DETAIL_PE", "D_detailed_pe"),
    ]

    cw, ch, label_h, gap = 300, 420, 28, 10
    sheet = Image.new(
        "RGB",
        (gap + len(groups) * (cw + gap),
         gap + len(IDS) * (ch + label_h + gap)),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for row, stem in enumerate(IDS):
        y = gap + row * (ch + label_h + gap)
        for col, (label, folder) in enumerate(groups):
            x = gap + col * (cw + gap)
            draw.text((x + 2, y), f"{stem} | {label}", fill="black")

            path = find_input(stem) if folder is None else (
                ROOT / folder / f"{stem}_{folder}.png"
            )

            if path.is_file():
                with Image.open(path) as im:
                    im = ImageOps.contain(
                        im.convert("RGB"), (cw - 8, ch - 8)
                    )
                px = x + (cw - im.width) // 2
                py = y + label_h + (ch - im.height) // 2
                sheet.paste(im, (px, py))
            else:
                draw.text((x + 10, y + label_h + 10), "FAILED / MISSING", fill="red")

    dest = ROOT / "comparison_4groups.jpg"
    sheet.save(dest, quality=92)
    print(f"[拼图] {dest}")

def generate():
    # 官方 runner 的输出字段可能使用 positive_prompt 或 rewritten_prompt。
    pe_path = ROOT / "metadata" / "pe_output.jsonl"
    if not pe_path.is_file():
        raise FileNotFoundError(f"PE-I2I 输出不存在：{pe_path}")

    rewrites = {}
    for obj in read_jsonl(pe_path):
        if obj.get("parse_ok") is False:
            raise RuntimeError(f"PE-I2I 解析失败：{obj.get('id')}")
        prompt = obj.get("positive_prompt") or obj.get("rewritten_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise RuntimeError(f"没有有效改写提示词：{obj.get('id')}")
        rewrites[obj["id"]] = prompt.strip()

    if len(rewrites) != 10:
        raise RuntimeError(f"预期 10 条改写结果，实际 {len(rewrites)} 条")

    save_json(ROOT / "metadata" / "rewritten_prompts.json", rewrites)

    from diffusers import QwenImage21Pipeline

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA")

    print("[模型] 加载 Qwen-Image-2.1 ...", flush=True)
    pipe = QwenImage21Pipeline.from_pretrained(
        os.environ["POSE_EDIT_MODEL"],
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    groups = [
        ("A_short_raw", lambda stem: SHORT[stem]),
        ("B_short_pe", lambda stem: rewrites[f"short_{stem}"]),
        ("C_detailed_raw", lambda stem: DETAILED[stem]),
        ("D_detailed_pe", lambda stem: rewrites[f"detailed_{stem}"]),
    ]

    results = []
    for i, stem in enumerate(IDS):
        source_path = find_input(stem)
        source = load_rgb(source_path)
        seed = 20261026 + i

        for group, get_prompt in groups:
            prompt = get_prompt(stem)
            out_path = ROOT / group / f"{stem}_{group}.png"
            print(f"\n[生成] {group} | {stem} | seed={seed}", flush=True)
            print(f"[提示词] {prompt}", flush=True)

            try:
                generator = torch.Generator(device="cuda").manual_seed(seed)
                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        image=source.copy(),
                        num_inference_steps=40,
                        generator=generator,
                    ).images[0]

                result.save(out_path)
                results.append({
                    "id": stem,
                    "group": group,
                    "seed": seed,
                    "steps": 40,
                    "source": str(source_path),
                    "output": str(out_path),
                    "prompt": prompt,
                    "status": "success",
                })
                print(f"[成功] {out_path}", flush=True)

            except Exception as exc:
                results.append({
                    "id": stem,
                    "group": group,
                    "seed": seed,
                    "source": str(source_path),
                    "output": str(out_path),
                    "prompt": prompt,
                    "status": "failed",
                    "error": repr(exc),
                })
                print(f"[失败] {group} {stem}: {exc}", flush=True)
                traceback.print_exc()

            save_json(ROOT / "metadata" / "generation_results.json", results)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    make_sheet()
    success = sum(r["status"] == "success" for r in results)
    print(f"\n[完成] 成功生成 {success}/20 张。")
    print(f"[结果目录] {ROOT}")
    print(f"[对比拼图] {ROOT / 'comparison_4groups.jpg'}")
    print(f"[提示词记录] {ROOT / 'metadata'}")

    if success != 20:
        raise SystemExit(2)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("prepare", "generate"):
        raise SystemExit("用法：experiment.py prepare|generate")
    if sys.argv[1] == "prepare":
        prepare()
    else:
        generate()
PY

export POSE_OUT="$OUT"
export POSE_EDIT_MODEL="$EDIT_MODEL"
export HF_HOME="${HF_HOME:-/data/shangguanrz/hf-cache}"
mkdir -p "$OUT/metadata"

echo
echo "========== 1. 准备四组实验 =========="
"$PY" "$OUT/experiment.py" prepare

echo
echo "========== 2. 官方 PE-I2I 改写 =========="
"$PE_PY" "$REPO/prompt_rewrite/run_transformers.py" \
  --task edit \
  --ckpt "$PE_CKPT" \
  --input "$OUT/metadata/pe_input.jsonl" \
  --output "$OUT/metadata/pe_output.jsonl" \
  --dtype bfloat16 \
  --device cuda \
  --max-new-tokens 8192 \
  --seed 42

echo
echo "========== 3. Qwen-Image-2.1 生成 20 张图 =========="
"$PY" "$OUT/experiment.py" generate

echo
echo "========== 全部完成 =========="
echo "实验目录：$OUT"
echo "对照拼图：$OUT/comparison_4groups.jpg"
echo "改写提示词：$OUT/metadata/rewritten_prompts.json"
echo "完整日志：$OUT/run.log"
