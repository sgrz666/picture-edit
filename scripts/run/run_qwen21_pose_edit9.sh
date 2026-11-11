#!/usr/bin/env bash
set -Eeuo pipefail

# ===== 1. 使用 node01 已有环境 =====
# Conda 激活钩子需要允许未定义变量
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /home/shangguanrz/nihaochen/envs/qwen21
set -u

PY="/home/shangguanrz/nihaochen/envs/qwen21/bin/python"
EDIT="/home/shangguanrz/project/pic-edit/code/qwen21/edit.py"
MODEL="/data/shangguanrz/pic-edit-models/Qwen-Image-2.1"

ROOT="$HOME/data/baidu_pose_collection/qwen_review_9_selected_20261025_184010"
SRC="$ROOT/images"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$HOME/data/baidu_pose_collection/qwen_pose_edit9_$STAMP"

[[ -x "$PY" ]] || { echo "[错误] Python 不存在：$PY"; exit 1; }
[[ -f "$EDIT" ]] || { echo "[错误] 编辑脚本不存在：$EDIT"; exit 1; }
[[ -d "$MODEL" ]] || { echo "[错误] 模型目录不存在：$MODEL"; exit 1; }
[[ -d "$SRC/single" && -d "$SRC/double" ]] || {
    echo "[错误] 输入目录不符合预期：$SRC"
    exit 1
}

mkdir -p "$OUT/single" "$OUT/double"

LOG="$OUT/run.log"
FAIL="$OUT/failed.txt"
MANIFEST="$OUT/prompt_manifest.tsv"
printf 'source\toutput\tprompt\n' > "$MANIFEST"
: > "$FAIL"

# ===== 2. 检查本次的 9 张输入图片 =====
FILES=(
  "single/single_0002.jpg"
  "single/single_0007.jpg"
  "single/single_0026.jpg"
  "single/single_0042.jpg"
  "single/single_0050.jpg"
  "double/double_0001.jpg"
  "double/double_0003.jpg"
  "double/double_0012.jpg"
  "double/double_0021.jpg"
)

for f in "${FILES[@]}"; do
    [[ -f "$SRC/$f" ]] || {
        echo "[错误] 缺少输入图片：$SRC/$f"
        exit 1
    }
done

echo "[OK] 输入图片：9 张（单人 5 张，双人 4 张）"
echo "[OK] 输出目录：$OUT"
echo "[OK] Python：$PY"
echo "[OK] 模型：$MODEL"

# ===== 3. 逐张编辑 =====
i=0
for rel in "${FILES[@]}"; do
    i=$((i+1))
    stem="${rel%.*}"
    dst="$OUT/${stem}_target.png"
    seed=$((20261025+i))

    case "$rel" in

    single/single_0002.jpg)
        PROMPT='只改变原图中女性的身体姿态。保持同一人物的面部、发型、体型、粉色上衣、牛仔短裙、鞋子、影棚背景、光照和拍摄视角不变。将原来的单腿抬起姿势改成明显的侧向弓步：一条腿向侧面迈出并屈膝，另一条腿伸直，躯干向侧面轻微倾斜，双臂向不同方向自然舒展。动作幅度明显，关节自然，双脚完整可见，保持真实摄影质感。'

        ;;

    single/single_0007.jpg)
        PROMPT='只改变原图中男性的身体姿态。保持同一人物的面部、发型、黑色长外套、内搭、裤子、鞋子、街道环境和摄影视角不变。将静态站姿改成明显的向前快走动作：前腿跨出并屈膝，后腿向后蹬伸，双臂自然前后摆动，躯干略微转向行进方向。体现真实的运动过程，双腿与手臂轮廓清晰，人体比例自然，头部和双脚完整入镜。'

        ;;

    single/single_0026.jpg)
        PROMPT='只改变原图中女性的身体姿态。保持同一人物的面部、发型、服装、鞋子、蓝色拱形门、墙壁、地面和光照不变。将原来的站立姿势改为动态的侧向跨步：一条腿向侧面迈出并屈膝，另一条腿伸直，躯干向另一侧倾斜，一只手斜向上伸展，另一只手向外侧打开。动作舒展自然，手臂与身体清楚分离，人体比例正确，头部到双脚完整可见，保持真实摄影风格。'

        ;;

    single/single_0042.jpg)
        PROMPT='只改变原图中女性的身体姿态。保持同一人物的面部、发型、太阳镜、上衣、裤子、鞋子、随身包、草地和树木背景不变。将双手扶腰的静态站姿改为大幅度侧弓步：一条腿屈膝向侧面迈出，另一条腿伸直，躯干向侧面倾斜，一只手向斜上方伸展，另一只手向相反方向打开。动作明显且自然，保留原始人物比例和摄影质感，双脚完整可见。'

        ;;

    single/single_0050.jpg)
        PROMPT='只改变原图中女性的身体姿态。保持同一人物的面部、发型、灰色上衣、白色长裙、鞋子、湖水、天空、石墙、树木和光照不变。将原来的站姿改为动态的侧向弓步：一条腿向侧前方迈出并屈膝，另一条腿伸直，躯干轻微侧倾，一只手向斜上方伸展，另一只手向侧面打开以保持平衡，长裙随动作自然摆动。人体比例正确，动作自然，头部和双脚完整可见，保持真实摄影质感。'

        ;;

    double/double_0001.jpg)
        PROMPT='只改变原图中两个人的身体姿态。保留原图男女双方各自的面部、发型、身材、浅灰色正式服装、鞋子、手捧花、影棚背景、光照和原始构图。将静态合影改为自然的双人行走互动：男士向前跨出一步，身体转向女士并自然伸出一只手；女士侧身迈步回应，另一只手自然向侧面伸展。两人的姿态不同但互动协调，身体之间保持合理距离，手臂和腿部互不融合，双方头部到双脚完整入镜，保持真实摄影风格。'

        ;;

    double/double_0003.jpg)
        PROMPT='只改变原图中男女双方的身体姿态。保留两人的面部、发型、服装、原始人物相对位置、椅子、窗帘、室内背景和光照。女生从原来的静态坐姿改为身体向侧面转动、一只手向斜上方伸展、另一只手自然打开；男生在她身后将原来的姿势改为身体轻微转向另一侧、一只手向外侧伸展、另一只手自然弯曲。两人形成协调但不同的动态姿态，四肢结构清晰，不穿插、不融合，保持原始摄影风格和构图。'

        ;;

    double/double_0012.jpg)
        PROMPT='只改变原图中两位女性的身体姿态。保持两人的面部、发型、服装、鞋子、海滩、海面、天空和自然光不变。将原来两人双手相接的姿势改成相反方向的动态侧弓步：左侧人物向左跨步并屈膝，躯干向左倾斜，双臂向外侧舒展；右侧人物向右跨步并屈膝，躯干向右倾斜，双臂向外侧打开。两人不再互相牵手，动作幅度明显，四肢自然分离，保持人物比例、原始机位和完整人体构图。'

        ;;

    double/double_0021.jpg)
        PROMPT='只改变原图中两个人的身体姿态。保持两人各自的面部、发型、衣服、鞋子、原有外观、树木和户外环境不变。将原来的静态合影改成方向不同的动态动作：左侧人物向左侧跨出大步并屈膝，一只手向斜上方伸展；右侧人物向右侧跨步并屈膝，躯干向外侧倾斜，另一只手向侧上方打开。两人的动作明显不同，形成协调的双人动态构图，四肢轮廓清楚，不互相融合，保持真实摄影质感。'

        ;;

    esac

    printf '\n========== [%s/9] %s ==========\n' "$i" "$rel"
    printf '%s\t%s\t%s\n' "$rel" "$dst" "$PROMPT" >> "$MANIFEST"

    # 已有有效输出则跳过，便于失败后继续运行
    if [[ -s "$dst" ]] && "$PY" -c \
        'from PIL import Image; import sys; Image.open(sys.argv[1]).verify()' \
        "$dst" >/dev/null 2>&1; then
        echo "[跳过] 已存在有效输出：$dst"
        continue
    fi

    if "$PY" "$EDIT" \
        --image "$SRC/$rel" \
        --prompt "$PROMPT" \
        --out "$dst" \
        --steps 40 \
        --seed "$seed" 2>&1 | tee -a "$LOG"; then

        if [[ -s "$dst" ]] && "$PY" -c \
            'from PIL import Image; import sys; Image.open(sys.argv[1]).verify()' \
            "$dst" >/dev/null 2>&1; then
            echo "[完成] $dst"
        else
            echo "[失败] 输出文件无效：$rel" | tee -a "$LOG"
            echo "$rel" >> "$FAIL"
        fi
    else
        echo "[失败] $rel" | tee -a "$LOG"
        echo "$rel" >> "$FAIL"
    fi
done

# ===== 4. 生成目标拼图和原图/目标图对比拼图 =====
"$PY" - "$SRC" "$OUT" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

src_root = Path(sys.argv[1])
out_root = Path(sys.argv[2])

targets = sorted(
    p for p in out_root.glob("*/*_target.png")
    if p.is_file() and p.stat().st_size > 0
)

valid = []
for p in targets:
    try:
        with Image.open(p) as im:
            im.verify()
        original_name = p.stem.removesuffix("_target") + ".jpg"
        original = src_root / p.parent.name / original_name
        if original.is_file():
            valid.append((original, p))
    except Exception as e:
        print(f"[拼图跳过] {p.name}: {e}")

if not valid:
    print("[提醒] 没有有效目标图，暂不生成拼图")
    raise SystemExit(0)

# A. 仅包含生成结果的总览
tw, th, cols = 300, 370, 3
rows = (len(valid) + cols - 1) // cols
sheet = Image.new("RGB", (tw * cols, th * rows), "white")
draw = ImageDraw.Draw(sheet)

for i, (_, target) in enumerate(valid):
    x, y = (i % cols) * tw, (i // cols) * th
    with Image.open(target) as im:
        thumb = ImageOps.contain(im.convert("RGB"), (tw - 12, th - 42))
    sheet.paste(thumb, (x + (tw-thumb.width)//2, y + 4))
    draw.text((x+5, y+th-30), target.name, fill="black")

sheet.save(out_root / "contact_sheet_targets.jpg", quality=92)

# B. 原图与生成图并排对比
cw, ch, cols = 660, 410, 2
rows = (len(valid) + cols - 1) // cols
comparison = Image.new("RGB", (cw*cols, ch*rows), "white")
draw = ImageDraw.Draw(comparison)

for i, (original, target) in enumerate(valid):
    x, y = (i % cols)*cw, (i // cols)*ch
    draw.text((x+8, y+5), original.parent.name + "/" + original.name,
              fill="black")
    draw.text((x+cw//2+5, y+5), "GENERATED: " + target.name,
              fill="black")

    for path, bx in [(original, x), (target, x+cw//2)]:
        with Image.open(path) as im:
            thumb = ImageOps.contain(im.convert("RGB"), (cw//2-16, ch-38))
        comparison.paste(
            thumb,
            (bx+(cw//2-thumb.width)//2, y+30)
        )

comparison.save(out_root / "comparison_contact_sheet.jpg", quality=92)

print(f"\n[拼图完成] 成功配对 {len(valid)}/9 张")
print("[生成结果总览]", out_root / "contact_sheet_targets.jpg")
print("[原图与生成图对比]", out_root / "comparison_contact_sheet.jpg")
PY

echo
echo "========== 全部处理结束 =========="
echo "输出目录：$OUT"
echo "日志：$LOG"
echo "提示词清单：$MANIFEST"
echo "失败清单：$FAIL"
echo "目标拼图：$OUT/contact_sheet_targets.jpg"
echo "前后对比：$OUT/comparison_contact_sheet.jpg"

if [[ -s "$FAIL" ]]; then
    echo "[提醒] 存在失败图片，请查看：$FAIL"
else
    echo "[OK] 没有记录到编辑失败"
fi
