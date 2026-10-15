set -u
cd /home/shangguanrz/project/pic-edit/workflow || exit 1
echo '=== workflow 顶层 ==='
ls -la
echo '=== runs 列表（最近 8 个）==='
ls -1 runs 2>/dev/null | tail -8
R=$(ls -1d runs/*/ 2>/dev/null | tail -1)
echo "=== 抽样 run: $R ==="
find "$R" -maxdepth 2 | head -40
echo '=== stage.json 字段 ==='
for f in "$R"stage.json "$R"*/stage.json; do [ -f "$f" ] && { echo "--- $f"; head -c 900 "$f"; echo; }; done 2>/dev/null | head -60
echo '=== run.py 阶段编排（关键片段）==='
grep -n 'stage\|prepare\|geometry\|prompt\|generation\|def main' run.py 2>/dev/null | head -30
echo '=== 现有可复用评测资产（仓级）==='
cd /home/shangguanrz/project/pic-edit || exit 1
ls -1 src/evaluation/ 2>/dev/null
grep -rl 'pck\|PCK\|arcface\|ArcFace\|identity_cosine\|lpips\|LPIPS\|ssim' scripts/ src/ 2>/dev/null | head -20
echo '=== dwpose/antelopev2/sapiens 权重是否就位 ==='
ls -1 models/DWPose models/antelopev2 add/models/sapiens2-seg-0.4b add/models/sapiens-seg-0.3b-torchscript 2>/dev/null
