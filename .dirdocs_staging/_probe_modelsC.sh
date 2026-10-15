set -e
R=/data/shangguanrz
M=$R/pic-edit-models
echo "===== models L1 目录说明.md ====="
cat $M/目录说明.md
echo "===== caches L1 目录说明.md ====="
cat $R/pic-edit-caches/目录说明.md
echo "===== Qwen-Image-2.1 top ls ====="
ls -la $M/Qwen-Image-2.1
echo "===== model_index.json ====="
cat $M/Qwen-Image-2.1/model_index.json
echo "===== transformer ls ====="
ls -la $M/Qwen-Image-2.1/transformer
echo "===== transformer config ====="
cat $M/Qwen-Image-2.1/transformer/config.json
echo "===== transformer index ====="
cat $M/Qwen-Image-2.1/transformer/model.safetensors.index.json 2>/dev/null | head -c 3000
echo
echo "===== text_encoder ls ====="
ls -la $M/Qwen-Image-2.1/text_encoder
echo "===== text_encoder config ====="
cat $M/Qwen-Image-2.1/text_encoder/config.json
echo "===== text_encoder index (head) ====="
head -c 2000 $M/Qwen-Image-2.1/text_encoder/model.safetensors.index.json 2>/dev/null
echo
echo "===== vae ls ====="
ls -la $M/Qwen-Image-2.1/vae
echo "===== vae config ====="
cat $M/Qwen-Image-2.1/vae/config.json
echo "===== processor ls ====="
ls -la $M/Qwen-Image-2.1/processor
echo "===== processor processor_config + chat_template ====="
cat $M/Qwen-Image-2.1/processor/processor_config.json 2>/dev/null
echo "===== scheduler ls ====="
ls -la $M/Qwen-Image-2.1/scheduler
echo "===== scheduler config ====="
cat $M/Qwen-Image-2.1/scheduler/scheduler_config.json
echo "===== assets ls ====="
ls -la $M/Qwen-Image-2.1/assets
echo "===== PE-I2I ls ====="
ls -la $M/Qwen-Image-2.1-PE-I2I
echo "===== PE-I2I json files ====="
for f in $M/Qwen-Image-2.1-PE-I2I/*.json; do echo "--- $f"; head -c 4000 "$f"; echo; done
echo "===== PE-I2I safetensors keys header via python ====="
python3 - <<'PY'
import json, os, glob
d='/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I'
for p in sorted(glob.glob(d+'/*.index.json')):
    j=json.load(open(p))
    print(p, 'meta keys:', list(j.keys()))
    print('  total_size=', j.get('metadata'))
    wm=j.get('weight_map',{})
    print('  tensors=', len(wm), 'shards=', sorted(set(wm.values()))[:12])
PY
echo "===== caches ls count ====="
ls $R/pic-edit-caches/qwen21_hi4d_controls | wc -l
ls $R/pic-edit-caches/qwen21_hi4d_controls | head -20
ls $R/pic-edit-caches/qwen21_hi4d_controls | tail -10
echo "===== caches du ====="
du -sh --si $R/pic-edit-caches/qwen21_hi4d_controls
du -sh $R/pic-edit-caches/qwen21_hi4d_controls
df -h $R | tail -1
