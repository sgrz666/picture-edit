#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash train.sh DATASET_DIR BASE_SHA256_FILE NEW_RUN_DIR" >&2
    exit 2
fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+$ ]]; then
    echo "Select one GPU with CUDA_VISIBLE_DEVICES." >&2
    exit 2
fi

python_bin=${QWEN_LORA_PYTHON:-/data/shangguanrz/LoRA/envs/flow-factory-v050/bin/python}
base=${QWEN_MODEL:-/data/shangguanrz/pic-edit-models/Qwen-Image-2.1}
trainer=${QWEN_LORA_TRAINER:-/data/shangguanrz/LoRA/source/diffusers/train_dreambooth_lora_qwenimage21_img2img.py}
trainer_sha=3c92bfcea1050003625ab178d2934425ddb667e0b9d105b74b6e97692fd040ca
scripts_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
dataset=$(realpath -e "$1")
base_checksums=$(realpath -e "$2")
run=$(realpath -m "$3")
parquet=train-00000-of-00001.parquet

if [[ $(find "$dataset" -mindepth 1 -maxdepth 1 -printf '%f\n') != "$parquet" ]]; then
    echo "The dataset directory must contain only the exported train Parquet." >&2
    exit 2
fi
printf '%s  %s\n' "$trainer_sha" "$trainer" | sha256sum --check -
dataset_recipe=$("$python_bin" -c '
import json
from pathlib import Path
import sys
record = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if record["split"] != "train" or record["parquet"] != sys.argv[2] or record["pairs"] < 1:
    raise SystemExit("Expected a completed train dataset exported by prepare_dataset.py")
columns = record.get("condition_columns", ["source"])
if columns not in (["source"], ["source", "pose"]):
    raise SystemExit(f"Unsupported ordered condition columns: {columns}")
print(record["parquet_sha256"], ",".join(columns))
' "$dataset.json" "$parquet")
read -r dataset_sha columns <<< "$dataset_recipe"
printf '%s  %s\n' "$dataset_sha" "$dataset/$parquet" | sha256sum --check -
expected_files=$(cd "$base" && find -L model_index.json processor scheduler text_encoder transformer vae -type f | LC_ALL=C sort)
if [[ $(cut -c67- "$base_checksums") != "$expected_files" ]]; then
    echo "Base checksum file must cover all pipeline components in sorted order." >&2
    exit 2
fi
base_sha=$(sha256sum "$base_checksums" | cut -d' ' -f1)

mkdir "$run"
cp "$base_checksums" "$run/base.sha256"
cp "$dataset.json" "$run/dataset.json"
upstream="$run/train_dreambooth_lora_qwenimage21_img2img.py"
cp "$trainer" "$upstream"
printf '%s  %s\n' "$trainer_sha" "$upstream" | sha256sum --check -
effective_trainer=$upstream
condition_args=(--cond_image_column source)
if [[ $columns == source,pose ]]; then
    cp "$scripts_dir/qwen21_multiimg.py" "$scripts_dir/qwen21_multiimg.patch" "$run/"
    effective_trainer="$run/train_dreambooth_lora_qwenimage21_multiimg.py"
    patch --batch --forward --fuzz=0 --output="$effective_trainer" "$upstream" "$run/qwen21_multiimg.patch"
    "$python_bin" -m py_compile "$effective_trainer" "$run/qwen21_multiimg.py"
    condition_args=(--cond_image_columns source,pose --condition_cache_dir "$run/condition-cache")
fi
{
    printf 'base=%s\nbase_manifest_sha256=%s\ntrainer=%s\ntrainer_sha256=%s\ndataset=%s\ndataset_sha256=%s\n' \
        "$base" "$base_sha" "$trainer" "$trainer_sha" "$dataset" "$dataset_sha"
    printf 'gpu=%s\npython=%s\n' "$CUDA_VISIBLE_DEVICES" "$python_bin"
    printf 'condition_columns=%s\neffective_trainer_sha256=%s\n' "$columns" "$(sha256sum "$effective_trainer" | cut -d' ' -f1)"
    if [[ $columns == source,pose ]]; then
        printf 'patch_sha256=%s\nhelper_sha256=%s\n' \
            "$(sha256sum "$run/qwen21_multiimg.patch" | cut -d' ' -f1)" \
            "$(sha256sum "$run/qwen21_multiimg.py" | cut -d' ' -f1)"
    fi
    "$python_bin" -m pip show diffusers torch transformers accelerate peft datasets | sed -n '/^Name:/p; /^Version:/p'
} > "$run/provenance.txt"
(cd "$base" && sha256sum --strict --check "$run/base.sha256") > "$run/base-verify.log"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$run/hf-cache" PYTHONUNBUFFERED=1
args=(
    "$effective_trainer"
    --pretrained_model_name_or_path "$base"
    --dataset_name "$dataset"
    "${condition_args[@]}" --image_column target --caption_column caption
    --instance_prompt "Edit the hand gesture."
    --output_dir "$run/weights"
    --mixed_precision bf16 --resolution 512 --center_crop
    --rank 16 --lora_alpha 16 --lora_layers to_q,to_k,to_v,to_out.0
    --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0
    --train_batch_size 1 --gradient_accumulation_steps "${GRAD_ACCUM:-8}" --gradient_checkpointing
    --max_train_steps "${MAX_STEPS:-1}" --checkpointing_steps "${CHECKPOINT_STEPS:-${MAX_STEPS:-1}}"
    --cache_latents --seed 42 --report_to tensorboard --skip_final_inference
)
launch=("$python_bin" -m accelerate.commands.launch --num_processes 1 --num_machines 1
    --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1)
printf '%q ' "${launch[@]}" "${args[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
"${launch[@]}" "${args[@]}" 2>&1 | tee "$run/train.log"
