#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: CUDA_VISIBLE_DEVICES=0 bash train.sh DATASET_DIR BASE_SHA256_FILE NEW_RUN_DIR" >&2
    exit 2
fi
if [[ ! ${CUDA_VISIBLE_DEVICES:-} =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Set CUDA_VISIBLE_DEVICES to the ordered GPU list." >&2
    exit 2
fi
IFS=, read -ra devices <<< "$CUDA_VISIBLE_DEVICES"
world=${#devices[@]}
deterministic=${DETERMINISTIC:-0}
if [[ $deterministic != 0 && $deterministic != 1 ]]; then
    echo "DETERMINISTIC must be 0 or 1." >&2
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
printf '%s  %s\n' "$trainer_sha" "$trainer" | sha256sum --check -
dataset_recipe=$(PYTHONPATH="$scripts_dir" "$python_bin" -c '
from dataset_contract import read_export
import sys
record, _ = read_export(sys.argv[1], "train")
columns = record["condition_columns"]
if columns not in (["source"], ["source", "pose"]):
    raise SystemExit(f"Unsupported ordered condition columns: {columns}")
print(record["dataset_sha256"], ",".join(columns), record["pairs"])
' "$dataset")
read -r dataset_sha columns pair_count <<< "$dataset_recipe"
expected_files=$(cd "$base" && find -L model_index.json processor scheduler text_encoder transformer vae -type f | LC_ALL=C sort)
if [[ $(cut -c67- "$base_checksums") != "$expected_files" ]]; then
    echo "Base checksum file must cover all pipeline components in sorted order." >&2
    exit 2
fi
base_sha=$(sha256sum "$base_checksums" | cut -d' ' -f1)

cache_index_sha=none
init_sha=none
resume=none
if [[ $columns == source,pose ]]; then
    cache=$(realpath -e "${CONDITION_CACHE:?set CONDITION_CACHE to the completed prepare_cache.py output}")
    cache_index_sha=$(PYTHONPATH="$scripts_dir" "$python_bin" -c '
import json
from pathlib import Path
import sys
from dataset_contract import file_sha256, fingerprint

root, base, data, count = sys.argv[1:]
root = Path(root)
index = json.loads((root / "index.json").read_text(encoding="utf-8"))
spec = index["spec"]
if (index["identity"] != fingerprint(spec) or spec["schema"] != 1
        or spec["base_manifest_sha256"] != base or spec["dataset_sha256"] != data
        or spec["seed"] != 42 or len(index["rows"]) != int(count)):
    raise ValueError("Prepared cache does not match the requested training inputs")
for number, row in enumerate(index["rows"]):
    name = f"{number:06d}.safetensors"
    if row["file"] != name or file_sha256(root / name) != row["sha256"]:
        raise ValueError(f"Cache shard changed: {name}")
print(file_sha256(root / "index.json"))
' "$cache" "$base_sha" "$dataset_sha" "$pair_count")
    if [[ -n ${INIT_LORA:-} ]]; then
        init_sha=${INIT_LORA_SHA256:?provide the locked initialization SHA256}
        printf '%s  %s\n' "$init_sha" "$INIT_LORA" | sha256sum --strict --check -
    fi
    if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
        resume=$(realpath -e "$RESUME_CHECKPOINT")
        test -s "$resume/resume.json"
    fi
elif (( world != 1 )) || [[ -n ${RESUME_CHECKPOINT:-}${INIT_LORA:-} ]]; then
    echo "The L0 launcher uses a single GPU and a fresh adapter." >&2
    exit 2
fi

mkdir "$run"
cp "$base_checksums" "$run/base.sha256"
cp "$dataset.json" "$run/dataset.json"
upstream="$run/train_dreambooth_lora_qwenimage21_img2img.py"
cp "$trainer" "$upstream"
printf '%s  %s\n' "$trainer_sha" "$upstream" | sha256sum --check -
effective_trainer=$upstream
condition_args=(--cond_image_column source)
if [[ $columns == source,pose ]]; then
    cp "$scripts_dir"/{dataset_contract.py,prepare_cache.py,qwen21_multiimg.py,qwen21_multiimg.patch} "$run/"
    effective_trainer="$run/train_dreambooth_lora_qwenimage21_multiimg.py"
    patch --batch --forward --fuzz=0 --output="$effective_trainer" "$upstream" "$run/qwen21_multiimg.patch"
    "$python_bin" -m py_compile "$effective_trainer" "$run/qwen21_multiimg.py"
    condition_args=(--cond_image_columns source,pose --condition_cache_dir "$cache")
    if [[ $init_sha != none ]]; then
        cp "$INIT_LORA" "$run/initial.safetensors"
        printf '%s  %s\n' "$init_sha" "$run/initial.safetensors" | sha256sum --strict --check -
        condition_args+=(--init_lora "$run/initial.safetensors")
    fi
    if [[ $resume != none ]]; then
        condition_args+=(--resume_from_checkpoint "$resume")
    fi
fi
{
    printf 'base=%s\nbase_manifest_sha256=%s\ntrainer=%s\ntrainer_sha256=%s\ndataset=%s\ndataset_sha256=%s\n' \
        "$base" "$base_sha" "$trainer" "$trainer_sha" "$dataset" "$dataset_sha"
    printf 'gpu=%s\npython=%s\n' "$CUDA_VISIBLE_DEVICES" "$python_bin"
    printf 'condition_columns=%s\neffective_trainer_sha256=%s\n' "$columns" "$(sha256sum "$effective_trainer" | cut -d' ' -f1)"
    if [[ $columns == source,pose ]]; then
        printf 'cache=%s\ncache_index_sha256=%s\ninit_lora_sha256=%s\nresume_checkpoint=%s\ndeterministic=%s\n' \
            "$cache" "$cache_index_sha" "$init_sha" "$resume" "$deterministic"
        printf 'patch_sha256=%s\nhelper_sha256=%s\n' \
            "$(sha256sum "$run/qwen21_multiimg.patch" | cut -d' ' -f1)" \
            "$(sha256sum "$run/qwen21_multiimg.py" | cut -d' ' -f1)"
    fi
    "$python_bin" -m pip show diffusers torch transformers accelerate peft datasets | sed -n '/^Name:/p; /^Version:/p'
} > "$run/provenance.txt"
(cd "$base" && sha256sum --strict --check "$run/base.sha256") > "$run/base-verify.log"

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$run/hf-cache" PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42 HAND_LORA_DETERMINISTIC="$deterministic"
if [[ $deterministic == 1 ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
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
launch=("$python_bin" -m accelerate.commands.launch --num_processes "$world" --num_machines 1
    --mixed_precision bf16 --dynamo_backend no --num_cpu_threads_per_process 1)
if (( world > 1 )); then
    launch+=(--multi_gpu --main_process_port "${MASTER_PORT:-29571}")
fi
printf '%q ' "${launch[@]}" "${args[@]}" > "$run/command.txt"
printf '\n' >> "$run/command.txt"
"${launch[@]}" "${args[@]}" 2>&1 | tee "$run/train.log"
