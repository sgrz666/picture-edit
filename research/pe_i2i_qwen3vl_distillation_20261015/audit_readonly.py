"""Read checkpoint headers and existing workflow records without running inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import statistics
import struct
from zoneinfo import ZoneInfo


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(values):
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "min": min(values), "max": max(values)} if values else {"count": 0}


def checkpoint_summary(directory):
    index = read_json(directory / "model.safetensors.index.json")
    tensors = {}
    for filename in sorted(set(index["weight_map"].values())):
        with (directory / filename).open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        tensors.update({key: value for key, value in header.items() if key != "__metadata__"})
    targets = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
               "gate_proj.weight", "up_proj.weight", "down_proj.weight")
    lora_parameters = sum(16 * sum(tensor["shape"]) for name, tensor in tensors.items()
                          if ".language_model.layers." in name and name.endswith(targets))
    return {"path": str(directory), "config": read_json(directory / "config.json"),
            "config_sha256": sha256(directory / "config.json"),
            "index_sha256": sha256(directory / "model.safetensors.index.json"),
            "parameter_count": sum(math.prod(tensor["shape"]) for tensor in tensors.values()),
            "tensor_count": len(tensors), "lm_head": tensors.get("lm_head.weight"),
            "language_attention_mlp_lora_rank16_parameters": lora_parameters,
            "weight_bytes": index.get("metadata", {}).get("total_size")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.project / "workflow/config.json")
    run_root = args.project / "workflow/runs"
    stages = {}
    records = {}
    for stage in ("geometry", "prompt", "generation", "evaluation"):
        rows = []
        for path in sorted(run_root.rglob(f"{stage}/stage.json")):
            record = read_json(path)
            seconds = record.get("metadata", {}).get("seconds")
            if isinstance(seconds, (int, float)):
                rows.append({"path": str(path), "seconds": seconds})
                records.setdefault(str(path.parent.parent), {})[stage] = seconds
        stages[stage] = {"seconds": summary([row["seconds"] for row in rows]), "records": rows}
    complete = [{"run": name, **times} for name, times in records.items()
                if all(stage in times for stage in ("geometry", "prompt", "generation"))]
    prompt_rows = []
    for path in sorted(run_root.rglob("prompt/pe_result.json")):
        result = read_json(path)
        signatures = tuple(item.get("sha256") for item in result.get("input_roles", []))
        prompt_rows.append({"path": str(path), "seconds": result.get("seconds"),
                            "output_tokens": result.get("output_tokens"),
                            "attempts": result.get("attempts", []), "input_signature": signatures,
                            "peak_gpu_memory_gib": result.get("peak_gpu_memory_gib")})
    from transformers import AutoTokenizer
    teacher_tokenizer = AutoTokenizer.from_pretrained(config["pe_model"], local_files_only=True)
    student_tokenizer = AutoTokenizer.from_pretrained(
        Path(config["qwen_model"]) / "processor", local_files_only=True)
    sample = run_root / "wp1_accept_hard02/prompt"
    sample_result = read_json(sample / "pe_result.json")
    raw = (sample / "pe_raw.txt").read_text(encoding="utf-8")
    answer = raw.split("</think>")[-1].strip()
    rewritten = sample_result["official_result"]["rewritten_prompt"]
    compiled = sample_result["compiled_prompt"]
    packages = {}
    for package in ("torch", "transformers", "diffusers", "peft", "trl", "ms-swift", "unsloth"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    code_files = ["workflow/run.py", "workflow/pe_worker.py", "workflow/generate_worker.py",
                  "workflow/pose_transfer/prompts.py", "src/pose_edit/prompt_contract.py"]
    report = {
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Existing records and CPU-only tokenization; no inference or training",
        "project": str(args.project), "workflow_config": config, "packages": packages,
        "code_sha256": {name: sha256(args.project / name) for name in code_files},
        "teacher": checkpoint_summary(Path(config["pe_model"])),
        "student": checkpoint_summary(Path(config["qwen_model"]) / "text_encoder"),
        "stages": stages, "complete_runs": complete,
        "pe_fraction_within_complete_runs": summary([
            row["prompt"] / sum(row[stage] for stage in ("geometry", "prompt", "generation"))
            for row in complete]),
        "prompt_records": prompt_rows,
        "unique_prompt_image_triples": len(Counter(tuple(row["input_signature"]) for row in prompt_rows)),
        "sample_token_counts": {
            "run": "wp1_accept_hard02", "recorded_generated_tokens": sample_result["output_tokens"],
            "decoded_raw_teacher_tokens": len(teacher_tokenizer.encode(raw, add_special_tokens=False)),
            "final_answer_teacher_tokens": len(teacher_tokenizer.encode(answer, add_special_tokens=False)),
            "rewritten_teacher_tokens": len(teacher_tokenizer.encode(rewritten, add_special_tokens=False)),
            "compiled_student_tokens": len(student_tokenizer.encode(compiled, add_special_tokens=False)),
            "compiled_words": len(compiled.split()), "seconds_including_loading": sample_result["seconds"],
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
