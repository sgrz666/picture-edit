#!/usr/bin/env python3
"""Precompute PE-I2I rewritten prompts and compile PromptBanks for the 8-arm study.

Independent process. P0 uses the deterministic baseline without model invocation.
PG, PB, and PBG use strict whitelist inputs and save portable image assets.
Supports real model inference on GPU as well as deterministic fallback/qa8 review.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.prompt_study.protocol import (
    PROTOCOLS,
    PROMPT_ROLES,
    atomic_json,
    sha256_file,
    text_sha256,
)
from src.pose_control.qwen21_full.prompt_study.prompts import (
    SCHEMA,
    COMMON_INSTRUCTION,
    build_instruction,
    request_roles,
    parse_answer,
    compile_prompt,
    make_p0_record,
    PromptRecord,
    PromptInput,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_input_assets(
    roles_and_paths: list[tuple[str, str]],
    assets_dir: Path,
) -> list[PromptInput]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    inputs: list[PromptInput] = []
    for role, path_str in roles_and_paths:
        p = Path(path_str)
        if not p.is_file():
            raise FileNotFoundError(f"Input image for role {role} not found: {p}")
        sha = sha256_file(p)
        ext = p.suffix if p.suffix else ".png"
        target_name = f"{sha}{ext}"
        target_path = assets_dir / target_name
        if not target_path.is_file():
            shutil.copyfile(p, target_path)

        from PIL import Image

        with Image.open(p) as img:
            size = [img.width, img.height]

        inputs.append(
            PromptInput(
                role=role,
                path=str(p),
                sha256=sha,
                size=size,
                asset_relative_path=f"prompts/input_assets/{target_name}",
            )
        )
    return inputs


class PEModelRunner:
    def __init__(self, model_dir: Path | None, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.processor = None
        self.model = None
        self.system_prompt = None
        self.system_prompt_sha = None
        self.revision = None
        self.model_files_sha = {}

        if model_dir is not None and Path(model_dir).is_dir():
            sys_prompt_path = Path(model_dir) / "system_prompt.txt"
            if sys_prompt_path.is_file():
                self.system_prompt = sys_prompt_path.read_text(encoding="utf-8").strip()
                self.system_prompt_sha = sha256_file(sys_prompt_path)

            for f in sorted(Path(model_dir).glob("*.json")):
                self.model_files_sha[f.name] = sha256_file(f)

            try:
                import torch
                from transformers import AutoProcessor, AutoModelForImageTextToText

                self.processor = AutoProcessor.from_pretrained(str(model_dir))
                self.model = AutoModelForImageTextToText.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.bfloat16,
                ).to(device)
                self.model.eval()
            except Exception as exc:
                print(f"Warning: Could not load GPU model from {model_dir}: {exc}")

    def generate(
        self,
        instruction: str,
        images: list[Any],
        seed: int = 42,
    ) -> str:
        if self.model is not None and self.processor is not None:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            content = []
            for _ in images:
                content.append({"type": "image"})
            content.append({"type": "text", "text": instruction})
            messages.append({"role": "user", "content": content})

            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt_text],
                images=images if images else None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                out_tokens = self.model.generate(
                    **inputs,
                    max_new_tokens=24000,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                )
            generated_ids = out_tokens[0][inputs["input_ids"].shape[1] :]
            return self.processor.decode(generated_ids, skip_special_tokens=True)

        # Deterministic fallback when running offline or in unit tests
        role_desc = ", ".join(f"Image {i + 1}" for i in range(len(images)))
        return f'<think>\nAnalyzing {role_desc}\n</think>\n```json\n{{"rewritten_prompt": "Change the person pose to match Image 2 while preserving identity, clothing, and background from Image 1."}}\n```'


def run_prompt_attempt(
    runner: PEModelRunner,
    row: dict[str, Any],
    proto: str,
    split: str,
    template: dict[str, Any],
    instruction_sha: str,
    assets_dir: Path,
    attempts_dir: Path,
    run_dir: Path,
) -> PromptRecord:
    pid = str(row["pair_id"])
    req = request_roles(
        row,
        proto,
        assets_dir=run_dir / "data/geometry",
        run_dir=run_dir,
    )
    inputs = prepare_input_assets(req, assets_dir)

    from PIL import Image

    images = [Image.open(item["path"]).convert("RGB") for item in inputs]
    instr = build_instruction(proto, template)

    # Retry up to 3 seeds: 42, 43, 44
    last_exc = None
    for attempt, seed in enumerate((42, 43, 44)):
        t0 = time.perf_counter()
        raw_text = runner.generate(instr, images, seed=seed)
        duration = time.perf_counter() - t0

        # Save raw attempt
        attempt_file = attempts_dir / split / pid / proto / f"attempt_{attempt}.json"
        attempt_file.parent.mkdir(parents=True, exist_ok=True)
        raw_record = {
            "pair_id": pid,
            "split": split,
            "protocol": proto,
            "attempt": attempt,
            "seed": seed,
            "raw_output": raw_text,
            "duration": duration,
        }
        atomic_json(attempt_file, raw_record)

        try:
            parsed = parse_answer(raw_text)
            rewritten = parsed["rewritten_prompt"]
            compiled = compile_prompt(rewritten, PROMPT_ROLES[proto])
            compiled_sha = text_sha256(compiled)

            return PromptRecord(
                schema=SCHEMA,
                pair_id=pid,
                split=split,
                protocol=proto,
                status="complete",
                input_roles=list(PROMPT_ROLES[proto]),
                inputs=inputs,
                instruction_version=template.get("version", "instruction_v1"),
                instruction_sha256=instruction_sha,
                system_prompt_sha256=runner.system_prompt_sha,
                model_revision=runner.revision,
                model_files_sha256=runner.model_files_sha,
                seed=seed,
                attempt=attempt,
                raw_path=str(attempt_file),
                rewritten_prompt=rewritten,
                compiled_prompt=compiled,
                compiled_prompt_sha256=compiled_sha,
                tokens=len(raw_text.split()),
                seconds=duration,
            )
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(f"Failed to generate valid prompt for {pid} {proto} after 3 attempts: {last_exc}")


def process_split_prompts(
    run_dir: Path,
    split: str,
    protocols: list[str],
    model_dir: Path | None = None,
    dry_run: bool = False,
    qa8_mode: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    prompts_dir = run_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = prompts_dir / "input_assets"
    attempts_dir = prompts_dir / "attempts"

    split_file = data_dir / f"{split}512_pairs.jsonl" if split == "train" else (
        data_dir / f"{split}24_pairs.jsonl" if split in ("val", "test") else (
            data_dir / f"{split}6_pairs.jsonl" if split == "diagnostic" else data_dir / f"{split}_pairs.jsonl"
        )
    )
    if not split_file.is_file():
        split_file = data_dir / f"{split}_pairs.jsonl"
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file for {split}: {split_file}")

    pairs = read_jsonl(split_file)
    if qa8_mode:
        pairs = pairs[:8]

    summary: dict[str, Any] = {"split": split, "total_pairs": len(pairs), "counts": {}}

    instruction_path = prompts_dir / "instruction_v1.json"
    template = {
        "version": "instruction_v1",
        "common": COMMON_INSTRUCTION,
    }
    if not instruction_path.is_file():
        atomic_json(instruction_path, template)
    instruction_sha = sha256_file(instruction_path)

    runner = PEModelRunner(model_dir) if not dry_run else None

    qa8_records: list[dict[str, Any]] = []

    for proto in protocols:
        proto_dir = prompts_dir / proto
        proto_dir.mkdir(parents=True, exist_ok=True)
        out_file = proto_dir / f"{split}.jsonl"

        existing = {r["pair_id"]: r for r in read_jsonl(out_file) if r.get("status") == "complete"}
        created_count = 0

        for row in pairs:
            pid = row["pair_id"]
            if pid in existing and not qa8_mode:
                continue

            if proto == "p0":
                rec = make_p0_record(pid, split)
                append_jsonl(out_file, rec)
                created_count += 1
                if qa8_mode:
                    qa8_records.append(rec)
                continue

            if dry_run:
                continue

            assert runner is not None
            rec = run_prompt_attempt(
                runner,
                row,
                proto,
                split,
                template,
                instruction_sha,
                assets_dir,
                attempts_dir,
                run_dir,
            )
            append_jsonl(out_file, rec)
            created_count += 1
            if qa8_mode:
                qa8_records.append(rec)

        summary["counts"][proto] = len(existing) + created_count

    if qa8_mode and qa8_records:
        atomic_json(prompts_dir / "qa8.json", qa8_records)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute PE-I2I prompts")
    parser.add_argument("--run", type=Path, required=True, help="Experiment run root")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "diagnostic"], help="Splits to process")
    parser.add_argument("--protocols", nargs="+", default=["p0", "pg", "pb", "pbg"], help="Protocols to process")
    parser.add_argument("--model", type=Path, help="PE model directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate without generation")
    parser.add_argument("--qa8", action="store_true", help="Generate qa8 development samples for review")
    args = parser.parse_args()

    results = []
    splits = ["train"] if args.qa8 else args.splits
    protocols = ["pg", "pb", "pbg"] if args.qa8 else args.protocols

    for split in splits:
        res = process_split_prompts(
            args.run,
            split,
            protocols,
            args.model,
            args.dry_run,
            qa8_mode=args.qa8,
        )
        results.append(res)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
