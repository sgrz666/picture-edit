#!/usr/bin/env python3
"""Precompute official PE-I2I rewrites with immutable evidence and exact reuse."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pose_control.qwen21_full.prompt_study.protocol import (
    PROTOCOLS, PROMPT_ROLES, atomic_json, sha256_file, text_sha256, load_study_lock,
)
from src.pose_control.qwen21_full.prompt_study.prompts import (
    SCHEMA, COMMON_INSTRUCTION, ROLE_INSTRUCTIONS, COMPILER_VERSION, PE_SAMPLING,
    build_instruction, request_roles, parse_answer, compile_prompt, compiler_sha256,
    request_sha256, validate_prompt_record, make_p0_record, PromptInput, PromptRecord,
    PromptBank,
)

DEFAULT_PE_MODEL = Path("/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8", newline="\n")
    temporary.replace(path)


def split_manifest(run_dir: Path, split: str) -> Path:
    suffix = {"train": "512", "val": "24", "test": "24", "diagnostic": "6"}.get(split, "")
    for name in (f"{split}{suffix}_pairs.jsonl", f"{split}_pairs.jsonl"):
        path = run_dir / "data" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing split file for {split} under {run_dir / 'data'}")


def default_template() -> dict[str, Any]:
    return {"version": "instruction_v2", "common": COMMON_INSTRUCTION,
            "role_instructions": dict(ROLE_INSTRUCTIONS), "sampling": dict(PE_SAMPLING),
            "compiler_version": COMPILER_VERSION}


def prepare_input_assets(roles_and_paths: list[tuple[str, str]], assets_dir: Path) -> list[PromptInput]:
    from PIL import Image
    assets_dir.mkdir(parents=True, exist_ok=True)
    inputs = []
    for role, path_str in roles_and_paths:
        path = Path(path_str)
        if not path.is_file():
            raise FileNotFoundError(f"Input image for role {role} not found: {path}")
        sha = sha256_file(path)
        name = f"{sha}{path.suffix or '.png'}"
        target = assets_dir / name
        if not target.is_file():
            shutil.copyfile(path, target)
        elif sha256_file(target) != sha:
            raise ValueError(f"Corrupt PE input asset: {target}")
        with Image.open(path) as image:
            size = [image.width, image.height]
        inputs.append(PromptInput(role=role, path=str(path), sha256=sha, size=size,
                                  asset_relative_path=f"prompts/input_assets/{name}"))
    return inputs


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    tokens: int
    peak_gpu_memory_bytes: int | None = None


class PEModelRunner:
    def __init__(self, model_dir: Path | None, device: str = "cuda", *, processor=None, model=None):
        if model_dir is None:
            raise ValueError("PE model directory is required")
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"PE model directory not found: {self.model_dir}")
        system_path = self.model_dir / "system_prompt.txt"
        if not system_path.is_file():
            raise FileNotFoundError(f"Official system_prompt.txt missing: {system_path}")
        self.system_prompt = system_path.read_bytes().decode("utf-8")
        if not self.system_prompt.strip():
            raise ValueError("Official system_prompt.txt is empty")
        self.system_prompt_sha = sha256_file(system_path)
        self.device = device
        self.model_files_sha = {
            path.relative_to(self.model_dir).as_posix(): sha256_file(path)
            for path in sorted(self.model_dir.rglob("*"))
            if path.is_file() and ".cache" not in path.parts
            and path.suffix in (".json", ".txt", ".model", ".jinja", ".safetensors", ".bin")
        }
        if not any(name.endswith((".safetensors", ".bin")) for name in self.model_files_sha):
            raise FileNotFoundError(f"PE model weight files missing: {self.model_dir}")
        self.revision = None
        config_path = self.model_dir / "config.json"
        if config_path.is_file():
            self.revision = json.loads(config_path.read_text(encoding="utf-8")).get("_commit_hash")
        if self.revision is None and self.model_dir.parent.name == "snapshots":
            self.revision = self.model_dir.name
        if (processor is None) != (model is None):
            raise ValueError("Explicit test injection requires both processor and model")
        if model is None:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
            processor = AutoProcessor.from_pretrained(str(self.model_dir))
            model = AutoModelForImageTextToText.from_pretrained(
                str(self.model_dir), torch_dtype=torch.bfloat16,
            ).to(device)
            model.eval()
        self.processor, self.model = processor, model

    def generate(self, instruction: str, images: list[Any], seed: int = 42) -> GenerationResult:
        import numpy as np
        import torch
        if self.model is None or self.processor is None:
            raise RuntimeError("PE model/processor must be loaded before generation")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        using_cuda = torch.device(self.device).type == "cuda" and torch.cuda.is_available()
        if using_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": instruction})
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        inputs = self.processor(text=[prompt_text], images=images if images else None,
                                padding=True, return_tensors="pt").to(self.device)
        sampling = {key: value for key, value in PE_SAMPLING.items() if key != "enable_thinking"}
        with torch.inference_mode():
            output = self.model.generate(**inputs, **sampling)
        generated = output[0][inputs["input_ids"].shape[1]:]
        if len(generated) == 0:
            raise RuntimeError("PE generated no tokens")
        peak = torch.cuda.max_memory_allocated(self.device) if using_cuda else None
        return GenerationResult(self.processor.decode(generated, skip_special_tokens=True), len(generated), peak)


def record_attempt_timing(timing_path: Path, raw: dict[str, Any], attempt_file: Path, run_dir: Path) -> None:
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.is_file() else {
        "schema": "qwen21_pei2i_timing_v1", "attempt_count": 0, "failed_attempt_count": 0,
        "retry_attempt_count": 0, "generated_tokens": 0, "seconds": 0.0, "first20": [],
        "peak_gpu_memory_bytes": None,
    }
    timing["attempt_count"] += 1
    timing["failed_attempt_count"] += int(raw["status"] != "complete")
    timing["retry_attempt_count"] += int(raw["attempt"] > 0)
    timing["generated_tokens"] += raw["tokens"]
    timing["seconds"] += raw["seconds"]
    if raw.get("peak_gpu_memory_bytes") is not None:
        timing["peak_gpu_memory_bytes"] = max(timing["peak_gpu_memory_bytes"] or 0, raw["peak_gpu_memory_bytes"])
    if len(timing["first20"]) < 20:
        timing["first20"].append({**{key: raw[key] for key in (
            "pair_id", "split", "protocol", "attempt", "seed", "tokens", "seconds", "status")},
            "raw_path": attempt_file.relative_to(run_dir).as_posix(),
            "peak_gpu_memory_bytes": raw.get("peak_gpu_memory_bytes")})
    timing["failure_rate"] = timing["failed_attempt_count"] / timing["attempt_count"]
    timing["mean_attempt_seconds"] = timing["seconds"] / timing["attempt_count"]
    atomic_json(timing_path, timing)


def prompt_signature(runner: PEModelRunner, pair_id: str, split: str, protocol: str,
                     inputs: list[PromptInput], instruction_sha: str, instruction: str) -> dict[str, Any]:
    return {"pair_id": pair_id, "split": split, "protocol": protocol, "inputs": inputs,
            "instruction_sha256": instruction_sha, "request_instruction_sha256": text_sha256(instruction),
            "system_prompt_sha256": runner.system_prompt_sha, "model_dir": str(runner.model_dir),
            "model_revision": runner.revision, "model_files_sha256": runner.model_files_sha,
            "sampling": dict(PE_SAMPLING), "compiler_version": COMPILER_VERSION,
            "compiler_sha256": compiler_sha256()}


def run_prompt_attempt(runner: PEModelRunner, row: dict[str, Any], proto: str, split: str,
                       template: dict[str, Any], instruction_sha: str, assets_dir: Path,
                       attempts_dir: Path, run_dir: Path, *, inputs=None, signature=None) -> PromptRecord:
    from PIL import Image
    pid = str(row["pair_id"])
    if pid in ("", ".", "..") or "/" in pid or "\\" in pid:
        raise ValueError(f"Unsafe pair_id: {pid}")
    if inputs is None:
        inputs = prepare_input_assets(request_roles(row, proto, assets_dir=run_dir / "data/geometry",
                                                     run_dir=run_dir), assets_dir)
    images = []
    for item in inputs:
        with Image.open(item["path"]) as image:
            images.append(image.convert("RGB"))
    instruction = build_instruction(proto, template)
    signature = signature or prompt_signature(runner, pid, split, proto, inputs, instruction_sha, instruction)
    request_hash = request_sha256(signature)
    attempt_root = attempts_dir / split / pid / proto / request_hash / str(time.time_ns())
    last_error = None
    try:
        for attempt, seed in enumerate((42, 43, 44)):
            attempt_file = attempt_root / f"attempt_{attempt}.json"
            raw_record = {"schema": SCHEMA, "pair_id": pid, "split": split, "protocol": proto,
                          "attempt": attempt, "seed": seed, "request_signature": signature,
                          "request_sha256": request_hash, "instruction": instruction,
                          "system_prompt": runner.system_prompt, "system_prompt_sha256": runner.system_prompt_sha,
                          "input_roles": list(PROMPT_ROLES[proto]), "inputs": inputs,
                          "processor_preprocessing": "Official AutoProcessor; original RGB dimensions provided",
                          "raw_output": None, "tokens": 0, "status": "failed"}
            started = time.perf_counter()
            try:
                result = runner.generate(instruction, images, seed=seed)
                if not isinstance(result, GenerationResult) or result.tokens <= 0:
                    raise RuntimeError("PE must return actual generated token ids/count")
                raw_record.update(raw_output=result.raw_output, tokens=result.tokens,
                                  peak_gpu_memory_bytes=result.peak_gpu_memory_bytes)
            except Exception as exc:
                raw_record.update(seconds=time.perf_counter() - started,
                                  error=f"{type(exc).__name__}: {exc}")
                atomic_json(attempt_file, raw_record)
                record_attempt_timing(attempts_dir.parent / "timing.json", raw_record, attempt_file, run_dir)
                raise
            duration = time.perf_counter() - started
            raw_record["seconds"] = duration
            try:
                rewritten = parse_answer(result.raw_output)["rewritten_prompt"]
                compiled = compile_prompt(rewritten, PROMPT_ROLES[proto])
            except (ValueError, TypeError) as exc:
                last_error = exc
                raw_record["error"] = f"{type(exc).__name__}: {exc}"
                atomic_json(attempt_file, raw_record)
                record_attempt_timing(attempts_dir.parent / "timing.json", raw_record, attempt_file, run_dir)
                continue
            raw_record.update(status="complete", rewritten_prompt=rewritten, compiled_prompt=compiled)
            atomic_json(attempt_file, raw_record)
            record_attempt_timing(attempts_dir.parent / "timing.json", raw_record, attempt_file, run_dir)
            return PromptRecord(
                schema=SCHEMA, pair_id=pid, split=split, protocol=proto, status="complete",
                input_roles=list(PROMPT_ROLES[proto]), inputs=inputs,
                instruction_version=template["version"], instruction_sha256=instruction_sha,
                template_path="prompts/instruction_v1.json", **{
                    key: signature[key] for key in ("request_instruction_sha256", "system_prompt_sha256",
                        "model_dir", "model_revision", "model_files_sha256", "sampling",
                        "compiler_version", "compiler_sha256")},
                request_signature=signature, request_sha256=request_hash,
                seed=seed, attempt=attempt, raw_path=attempt_file.relative_to(run_dir).as_posix(),
                raw_sha256=sha256_file(attempt_file), rewritten_prompt=rewritten,
                compiled_prompt=compiled, compiled_prompt_sha256=text_sha256(compiled),
                tokens=result.tokens, seconds=duration, peak_gpu_memory_bytes=result.peak_gpu_memory_bytes,
            )
    finally:
        for image in images:
            image.close()
    raise RuntimeError(f"Failed valid PE prompt for {pid}/{proto} after 3 attempts: {last_error}")


def process_split_prompts(run_dir: Path, split: str, protocols: list[str],
                         model_dir: Path | None = DEFAULT_PE_MODEL, dry_run: bool = False,
                         qa8_mode: bool = False, *, runner: PEModelRunner | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if split == "test":
        from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed
        require_eight_arms_completed(run_dir)
    if not protocols or any(proto not in PROTOCOLS for proto in protocols) or len(set(protocols)) != len(protocols):
        raise ValueError("Unsupported or duplicate prompt protocols")
    pairs = read_jsonl(split_manifest(run_dir, split))
    if qa8_mode:
        if split != "train":
            raise ValueError("QA8 only uses train development pairs")
        pairs = pairs[:8]
    pair_ids = [str(row["pair_id"]) for row in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError(f"Duplicate pair IDs in {split} split")
    prompts_dir = run_dir / "prompts"
    namespace = prompts_dir / "qa8" if qa8_mode else prompts_dir
    template_path = prompts_dir / "instruction_v1.json"
    if template_path.is_file():
        template = json.loads(template_path.read_text(encoding="utf-8"))
    else:
        template = default_template()
        if not dry_run:
            atomic_json(template_path, template)
    if (not isinstance(template.get("common"), str) or not template["common"].strip()
            or template.get("sampling", PE_SAMPLING) != PE_SAMPLING
            or template.get("compiler_version", COMPILER_VERSION) != COMPILER_VERSION):
        raise ValueError("Invalid locked PE template/sampling/compiler")
    instruction_sha = sha256_file(template_path) if template_path.is_file() else request_sha256(template)
    if not dry_run and any(proto != "p0" for proto in protocols) and runner is None:
        runner = PEModelRunner(model_dir)
    lock_path = run_dir / "data/lock.json"
    if lock_path.is_file():
        lock = load_study_lock(lock_path, verify_assets=True)
        artifact = lock["artifacts"].get("pe_model")
        if runner is not None and artifact is not None:
            if (Path(artifact["path"]).resolve() != runner.model_dir
                    or not artifact.get("files")
                    or any(runner.model_files_sha.get(name) != value["sha256"] for name, value in artifact["files"].items())):
                raise ValueError("PE model provenance differs from locked official model")
    summary = {"split": split, "total_pairs": len(pairs), "counts": {}, "created": {}, "reused": {}}
    for proto in protocols:
        out_file = namespace / proto / f"{split}.jsonl"
        existing_rows = read_jsonl(out_file)
        existing = {str(row["pair_id"]): row for row in existing_rows}
        if len(existing) != len(existing_rows):
            raise ValueError(f"Duplicate prompt records in {out_file}")
        if set(existing) - set(pair_ids):
            raise ValueError(f"Prompt manifest contains pairs outside {split}")
        qa_rows = [] if qa8_mode else read_jsonl(prompts_dir / "qa8" / proto / f"{split}.jsonl")
        qa_existing = {str(row["pair_id"]): row for row in qa_rows}
        output = {}
        created = reused = 0
        for row in pairs:
            pid = str(row["pair_id"])
            if proto == "p0":
                rec = make_p0_record(pid, split)
                reused += int(existing.get(pid) == rec)
                created += int(existing.get(pid) != rec and not dry_run)
            else:
                inputs = prepare_input_assets(request_roles(row, proto, assets_dir=run_dir / "data/geometry",
                                                           run_dir=run_dir), prompts_dir / "input_assets") if not dry_run else None
                if dry_run:
                    for _, path in request_roles(row, proto, assets_dir=run_dir / "data/geometry", run_dir=run_dir):
                        if not Path(path).is_file():
                            raise FileNotFoundError(f"PE input missing: {path}")
                    continue
                assert runner is not None
                signature = prompt_signature(runner, pid, split, proto, inputs, instruction_sha,
                                             build_instruction(proto, template))
                rec = None
                for candidate in (existing.get(pid), qa_existing.get(pid)):
                    if candidate is None or candidate.get("request_sha256") != request_sha256(signature):
                        continue
                    try:
                        validate_prompt_record(candidate, run_dir=run_dir)
                    except ValueError:
                        continue
                    rec = candidate
                    reused += 1
                    break
                if rec is None:
                    rec = run_prompt_attempt(runner, row, proto, split, template, instruction_sha,
                                             prompts_dir / "input_assets", namespace / "attempts", run_dir,
                                             inputs=inputs, signature=signature)
                    validate_prompt_record(rec, run_dir=run_dir)
                    created += 1
            output[pid] = rec
            if not dry_run:
                # Persist each success atomically; never append duplicate pair IDs.
                existing[pid] = rec
                write_jsonl(out_file, [existing[key] for key in pair_ids if key in existing])
        summary["counts"][proto] = len(output) if not dry_run else 0
        summary["created"][proto] = created
        summary["reused"][proto] = reused
    if qa8_mode and not dry_run:
        manifests = {proto: (namespace / proto / f"{split}.jsonl").relative_to(run_dir).as_posix()
                     for proto in protocols}
        atomic_json(namespace / "manifest.json", {"schema": SCHEMA, "namespace": "qa8",
                    "split": split, "pair_ids": pair_ids, "instruction_sha256": instruction_sha,
                    "manifests": manifests, "counts": summary["counts"]})
        review_path = prompts_dir / "qa8.json"
        if not review_path.is_file():
            atomic_json(review_path, {"reviewed": False, "manifest": "prompts/qa8/manifest.json",
                                     "instruction_sha256": instruction_sha, "notes": []})
    return summary


def qa8_review_provenance(run_dir: Path) -> dict[str, Any]:
    """Build review metadata from 24 validated records; never marks them reviewed."""
    run_dir = Path(run_dir)
    try:
        lock = load_study_lock(run_dir / "data/lock.json", verify_assets=True)
        artifact = lock["artifacts"]["pe_model"]
        locked_rows = read_jsonl(run_dir / "data" / lock["splits"]["train"]["path"])[:8]
        pair_ids = [str(row["pair_id"]) for row in locked_rows]
        if len(pair_ids) != 8 or len(set(pair_ids)) != 8:
            raise ValueError("locked first eight train pairs are missing/duplicated")
        manifest_path = run_dir / "prompts/qa8/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        protocols = ("pg", "pb", "pbg")
        paths = {proto: f"prompts/qa8/{proto}/train.jsonl" for proto in protocols}
        instruction_sha = sha256_file(run_dir / "prompts/instruction_v1.json")
        if (manifest.get("schema") != SCHEMA or manifest.get("namespace") != "qa8"
                or manifest.get("split") != "train" or manifest.get("pair_ids") != pair_ids
                or manifest.get("manifests") != paths or manifest.get("counts") != {p: 8 for p in protocols}
                or manifest.get("instruction_sha256") != instruction_sha):
            raise ValueError("development manifest does not cover the current locked 8x3 requests")
        model_dir = str(Path(artifact["path"]).resolve())
        locked_hashes = {name: value["sha256"] for name, value in artifact["files"].items()}
        requests, bank_hashes = {}, {}
        model_hashes = model_revision = None
        for proto in protocols:
            path = run_dir / paths[proto]
            bank = PromptBank(path, proto, pair_ids)
            requests[proto] = {}
            bank_hashes[proto] = sha256_file(path)
            for pair in locked_rows:
                pid = str(pair["pair_id"])
                record = bank.records[pid]
                if record["split"] != "train" or Path(record["model_dir"]).resolve() != Path(model_dir):
                    raise ValueError("development split/model differs from locked PE")
                if any(record["model_files_sha256"].get(name) != digest for name, digest in locked_hashes.items()):
                    raise ValueError("development model files differ from locked PE")
                if model_hashes is None:
                    model_hashes, model_revision = record["model_files_sha256"], record["model_revision"]
                elif record["model_files_sha256"] != model_hashes or record["model_revision"] != model_revision:
                    raise ValueError("development records used different model versions")
                permitted = request_roles(pair, proto, assets_dir=run_dir / "data/geometry", run_dir=run_dir)
                for item, (role, source_path) in zip(record["inputs"], permitted, strict=True):
                    if (item["role"] != role or Path(item["path"]).resolve() != Path(source_path).resolve()
                            or item["sha256"] != sha256_file(source_path)):
                        raise ValueError("development inputs differ from locked permitted images")
                requests[proto][pid] = record["request_sha256"]
        return {"schema": "qwen21_pei2i_qa_review_v1", "manifest": "prompts/qa8/manifest.json",
                "instruction_sha256": instruction_sha, "qa_manifest_sha256": sha256_file(manifest_path),
                "pair_ids": pair_ids, "prompt_manifests_sha256": bank_hashes,
                "model_dir": model_dir, "model_revision": model_revision, "model_files_sha256": model_hashes,
                "model_manifest_sha256": artifact["manifest_sha256"], "compiler_version": COMPILER_VERSION,
                "compiler_sha256": compiler_sha256(), "request_sha256": requests}
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid QA8 development provenance: {exc}") from exc


def validate_qa8_review(run_dir: Path) -> dict[str, Any]:
    """Formal PE requires an explicit review bound to the current development evidence."""
    try:
        review = json.loads((Path(run_dir) / "prompts/qa8.json").read_text(encoding="utf-8"))
        if review.get("reviewed") is not True or not review.get("notes"):
            raise ValueError("24 development replies/images must be reviewed with recorded notes")
        unsigned = {key: value for key, value in review.items() if key != "review_signature_sha256"}
        if review.get("review_signature_sha256") != request_sha256(unsigned):
            raise ValueError("review signature differs")
        expected = qa8_review_provenance(Path(run_dir))
        if any(review.get(key) != value for key, value in expected.items()):
            raise ValueError("review is stale for the current template/model/compiler/development requests")
        return review
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(f"QA8 review required before formal PE: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute official PE-I2I prompts")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "diagnostic"],
                        choices=("train", "val", "test", "diagnostic"))
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--model", type=Path, default=DEFAULT_PE_MODEL)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without generating manifests")
    parser.add_argument("--qa8", action="store_true", help="Separate development namespace, first 8 train pairs")
    args = parser.parse_args()
    splits = ["train"] if args.qa8 else args.splits
    protocols = ["pg", "pb", "pbg"] if args.qa8 else args.protocols
    if "test" in splits:
        from src.pose_control.qwen21_full.prompt_study.protocol import require_eight_arms_completed
        require_eight_arms_completed(args.run)
    for split in splits:
        split_manifest(args.run, split)
    if not args.qa8 and any(proto != "p0" for proto in protocols):
        review = validate_qa8_review(args.run)
        if Path(args.model).resolve() != Path(review["model_dir"]).resolve():
            raise ValueError("QA8 reviewed model differs from requested formal PE model")
    runner = PEModelRunner(args.model) if not args.dry_run and any(proto != "p0" for proto in protocols) else None
    results = []
    for split in splits:
        results.append(process_split_prompts(args.run, split,
            protocols, args.model, args.dry_run, qa8_mode=args.qa8, runner=runner))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

