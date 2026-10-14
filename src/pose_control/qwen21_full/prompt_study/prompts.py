from __future__ import annotations

import json
import inspect
import math
from pathlib import Path
import re
from typing import Any, TypedDict, NotRequired

from .protocol import PROMPT_ROLES, P0_PROMPT, text_sha256, sha256_file

SCHEMA = "qwen21_pei2i_prompt_v1"
COMPILER_VERSION = "image_roles_v1"
PE_SAMPLING = {
    "enable_thinking": True,
    "max_new_tokens": 24000,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
}

ROLE_FIELDS = {
    "source_A": "source_rgb",
    "target_B": "target_rgb",
    "target_depth": "target_depth_vis",
    "target_normal": "target_normal",
    "target_dwpose": "filtered_dwpose",
}

ROLE_WORDS = {
    "source_A": "the source image",
    "target_B": "the target pose described in this instruction",
    "target_depth": "the target depth structure described in this instruction",
    "target_normal": "the target surface orientation described in this instruction",
    "target_dwpose": "the target joint arrangement described in this instruction",
}

ROLE_INSTRUCTIONS = {
    "source_A": "This source photograph supplies the person's identity, hair, outfit, accessories and scene.",
    "target_B": "This target photograph supplies only desired body pose, head direction, observable expression and hand configuration.",
    "target_depth": "This grayscale visualization encodes target depth and occlusion; it supplies no clothing or identity appearance.",
    "target_normal": "These RGB colors encode target surface orientation; they are not skin, fabric or lighting colors.",
    "target_dwpose": "This confidence-filtered skeleton encodes visible target joints; black regions and omitted joints supply no hidden anatomy.",
}

COMMON_INSTRUCTION = (
    "Write one standalone English image-edit instruction for changing the pose of the person in the source photograph. "
    "The source photograph supplies the person's facial identity, hair, clothing design, garment colors and material texture, accessories, and background. "
    "Describe the desired body pose, head direction, facial expression, and hand configuration only to the extent they are observable in the permitted target references. "
    "Preserve the source identity and outfit while allowing head orientation, expression, limb position, cloth folds and pose-dependent shadows to adapt. "
    "Keep the source scene, camera framing and overall illumination. Preserve source props, held objects, logos and accessories; pose-only guides cannot justify removing them. "
    "Use only viewer image-left and image-right for spatial directions, never mix them with anatomical left/right. "
    "Omit uncertain or invisible joint and finger details. Never transfer another person's identity, clothes or scene, and never render skeleton lines or normal/depth visualization colors. "
    "The editor keeps the source canvas: do not invent below-frame legs, hands, feet or new framing. "
    "The final prompt must state the visible pose changes explicitly and be understandable when the editor receives only the source RGB and numerical geometry controls. "
    "Write a single detailed paragraph of approximately 200 to 320 English words. State the edit directly, then source preservation and natural pose-dependent cloth/light changes. "
    "The final prompt must not mention Image numbers, reference images, depth/normal/skeleton maps, adapters, target guides, or unseen controls; express the reliable target pose in words."
)


class PromptInput(TypedDict):
    role: str
    path: str
    sha256: str
    size: list[int]
    asset_relative_path: str


class PromptRecord(TypedDict):
    schema: str
    pair_id: str
    split: str
    protocol: str
    status: str
    input_roles: list[str]
    inputs: list[PromptInput]
    instruction_version: str
    instruction_sha256: str
    system_prompt_sha256: str | None
    model_revision: str | None
    model_files_sha256: dict[str, str]
    seed: int | None
    attempt: int | None
    raw_path: str | None
    rewritten_prompt: str
    compiled_prompt: str
    compiled_prompt_sha256: str
    tokens: int
    seconds: float
    model_dir: NotRequired[str]
    template_path: NotRequired[str]
    request_instruction_sha256: NotRequired[str]
    compiler_version: NotRequired[str]
    compiler_sha256: NotRequired[str]
    sampling: NotRequired[dict[str, Any]]
    request_signature: NotRequired[dict[str, Any]]
    request_sha256: NotRequired[str]
    raw_sha256: NotRequired[str]
    peak_gpu_memory_bytes: NotRequired[int | None]


def request_roles(
    row: dict[str, Any],
    protocol: str,
    assets_dir: Path | None = None,
    run_dir: Path | None = None,
) -> list[tuple[str, str]]:
    result = []
    for role in PROMPT_ROLES[protocol]:
        field = ROLE_FIELDS[role]
        if field in row and row[field]:
            result.append((role, str(row[field])))
        elif role == "target_dwpose":
            pair_id = str(row.get("pair_id", ""))
            # Check candidate paths
            found = None
            if assets_dir is not None and (assets_dir / pair_id / "dwpose_filtered.png").is_file():
                found = str(assets_dir / pair_id / "dwpose_filtered.png")
            elif run_dir is not None and (Path(run_dir) / "data/geometry" / pair_id / "dwpose_filtered.png").is_file():
                found = str(Path(run_dir) / "data/geometry" / pair_id / "dwpose_filtered.png")
            if found:
                result.append((role, found))
            else:
                raise KeyError(f"Missing {role} for pair {pair_id}")
        else:
            raise KeyError(f"Missing required role field {field} for {role}")
    return result


def build_instruction(protocol: str, template: dict[str, Any] | None = None) -> str:
    roles = PROMPT_ROLES[protocol]
    if protocol == "p0":
        raise ValueError("P0 does not invoke PE")
    common_text = template.get("common", COMMON_INSTRUCTION) if template else COMMON_INSTRUCTION
    parts = [common_text]
    role_text = template.get("role_instructions", ROLE_INSTRUCTIONS) if template else ROLE_INSTRUCTIONS
    parts.extend(f"Image {i + 1}: {role_text[role]}" for i, role in enumerate(roles))
    if protocol == 'pg':
        parts.append(
            'Geometry-only mode: a fitted bare SMPL-X face is not evidence of a real smile, speaking mouth, gaze or emotion. '
            'Describe only reliable target head orientation. Do not freeze the source head tilt, gaze or expression. '
            'When a target expression cannot be established from geometry, omit expression and gaze instructions entirely; do not request that the source expression be preserved. '
            'Missing or disagreeing hand/finger/leg estimates must be omitted rather than guessed. '
            'The absence of a source prop in a bare-body geometry render is not evidence that the prop should be deleted.'
        )
    if protocol == "pbg":
        parts.append(
            "When the target photo and guides visibly disagree, state only reliable observable pose details and omit contradictory details. "
            "Use the actual target photo for visible expression and source photo for identity, objects and outfit."
        )
    return "\n".join(parts)


def parse_answer(raw: str) -> dict[str, Any]:
    _, sep, answer = raw.partition("</think>")
    answer = (answer if sep else raw).strip()
    # Check for fenced code block anywhere in text
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", answer)
    if fence_match:
        json_str = fence_match.group(1).strip()
    else:
        # Check if text is enclosed in { ... }
        brace_match = re.search(r"\{[\s\S]*\}", answer)
        if brace_match:
            json_str = brace_match.group(0).strip()
        else:
            json_str = answer
    try:
        result = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON response: {exc}") from exc

    text = result.get("rewritten_prompt") if isinstance(result, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Missing nonempty rewritten_prompt")
    return result


def compile_prompt(text: str, input_roles: tuple[str, ...] | list[str]) -> str:
    aliases = {i + 1: ROLE_WORDS[role] for i, role in enumerate(input_roles)}

    def replace(match: re.Match) -> str:
        index = int(match.group(1))
        if index not in aliases:
            raise ValueError(f"Unknown image reference: {index}")
        return aliases[index]

    text = re.sub(r"<image[_\s]*(\d+)>", replace, text, flags=re.I)
    text = re.sub(r"\bimage[_\s]*(\d+)\b", replace, text, flags=re.I)
    text = " ".join(text.split())
    if not text or re.search(r"<image|\bimage[_\s]*\d+", text, re.I):
        raise ValueError("Unresolved image reference")
    return text


def make_p0_record(pair_id: str, split: str) -> PromptRecord:
    compiled = P0_PROMPT
    return PromptRecord(
        schema=SCHEMA,
        pair_id=pair_id,
        split=split,
        protocol="p0",
        status="complete",
        input_roles=[],
        inputs=[],
        instruction_version="none",
        instruction_sha256="",
        system_prompt_sha256=None,
        model_revision=None,
        model_files_sha256={},
        seed=None,
        attempt=None,
        raw_path=None,
        rewritten_prompt=compiled,
        compiled_prompt=compiled,
        compiled_prompt_sha256=text_sha256(compiled),
        tokens=0,
        seconds=0.0,
    )


def compiler_sha256() -> str:
    return text_sha256(inspect.getsource(compile_prompt) + json.dumps(ROLE_WORDS, sort_keys=True))


def request_sha256(signature: dict[str, Any]) -> str:
    return text_sha256(json.dumps(signature, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _relative_file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("invalid relative path")
    relative = Path(value)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError("invalid relative path")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("relative path escapes run")
    return result


def validate_prompt_record(row: dict[str, Any], *, run_dir: Path | None = None) -> None:
    """Validate immutable PE evidence without needing the original model checkout."""
    protocol = row.get("protocol")
    if row.get("schema") != SCHEMA or protocol not in PROMPT_ROLES or row.get("status") != "complete":
        raise ValueError("Invalid prompt provenance: schema/protocol/status")
    if protocol == "p0":
        if (row.get("input_roles") != [] or row.get("inputs") != []
                or row.get("rewritten_prompt") != P0_PROMPT or row.get("compiled_prompt") != P0_PROMPT
                or row.get("tokens") != 0 or row.get("seconds") != 0
                or row.get("compiled_prompt_sha256") != text_sha256(P0_PROMPT)
                or row.get("instruction_version") != "none" or row.get("instruction_sha256") != ""
                or any(row.get(field) for field in ("system_prompt_sha256", "model_revision", "model_files_sha256", "raw_path",
                    "model_dir", "raw_sha256", "request_signature", "request_sha256", "sampling", "compiler_sha256"))
                or row.get("seed") is not None or row.get("attempt") is not None):
            raise ValueError("Invalid P0 provenance")
        return
    try:
        def digest(value):
            return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None

        if tuple(row["input_roles"]) != PROMPT_ROLES[protocol]:
            raise ValueError("roles")
        if tuple(item["role"] for item in row["inputs"]) != PROMPT_ROLES[protocol]:
            raise ValueError("input roles")
        for item in row["inputs"]:
            if not digest(item["sha256"]) or not item["path"] or len(item["size"]) != 2:
                raise ValueError("input metadata")
            if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in item["size"]):
                raise ValueError("input dimensions")
            _relative_file(run_dir or Path.cwd(), item["asset_relative_path"])
        required_hashes = ("instruction_sha256", "system_prompt_sha256", "request_instruction_sha256",
                           "compiler_sha256", "request_sha256", "raw_sha256")
        if not all(digest(row.get(key)) for key in required_hashes):
            raise ValueError("missing hashes")
        model_hashes = row["model_files_sha256"]
        if (not row.get("model_dir") or not isinstance(model_hashes, dict) or not model_hashes
                or not all(digest(value) for value in model_hashes.values())
                or not any(name.endswith((".safetensors", ".bin")) for name in model_hashes)
                or model_hashes.get("system_prompt.txt") != row["system_prompt_sha256"]):
            raise ValueError("model files")
        if (row["sampling"] != PE_SAMPLING or row["sampling"].get("enable_thinking") is not True
                or row["sampling"].get("do_sample") is not True
                or row["compiler_version"] != COMPILER_VERSION or row["compiler_sha256"] != compiler_sha256()):
            raise ValueError("sampling/compiler")
        if row["attempt"] not in (0, 1, 2) or row["seed"] != 42 + row["attempt"]:
            raise ValueError("seed")
        if not isinstance(row["tokens"], int) or isinstance(row["tokens"], bool) or not 0 < row["tokens"] <= PE_SAMPLING["max_new_tokens"]:
            raise ValueError("generated token count")
        if not isinstance(row["seconds"], (int, float)) or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
            raise ValueError("duration")
        raw_path = _relative_file(run_dir or Path.cwd(), row["raw_path"])
        _relative_file(run_dir or Path.cwd(), row["template_path"])
        signature = row["request_signature"]
        if request_sha256(signature) != row["request_sha256"]:
            raise ValueError("request signature")
        expected = {"pair_id": row["pair_id"], "split": row["split"], "protocol": protocol,
                    "inputs": row["inputs"], "instruction_sha256": row["instruction_sha256"],
                    "request_instruction_sha256": row["request_instruction_sha256"],
                    "system_prompt_sha256": row["system_prompt_sha256"], "model_dir": row["model_dir"],
                    "model_revision": row["model_revision"], "model_files_sha256": model_hashes,
                    "sampling": row["sampling"], "compiler_version": row["compiler_version"],
                    "compiler_sha256": row["compiler_sha256"]}
        if signature != expected:
            raise ValueError("request metadata")
        if compile_prompt(row["rewritten_prompt"], PROMPT_ROLES[protocol]) != row["compiled_prompt"]:
            raise ValueError("compiled text")
        if text_sha256(row["compiled_prompt"]) != row["compiled_prompt_sha256"]:
            raise ValueError("compiled hash")
        if run_dir is not None:
            template_path = _relative_file(run_dir, row["template_path"])
            if sha256_file(template_path) != row["instruction_sha256"]:
                raise ValueError("template changed")
            template = json.loads(template_path.read_text(encoding="utf-8"))
            if row["instruction_version"] != template["version"]:
                raise ValueError("template version")
            if text_sha256(build_instruction(protocol, template)) != row["request_instruction_sha256"]:
                raise ValueError("request instruction")
            for item in row["inputs"]:
                asset = _relative_file(run_dir, item["asset_relative_path"])
                if sha256_file(asset) != item["sha256"]:
                    raise ValueError("input asset hash")
                from PIL import Image
                with Image.open(asset) as image:
                    if [image.width, image.height] != item["size"]:
                        raise ValueError("input asset dimensions")
                source = Path(item["path"])
                if source.is_file() and sha256_file(source) != item["sha256"]:
                    raise ValueError("input source changed")
            if sha256_file(raw_path) != row["raw_sha256"]:
                raise ValueError("raw hash")
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if (raw["request_sha256"] != row["request_sha256"] or raw["request_signature"] != signature
                    or raw["status"] != "complete" or raw["tokens"] != row["tokens"]
                    or raw["seed"] != row["seed"] or raw["attempt"] != row["attempt"]
                    or raw["system_prompt_sha256"] != row["system_prompt_sha256"]
                    or text_sha256(raw["system_prompt"]) != row["system_prompt_sha256"]
                    or text_sha256(raw["instruction"]) != row["request_instruction_sha256"]
                    or raw["compiled_prompt"] != row["compiled_prompt"]
                    or parse_answer(raw["raw_output"])["rewritten_prompt"] != row["rewritten_prompt"]):
                raise ValueError("raw evidence")
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise ValueError(f"Invalid PE provenance for {row.get('pair_id')}: {exc}") from exc


class PromptBank:
    def __init__(self, path: str | Path, expected_protocol: str, expected_pair_ids: list[str] | set[str]):
        self.path = Path(path)
        self.protocol = expected_protocol
        self.records: dict[str, dict[str, Any]] = {}
        run_dir = next((parent.parent for parent in self.path.resolve().parents if parent.name == "prompts"), self.path.resolve().parent)
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["pair_id"]
            if key in self.records:
                raise ValueError(f"Duplicate prompt: {key}")
            if row["protocol"] != expected_protocol or row["status"] != "complete":
                raise ValueError(f"Invalid prompt record: {key}")
            if tuple(row["input_roles"]) != PROMPT_ROLES[expected_protocol]:
                raise ValueError(f"Prompt roles differ: {key}")
            if tuple(item["role"] for item in row["inputs"]) != PROMPT_ROLES[expected_protocol]:
                raise ValueError(f"Actual prompt inputs differ: {key}")
            if not isinstance(row["compiled_prompt"], str) or not row["compiled_prompt"].strip():
                raise ValueError(f"Empty compiled prompt: {key}")
            if expected_protocol == "p0" and row["compiled_prompt"] != P0_PROMPT:
                raise ValueError(f"P0 differs from fixed baseline: {key}")
            if text_sha256(row["compiled_prompt"]) != row["compiled_prompt_sha256"]:
                raise ValueError(f"Prompt hash differs: {key}")
            validate_prompt_record(row, run_dir=run_dir)
            self.records[key] = row
        if set(self.records) != set(expected_pair_ids):
            raise ValueError("Prompt manifest does not cover the locked split")

    def prompt_for(self, pair_id: str) -> str:
        return self.records[pair_id]["compiled_prompt"]
