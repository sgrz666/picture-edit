from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, TypedDict

from .protocol import PROMPT_ROLES, P0_PROMPT, text_sha256

SCHEMA = "qwen21_pei2i_prompt_v1"

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
    "Keep the source scene, camera framing and overall illumination. Use viewer image-left and image-right for spatial directions. "
    "Omit uncertain or invisible joint and finger details. Never transfer another person's identity, clothes or scene, and never render skeleton lines or normal/depth visualization colors. "
    "The final prompt must state the visible pose changes explicitly and be understandable when the editor receives only the source RGB and numerical geometry controls."
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
            elif "target_dwpose" in row and row["target_dwpose"]:
                found = str(row["target_dwpose"])
            elif "target_dwpose_png" in row and row["target_dwpose_png"]:
                found = str(row["target_dwpose_png"])
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
    parts.extend(f"Image {i + 1}: {ROLE_INSTRUCTIONS[role]}" for i, role in enumerate(roles))
    if protocol == "pbg":
        parts.append(
            "When the target photo and guides visibly disagree, state only reliable observable pose details and omit contradictory details."
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


class PromptBank:
    def __init__(self, path: str | Path, expected_protocol: str, expected_pair_ids: list[str] | set[str]):
        self.path = Path(path)
        self.protocol = expected_protocol
        self.records: dict[str, dict[str, Any]] = {}
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
            self.records[key] = row
        if set(self.records) != set(expected_pair_ids):
            raise ValueError("Prompt manifest does not cover the locked split")

    def prompt_for(self, pair_id: str) -> str:
        return self.records[pair_id]["compiled_prompt"]
