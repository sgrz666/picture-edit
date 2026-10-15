"""Prompt-contract pieces that do not depend on how many people a task has.

The single-person and the two-person workflow differ in which images exist and what role
each one plays. They do not differ in how the official prompt-engineering answer is
parsed, nor in the rule that a generator may only be pointed at images it actually
receives. Those shared halves live here so the two workflows cannot drift apart.
"""
from __future__ import annotations

import json
import re

PE_SAMPLING = {"max_new_tokens": 24000, "do_sample": True, "temperature": 1.0,
    "top_p": 0.95, "top_k": 20}
IMAGE_REFERENCE = re.compile(r"<image[_\s]*(\d+)>|\bimage[_\s]*(\d+)\b", re.I)


def parse_pe_answer(raw):
    """Read the official three-field JSON answer, tolerating thinking text and fences."""
    answer = raw.split("</think>")[-1].strip()
    if answer.startswith("```"):
        answer = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer, flags=re.I)
    start = answer.find("{")
    if start < 0:
        raise ValueError("PE returned no JSON answer")
    payload = answer[start:]
    result, end = json.JSONDecoder().raw_decode(payload)
    if payload[end:].strip() or not isinstance(result, dict):
        raise ValueError("PE returned malformed JSON or trailing content")
    text = result.get("rewritten_prompt")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("PE returned an empty rewritten_prompt")
    if result.get("ratio_follow") != "<image1>" or result.get("wh_ratio", "") != "":
        raise ValueError("Pose transfer must preserve the source image canvas")
    return result


def referenced_images(text):
    return sorted({int(match.group(1) or match.group(2))
        for match in IMAGE_REFERENCE.finditer(text)})


def validate_generator_prompt(text, allowed=(1, 2)):
    """Reject references the generator cannot resolve.

    A workflow hands the generator a fixed set of images, so a prompt naming anything
    outside that set is a contract break rather than a style choice. This guards both
    compiled PE output and a hand-supplied --prompt-file.
    """
    allowed = set(allowed)
    indices = set(referenced_images(text))
    unknown = sorted(indices - allowed)
    if unknown:
        raise ValueError(f"Generator prompt refers to image{unknown[0]}, but only images "
            f"{sorted(allowed)} exist downstream")
    if not indices:
        raise ValueError("Generator prompt refers to none of the images it will receive")
    return text
