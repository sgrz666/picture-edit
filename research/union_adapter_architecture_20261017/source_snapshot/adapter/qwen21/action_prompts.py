"""Deterministic action text for editing two existing people in a source photo."""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch


PROMPT_VERSION = "hi4d_dual_action_v2"

_PRESERVE = (
    "Preserve each person's face, hair, clothing and proportions, as well as the camera view, "
    "lighting and background. Do not add another person or change the setting."
)
_ACTIONS = {
    "highfive": ("turn toward each other and high-five, meeting their raised open palms", "high-five"),
    "fight": ("fight each other with opposing arm and body movements", "fight"),
    "talk": ("talk to each other face to face with natural conversational gestures", "talk"),
    "argue": ("argue face to face with emphatic hand gestures", "argue"),
    "dance": ("dance together with coordinated movements", "dance together"),
    "jump": ("jump together, with both people visibly off the ground", "jump together"),
    "bend": ("bend forward together at the waist", "bend forward"),
    "basketball": ("perform a basketball play together with athletic movements", "play basketball together"),
    "game": ("play a game together, interacting with each other", "play a game together"),
    "kiss": ("move toward each other and kiss", "kiss"),
    "backhug": ("move into a back hug, with one person wrapping their arms around the other from behind", "form a back hug"),
    "pose": ("pose together with deliberate body positions", "pose together"),
    "taichi": ("practice tai chi together with controlled, flowing movements", "practice tai chi together"),
    "yoga": ("practice yoga together in balanced poses", "practice yoga together"),
}


def _horizontal_center(mask: torch.Tensor) -> float:
    mask = torch.as_tensor(mask).detach().cpu()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("each source person mask must have shape [H, W] or [1, H, W]")
    foreground = mask > 0
    xs = foreground.nonzero(as_tuple=True)[1]
    if xs.numel() == 0:
        raise ValueError("source person mask is empty")
    return xs.float().mean().item()


def build_dual_action_prompts(
    target_action: str, source_masks: Sequence[torch.Tensor]
) -> tuple[str, str]:
    """Build native and adapter prompts from only the action label and source masks."""
    if not isinstance(target_action, str):
        raise ValueError("unsupported action label")
    action = re.sub(r"\d+$", "", target_action.strip().lower())
    if action not in _ACTIONS and action != "hug":
        raise ValueError(f"unsupported action: {target_action}")
    if len(source_masks) != 2:
        raise ValueError("exactly two source person masks are required")
    center_a, center_b = (_horizontal_center(mask) for mask in source_masks)
    if center_a == center_b:
        raise ValueError("source person masks have ambiguous left-right order")
    if action == "hug":
        return (
            "Edit the supplied photo of exactly two existing people. The person on the left and the person "
            "on the right move toward each other and hug. Their torsos touch and their arms wrap around "
            "each other's shoulders and backs. " + _PRESERVE,
            "Make the two existing people hug. Preserve their appearance and background.",
        )
    left, right = ("A", "B") if center_a < center_b else ("B", "A")
    motion, short_motion = _ACTIONS[action]
    return (
        "Edit the supplied photo of exactly two existing people. "
        f"The person on the left (source person {left}) and the person on the right "
        f"(source person {right}) {motion}. " + _PRESERVE,
        f"Make the two existing people {short_motion}. Preserve their appearance and background.",
    )
