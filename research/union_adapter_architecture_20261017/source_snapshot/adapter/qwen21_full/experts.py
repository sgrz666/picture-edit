from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RoutingDecision:
    num_people: torch.Tensor
    single_mask: torch.Tensor
    dual_mask: torch.Tensor


class TaskRouter:
    """Route exclusively from the validated number of SMPL-X instances."""

    def resolve(self, person_count: torch.Tensor) -> RoutingDecision:
        if person_count.ndim != 1:
            raise ValueError("person_count must have shape [B]")
        num_people = person_count.long()
        if torch.any((num_people < 1) | (num_people > 2)):
            raise ValueError("router supports one or two people only")
        return RoutingDecision(
            num_people=num_people,
            single_mask=num_people == 1,
            dual_mask=num_people == 2,
        )
