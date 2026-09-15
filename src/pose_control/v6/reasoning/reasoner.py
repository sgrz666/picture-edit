from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..conditions import ConditionBundle
from .blocks import ReasoningResBlock
from .config import AdapterReasoningConfig
from .contact import ContactConditionReasoner
from .cross_person import BidirectionalCrossPersonReasoner
from .fusion import DualFeatureFusion
from .person_geometry import PersonGeometryReasoner, PersonReasoningOutput
from .state import InternalControlState


class SMPLXAdapterReasoner(nn.Module):
    """Convert a ConditionBundle into batch-safe geometry/interaction state."""

    def __init__(self, config: AdapterReasoningConfig | None = None) -> None:
        super().__init__()
        self.config = config or AdapterReasoningConfig()
        hidden_dim = self.config.hidden_dim
        self.person_reasoner = PersonGeometryReasoner(self.config)
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.cross_person_reasoner = BidirectionalCrossPersonReasoner(self.config)
        self.contact_reasoner = ContactConditionReasoner(self.config)
        self.dual_fusion = DualFeatureFusion(self.config)
        self.single_geometry_block = ReasoningResBlock(hidden_dim)
        self.single_high_projection = nn.Conv2d(hidden_dim * 2, hidden_dim, 1)
        self.single_high_block = ReasoningResBlock(hidden_dim)

    @staticmethod
    def _role_bind(
        output: PersonReasoningOutput,
        role_embedding: torch.Tensor,
    ) -> PersonReasoningOutput:
        tokens = output.tokens + role_embedding[None, None].to(output.tokens.dtype)
        tokens = tokens * output.token_mask[..., None].to(tokens.dtype)
        return PersonReasoningOutput(output.local_feature, tokens, output.token_mask)

    @staticmethod
    def _token_map(tokens: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], *size)

    def _single_geometry(
        self,
        person: PersonReasoningOutput,
        high_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        low_size = (high_size[0] // 2, high_size[1] // 2)
        low_mask = person.token_mask.reshape(person.tokens.shape[0], 1, *low_size)
        geometry = self.single_geometry_block(
            self._token_map(person.tokens, low_size), low_mask
        )
        high_mask = F.interpolate(
            low_mask.to(geometry.dtype), size=high_size, mode="nearest"
        ) > 0
        geometry_up = F.interpolate(
            geometry, size=high_size, mode="bilinear", align_corners=False
        )
        geometry_high = self.single_high_projection(
            torch.cat((geometry_up, person.local_feature), dim=1)
        )
        geometry_high = self.single_high_block(geometry_high, high_mask)
        return geometry, geometry_high

    def _encode_person_a(self, bundle: ConditionBundle) -> PersonReasoningOutput:
        output = self.person_reasoner(
            bundle.person_a_spatial,
            bundle.person_a_mask,
            bundle.person_a_global_tokens,
            bundle.task_token,
        )
        return self._role_bind(output, self.role_embedding.weight[0])

    def _encode_person_b(self, bundle: ConditionBundle) -> PersonReasoningOutput:
        if (
            bundle.person_b_spatial is None
            or bundle.person_b_mask is None
            or bundle.person_b_global_tokens is None
        ):
            raise ValueError("dual-person rows require complete Person-B bundle fields")
        output = self.person_reasoner(
            bundle.person_b_spatial,
            bundle.person_b_mask,
            bundle.person_b_global_tokens,
            bundle.task_token,
        )
        return self._role_bind(output, self.role_embedding.weight[1])

    def forward(self, bundle: ConditionBundle) -> InternalControlState:
        bundle.validate()
        high_size = bundle.person_a_spatial.shape[-2:]
        if high_size[0] % 2 or high_size[1] % 2:
            raise ValueError("ConditionBundle height and width must be even")
        if bundle.person_a_mask.bool().flatten(1).sum(dim=1).eq(0).any():
            raise ValueError("valid person A has an empty mask")
        if bundle.person_b_mask is not None:
            invalid_b_mask = (
                bundle.person_b_mask.bool().flatten(1).sum(dim=1).eq(0)
                & bundle.person_valid[:, 1]
            )
            if invalid_b_mask.any():
                raise ValueError("valid person B has an empty mask")

        person_a = self._encode_person_a(bundle)
        geometry, geometry_high = self._single_geometry(person_a, high_size)
        batch_size, token_count, hidden_dim = person_a.tokens.shape
        interaction = geometry.new_zeros(geometry.shape)
        interaction_high = geometry_high.new_zeros(geometry_high.shape)
        person_tokens = person_a.tokens.new_zeros(batch_size, 2, token_count, hidden_dim)
        person_token_mask = torch.zeros(
            batch_size, 2, token_count, dtype=torch.bool, device=bundle.device
        )
        person_tokens[:, 0] = person_a.tokens
        person_token_mask[:, 0] = person_a.token_mask

        dual_indices = (bundle.person_count == 2).nonzero(as_tuple=False).flatten()
        if dual_indices.numel():
            dual_bundle = bundle.index_select(dual_indices)
            dual_a = PersonReasoningOutput(
                person_a.local_feature.index_select(0, dual_indices),
                person_a.tokens.index_select(0, dual_indices),
                person_a.token_mask.index_select(0, dual_indices),
            )
            dual_b = self._encode_person_b(dual_bundle)
            tokens_a, tokens_b = self.cross_person_reasoner(
                dual_a.tokens,
                dual_b.tokens,
                dual_a.token_mask,
                dual_b.token_mask,
                dual_bundle.relative_tokens,
            )
            contact = self.contact_reasoner(
                tokens_a,
                tokens_b,
                dual_a.token_mask,
                dual_b.token_mask,
                dual_bundle.contact_spatial,
                dual_bundle.contact_tokens,
                dual_bundle.contact_mask,
                high_size=high_size,
            )
            dual = self.dual_fusion(
                contact.person_a,
                contact.person_b,
                dual_a.token_mask,
                dual_b.token_mask,
                dual_a.local_feature,
                dual_b.local_feature,
                contact.spatial_low,
                contact.spatial_high,
            )
            geometry = geometry.index_copy(0, dual_indices, dual.geometry_feature)
            geometry_high = geometry_high.index_copy(
                0, dual_indices, dual.geometry_highres
            )
            interaction = interaction.index_copy(
                0, dual_indices, dual.interaction_feature
            )
            interaction_high = interaction_high.index_copy(
                0, dual_indices, dual.interaction_highres
            )
            person_tokens[:, 0] = person_tokens[:, 0].index_copy(
                0, dual_indices, contact.person_a
            )
            person_tokens[:, 1] = person_tokens[:, 1].index_copy(
                0, dual_indices, contact.person_b
            )
            person_token_mask[:, 1] = person_token_mask[:, 1].index_copy(
                0, dual_indices, dual_b.token_mask
            )

        interaction_valid = bundle.person_count == 2
        interaction = interaction * interaction_valid[:, None, None, None].to(
            interaction.dtype
        )
        interaction_high = interaction_high * interaction_valid[:, None, None, None].to(
            interaction_high.dtype
        )
        state = InternalControlState(
            geometry_feature=geometry,
            geometry_highres=geometry_high,
            interaction_feature=interaction,
            interaction_highres=interaction_high,
            person_tokens=person_tokens,
            person_token_mask=person_token_mask,
            person_valid=bundle.person_valid,
            person_count=bundle.person_count.long(),
            interaction_valid=interaction_valid,
        )
        return state.validate()
