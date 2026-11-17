"""Independent V7 switches; native Qwen tokens and side-attention budgets stay fixed."""
from dataclasses import dataclass

from ..native_v62.contracts import PoseLinkConfigV62


@dataclass(frozen=True)
class PoseLinkConfigV7(PoseLinkConfigV62):
    mask_pose_fusion: bool = True
    spatial_rope: bool = True
    geometry_contact: bool = True
    identity_modulation: bool = True

    def __post_init__(self):
        super().__post_init__()
        for name in ('mask_pose_fusion','spatial_rope','geometry_contact','identity_modulation'):
            if type(getattr(self,name)) is not bool:
                raise ValueError(name+' requires bool')
        # Each head has independent X/Y spaces, both made of complete pairs.
        if self.spatial_rope and (self.width//self.heads)%4:
            raise ValueError('XY RoPE requires head dimension divisible by four for even axis pairs')
        routed_features=[name for name in ('mask_pose_fusion','spatial_rope','geometry_contact') if getattr(self,name)]
        if not self.person_routing and routed_features:
            raise ValueError('V7 pose features '+', '.join(routed_features)+' require person_routing')
