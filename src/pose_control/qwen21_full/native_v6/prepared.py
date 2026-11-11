"""V6 conditions retain the exact V5 streams plus pure PoseLink geometry."""
from dataclasses import dataclass, replace

from ..native_v5.contracts import PreparedNativeFullControlV5
from .contracts import PoseGeometry


@dataclass(frozen=True)
class PreparedNativeFullControlV6(PreparedNativeFullControlV5):
    pose: PoseGeometry

    def validate(self):
        super().validate()
        self.pose.validate()
        if self.pose.geometry.shape != self.body.geometry.shape or self.pose.target_token_hw != self.body.target_token_hw:
            raise ValueError('V6 PoseLink and Body grids differ')
        if self.pose.geometry.device != self.body.geometry.device:
            raise ValueError('V6 streams must share a device')
        if not (self.pose.person_valid == self.body.person_valid).all():
            raise ValueError('V6 PoseLink and Body person slots differ')
        return self

    def to(self, *args, **kwargs):
        return replace(super().to(*args, **kwargs), pose=self.pose.to(*args, **kwargs))

    def index_select(self, indices):
        return replace(super().index_select(indices), pose=self.pose.index_select(indices))

    @classmethod
    def stack(cls, records):
        if not records or any(type(r) is not cls for r in records):
            raise ValueError('Cannot stack empty/mixed V6 conditions')
        old = PreparedNativeFullControlV5.stack([
            PreparedNativeFullControlV5(r.core, r.relation, r.relation_valid, r.local, r.metadata) for r in records])
        return cls(old.core, old.relation, old.relation_valid, old.local, old.metadata,
                   PoseGeometry.stack([r.pose for r in records])).validate()
