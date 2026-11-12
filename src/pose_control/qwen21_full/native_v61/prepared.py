"""V6.1 adds spatial records without changing the strict V6 record."""
from dataclasses import dataclass, replace

from ..native_v6.prepared import PreparedNativeFullControlV6
from .contracts import PoseSpatialConditionV61


@dataclass(frozen=True)
class PreparedNativeFullControlV61(PreparedNativeFullControlV6):
    spatial: PoseSpatialConditionV61

    def validate(self):
        super().validate()
        self.spatial.validate(self.pose)
        return self

    def to(self, *args, **kwargs):
        return replace(super().to(*args, **kwargs), spatial=self.spatial.to(*args, **kwargs))

    def index_select(self, indices):
        return replace(super().index_select(indices), spatial=self.spatial.index_select(indices))

    @classmethod
    def stack(cls, records):
        if not records or any(type(r) is not cls for r in records):
            raise ValueError('Cannot stack empty/mixed V6.1 controls')
        old = PreparedNativeFullControlV6.stack([PreparedNativeFullControlV6(
            r.core, r.relation, r.relation_valid, r.local, r.metadata, r.pose) for r in records])
        return cls(old.core, old.relation, old.relation_valid, old.local, old.metadata, old.pose,
                   PoseSpatialConditionV61.stack([r.spatial for r in records])).validate()
