"""V6.2 prepared records keep V6.1 streams and add explicit routing/gates."""
from dataclasses import dataclass,replace

from ..native_v61.prepared import PreparedNativeFullControlV61
from .contracts import RoutingConditionV62,BranchSpatialGatesV62


@dataclass(frozen=True)
class PreparedNativeFullControlV62(PreparedNativeFullControlV61):
    routing: RoutingConditionV62
    gates: BranchSpatialGatesV62

    def validate(self):
        super().validate();self.routing.validate(self.pose);self.gates.validate(self.pose)
        return self

    def to(self,*args,**kwargs):
        return replace(super().to(*args,**kwargs),routing=self.routing.to(*args,**kwargs),gates=self.gates.to(*args,**kwargs))

    def index_select(self,indices):
        return replace(super().index_select(indices),routing=self.routing.index_select(indices),gates=self.gates.index_select(indices))

    @classmethod
    def stack(cls,records):
        if not records or any(type(r) is not cls for r in records):raise ValueError('Cannot stack empty/mixed V6.2 conditions')
        previous=PreparedNativeFullControlV61.stack([PreparedNativeFullControlV61(r.core,r.relation,r.relation_valid,
            r.local,r.metadata,r.pose,r.spatial) for r in records])
        return cls(previous.core,previous.relation,previous.relation_valid,previous.local,previous.metadata,
            previous.pose,previous.spatial,RoutingConditionV62.stack([r.routing for r in records]),
            BranchSpatialGatesV62.stack([r.gates for r in records])).validate()
