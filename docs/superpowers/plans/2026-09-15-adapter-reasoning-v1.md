# Adapter Internal Reasoning V1 implementation record

Implemented on top of `main@c171fb4` as DeepGen V6.2.

The accepted design introduces `SMPLXAdapterReasoner(ConditionBundle) ->
InternalControlState`, a shared 512-wide person geometry path, gathered
dual-person cross-attention, gated raster/relation contact reasoning, and
independent geometry/interaction outputs at condition and half-condition
resolution. A thin compatibility bridge preserves the existing DeepGen
control core, six Zero Heads, residual alignment, appearance condition, and
public `forward(...)` signature.

Implementation is structural and forward-only. Parameter counts are reported
without an upper bound; no training or external weights are included.
