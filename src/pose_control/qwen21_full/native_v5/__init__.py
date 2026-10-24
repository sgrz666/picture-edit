"""Native Full V5. Import submodules explicitly; never instantiate legacy cores."""

__all__=['NativeFullRuntimeV5','NativeConfigV5','NativeControlOptionsV5','PreparedNativeFullControlV5']


def __getattr__(name):
    if name=='NativeFullRuntimeV5':
        from .runtime import NativeFullRuntimeV5
        return NativeFullRuntimeV5
    if name in __all__:
        from . import contracts
        return getattr(contracts,name)
    raise AttributeError(name)
