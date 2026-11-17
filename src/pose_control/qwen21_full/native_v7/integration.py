"""Install V7 through the existing V6.2 target-only implementation."""
from ..native_v62.integration import install_native_full_v62,bind_native_full_v62
from .runtime import NativeFullRuntimeV7,ARCHITECTURE


def install_native_full_v7(model,runtime):
    if not isinstance(runtime,NativeFullRuntimeV7):raise ValueError('V7 runtime required')
    report=install_native_full_v62(model,runtime)
    model._native_v7_architecture=ARCHITECTURE
    return {**report,'architecture':ARCHITECTURE}


def bind_native_full_v7(model,control,**kwargs):
    if getattr(model,'_native_v7_architecture',None)!=ARCHITECTURE:raise ValueError('Installed V7 model required')
    return bind_native_full_v62(model,control,**kwargs)
