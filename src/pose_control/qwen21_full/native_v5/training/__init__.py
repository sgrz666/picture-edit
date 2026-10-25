"""Training management only; importing this package never starts a run."""

__all__=['NativeV5Trainer','TrainingConfig']

def __getattr__(name):
    if name=='NativeV5Trainer':
        from .trainer import NativeV5Trainer
        return NativeV5Trainer
    if name=='TrainingConfig':
        from .config import TrainingConfig
        return TrainingConfig
    raise AttributeError(name)
