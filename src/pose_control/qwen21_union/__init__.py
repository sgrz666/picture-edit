"""Qwen 2.1 official Union adaptation. Heavy model imports remain lazy."""


def __getattr__(name):
    if name in {"build_pipeline", "build_transformer", "ReferenceUnionPipeline", "ScopedUnionTransformer",
                "model_asset_signature", "encoder_asset_signature"}:
        from . import pipeline
        return getattr(pipeline, name)
    if name in {"UnionConditionCache", "UnionSupervisionCache", "precompute_conditions", "precompute_supervision",
                "load_condition", "load_supervision", "cache_model_signature", "condition_fingerprint",
                "supervision_fingerprint", "expected_condition_fingerprint", "expected_supervision_fingerprint"}:
        from . import cache
        return getattr(cache, name)
    raise AttributeError(name)
