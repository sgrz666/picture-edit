"""Shared native/adapter BF16 reduction policy, with reversible global state.

CUDA matmul flags are process-wide. Wrapping an entire experiment also covers
source encoding, teacher calls and training; nested generation contexts retain
the outer policy. No attention implementation is selected here.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def precision_policy():
    """Stable JSON metadata for the policy applied inside qwen21_precision()."""
    return dict(name='qwen21_bf16_full_precision_reduction_v1',
                allow_bf16_reduced_precision_reduction=False)


@contextmanager
def qwen21_precision():
    """Use full precision BF16 matmul reductions and restore the prior flag.

    Can be used either as ``with qwen21_precision():`` or as the reusable
    ``@qwen21_precision()`` decorator. Restoration also runs after exceptions.
    """
    previous = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield precision_policy()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = previous
