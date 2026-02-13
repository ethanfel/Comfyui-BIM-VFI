"""
Sparse SageAttention — block-sparse INT8 attention via Triton.

https://github.com/jt-zhang/Sparse_SageAttention_API

Copyright (c) 2024 by SageAttention team.
Licensed under the Apache License, Version 2.0
"""

from .quant_per_block import per_block_int8
from .sparse_int8_attn import forward as sparse_sageattn_fwd
import torch


def sparse_sageattn(q, k, v, mask_id=None, is_causal=False, tensor_layout="HND"):
    if mask_id is None:
        mask_id = torch.ones(
            (q.shape[0], q.shape[1],
             (q.shape[2] + 128 - 1) // 128,
             (q.shape[3] + 64 - 1) // 64),
            dtype=torch.int8, device=q.device,
        )

    output_dtype = q.dtype
    if output_dtype == torch.bfloat16 or output_dtype == torch.float32:
        v = v.to(torch.float16)

    seq_dim = 1 if tensor_layout == "NHD" else 2
    km = k.mean(dim=seq_dim, keepdim=True)

    q_int8, q_scale, k_int8, k_scale = per_block_int8(
        q, k, km=km, tensor_layout=tensor_layout,
    )

    o = sparse_sageattn_fwd(
        q_int8, k_int8, mask_id, v, q_scale, k_scale,
        is_causal=is_causal, tensor_layout=tensor_layout,
        output_dtype=output_dtype,
    )
    return o
