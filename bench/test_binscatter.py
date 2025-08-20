# Copyright 2024 Databricks
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from test_bingather import binned_gather, _binned_copy


def assert_is_tensor(x, ndim):
    assert isinstance(x, torch.Tensor), f"Expected a tensor, got {type(x)}"
    assert x.ndim == ndim, f"Expected a tensor with {ndim} dimensions, got {x.ndim}"

def assert_is_vector(x):
    assert_is_tensor(x, 1)


def assert_equal(a, b):
    assert a == b, f"Expected {a} to equal {b}"


def binned_scatter(x, indices, weights, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)
    _binned_copy[(expert_capacity, num_experts)](
        out,
        x,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=hidden_size,
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, hidden_size)


_BINNED_SCATTER_TESTS = (
    (4, 2, 2, 1),
    (4, 2, 2, 2),
    (4, 2, 2, 4),
    (1024, 1536, 4, 1),
    (1024, 1536, 4, 2),
    (1024, 1536, 4, 4),
    (1024, 1536, 64, 1),
    (1024, 1536, 64, 2),
    (1024, 1536, 64, 4),
    (1024, 1536, 128, 1),
    (1024, 1536, 128, 2),
    (1024, 1536, 128, 4),
    (16384, 768, 4, 1),
    (16384, 768, 4, 2),
    (16384, 768, 4, 4),
    (16384, 768, 64, 1),
    (16384, 768, 64, 2),
    (16384, 768, 64, 4),
    (16384, 768, 128, 1),
    (16384, 768, 128, 2),
    (16384, 768, 128, 4),
)
@pytest.mark.gpu
@pytest.mark.parametrize(('seq_len', 'hidden_size', 'num_experts', 'top_k'), _BINNED_SCATTER_TESTS)
def testBinnedScatter(seq_len: int, hidden_size: int, num_experts: int, top_k: int):
    # NOTE: Capacity factor == 1.
    expert_capacity = (seq_len * top_k) // num_experts

    # Create the data and indices.
    x = torch.randn((seq_len, hidden_size), device='cuda', dtype=torch.half)

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, num_experts, (seq_len * top_k,), device='cuda', dtype=torch.int)
    _, indices = torch.sort(top_expert)
    bins = torch.cumsum(torch.bincount(top_expert, minlength=num_experts), 0).to(torch.int32)

    # Sample weights for the scatter reduce.
    weights = torch.rand((seq_len * top_k,), device='cuda', dtype=torch.half)

    x = binned_gather(x, indices, None, bins, expert_capacity, top_k)

    def binned_scatter_pytorch(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        # x: (ne, ec, hs)
        # indices: (sl * top_k,)
        # weights: (sl * top_k,)
        # bins: (ne,)
        # Output: (sl, hs)
        out = torch.zeros((seq_len, hidden_size), device=x.device, dtype=x.dtype)
        start = 0
        for i in range(num_experts):
            end = bins[i]
            num_tokens = min(expert_capacity, end - start)
            if num_tokens > 0:
                idx = indices[start : start + num_tokens]
                scale = weights[idx]
                idx_out = idx // top_k
                # x[i, :num_tokens, :] shape: (num_tokens, hs)
                # scale shape: (num_tokens,)
                # idx_out shape: (num_tokens,)
                # Use scatter_add to accumulate
                out.index_add_(0, idx_out, scale.unsqueeze(1) * x[i, :num_tokens, :])
            start = end
        return out

    out = binned_scatter(x, indices, weights, bins, top_k)
    expected_out = binned_scatter_pytorch(x, indices, weights, bins, top_k)

    # NOTE: We need to check approximate equality because the
    # scatter reduce uses atomics.
    torch.testing.assert_close(
        out.cpu(),
        expected_out.cpu(),
        rtol=5e-3,
        atol=1e-5,
    )
