import torch
from typing import Any
import triton
import triton.language as tl

def assert_is_matrix(x):
    if x.ndim != 2:
        raise ValueError(f'Expected 2-tensor but got {x.ndim}-tensor')

def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')

def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)

def assert_is_tensor(x, ndim):
    assert isinstance(x, torch.Tensor), f"Expected a tensor, got {type(x)}"
    assert x.ndim == ndim, f"Expected a tensor with {ndim} dimensions, got {x.ndim}"


def assert_equal(a, b):
    assert a == b, f"Expected {a} to equal {b}"


# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['NUM_COLUMNS'],
)
@triton.jit
def _binned_copy(
    a,
    b,
    num_experts,
    expert_capacity,
    indices,
    weights,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(1)
    entry_idx = tl.program_id(0)

    # Calculate our offset into the output.
    index_b = entry_idx * num_experts + expert_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    #
    # NOTE: We need to zero the output in both directions.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X

def binned_gather(x, indices, weights, bins, expert_capacity, top_k):
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert bins.shape[0] > 0, "bins must not be empty"
    assert_equal(indices.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)

    _binned_copy[(expert_capacity, num_experts)](
        x,
        out,
        num_experts,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


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

# a: (tokens, hidden_size), real.
# b: (num_experts, expert_capacity, num_columns), real.
# indices: (tokens * top_k), integer.
# weights: (tokens * top_k), real.
# bins: (num_experts), integer.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=2),
        triton.Config({'BLOCK_X': 256}, num_warps=2),
        triton.Config({'BLOCK_X': 128}, num_warps=4),
        triton.Config({'BLOCK_X': 256}, num_warps=4),
    ],
    key=['NUM_COLUMNS'],
)
@triton.jit
def _binned_copy_wgrad(
    x,
    grad,
    wgrad,
    num_experts,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_x = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)

def binned_scatter_wgrad(x, grad, indices, bins, top_k):
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens * top_k), dtype=x.dtype, device=x.device)
    _binned_copy_wgrad[(num_experts, expert_capacity)](
        x,
        grad,
        out,
        num_experts,
        expert_capacity,
        indices,
        bins,
        NUM_COLUMNS=hidden_size,
        TOP_K=top_k,
    )
    return out

class BinnedGatherOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        bin_size: int,
        top_k: int,
    ):
        ctx.save_for_backward(indices, bins)
        ctx.top_k = top_k
        return binned_gather(x, indices, None, bins, bin_size, top_k)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        # import pydevd; pydevd.settrace(suspend=False, trace_only_current_thread=True)
        grad = grad.contiguous()
        indices, bins = ctx.saved_tensors
        out = binned_scatter(grad, indices, None, bins, ctx.top_k)
        return out, None, None, None, None


class BinnedScatterOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.top_k = top_k

        # TODO(tgale): Don't save 'x' for backwards if we don't need to
        # calculate the gradient w.r.t. 'weights'.
        ctx.save_for_backward(x, indices, weights, bins)
        return binned_scatter(x, indices, weights, bins, top_k)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor):
        grad = grad.contiguous()
        x, indices, weights, bins = ctx.saved_tensors
        out = binned_gather(
            grad,
            indices,
            weights,
            bins,
            ctx.bin_size,
            ctx.top_k,
        )

        wgrad = None
        if ctx.needs_input_grad[2]:
            wgrad = binned_scatter_wgrad(
                x,
                grad,
                indices,
                bins,
                ctx.top_k,
            )
        return out, None, wgrad, None, None


binned_gather_triton = BinnedGatherOp.apply
binned_scatter_triton = BinnedScatterOp.apply