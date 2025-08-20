import torch
import triton
import triton.language as tl
import lovely_tensors as lt; lt.monkey_patch()
import pytest

def assert_is_matrix(x):
    if x.ndim != 2:
        raise ValueError(f'Expected 2-tensor but got {x.ndim}-tensor')

def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f'Expected 1-tensor but got {x.ndim}-tensor')

def assert_equal(a, b):
    if a != b:
        raise ValueError(f'Expected dimensions to be equal but got {a} and {b}.',)

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
    # Cast index_b to 64-bit before multiplying to prevent overflow on large tensors.
    # Otherwise The multiplication index_b * NUM_COLUMNS is performed using 32-bit integers, and the result overflows
    b += tl.multiple_of(index_b.to(tl.int64) * NUM_COLUMNS, NUM_COLUMNS)
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
    out = torch.zeros((expert_capacity, num_experts, x.shape[1]), dtype=x.dtype, device=x.device)

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

    out = out.transpose(0, 1)
    return out


BINNED_GATHER_TESTS = (
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
@pytest.mark.parametrize(('seq_len', 'hidden_size', 'num_experts', 'top_k'), BINNED_GATHER_TESTS)
def test_binned_gather(seq_len: int, hidden_size: int, num_experts: int, top_k: int):
    # NOTE: Capacity factor == 1.
    expert_capacity = (seq_len * top_k) // num_experts

    # Create the data and indices.
    x = torch.randn((seq_len, hidden_size), device='cuda', dtype=torch.half)

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, num_experts, (seq_len * top_k,), device='cuda', dtype=torch.int)
    _, indices = torch.sort(top_expert)
    bins = torch.cumsum(torch.bincount(top_expert, minlength=num_experts), 0).to(torch.int32)

    def binned_gather_pytorch(
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        bins: torch.Tensor,
        expert_capacity: int,
        top_k: int,
    ):
        start = 0
        out = torch.zeros((num_experts, expert_capacity, hidden_size), dtype=x.dtype, device=x.device)
        for i in range(num_experts):
            end = bins[i]
            num_tokens = min(expert_capacity, end - start)
            if num_tokens > 0:
                # indices[start:end] are the indices for this expert
                # For each slot j, get the input index and copy the row
                idx = indices[start : start + num_tokens] // top_k
                out[i, :num_tokens, :] = x[idx, :]
            start = end
        return out

    out = binned_gather(x, indices, None, bins, expert_capacity, top_k)
    expected_out = binned_gather_pytorch(x, indices, None, bins, expert_capacity, top_k)
    assert torch.all(torch.eq(out, expected_out))