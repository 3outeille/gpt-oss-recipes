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


def binned_gather_pytorch(x, indices, weights, bins, expert_capacity, top_k):
    """
    Baseline implementation of binned_gather using pure PyTorch.
    This is used to verify the correctness of the Triton kernel.
    """
    num_experts = bins.shape[0]
    hidden_size = x.shape[1]

    # Create the output tensor, same as in the Triton version.
    out = torch.zeros(
        (num_experts, expert_capacity, hidden_size), dtype=x.dtype, device=x.device
    )

    # The `bins` tensor is a cumulative sum of token counts for each expert.
    # We can find the tokens for each expert by looking at the differences.
    start_idx = 0
    for i in range(num_experts):
        # Find the end index for the current expert's tokens.
        end_idx = bins[i].item()
        num_tokens_for_expert = end_idx - start_idx

        # In MoE, tokens that overflow the expert's capacity are dropped.
        # We must model this in the baseline for it to be correct.
        tokens_to_copy = min(num_tokens_for_expert, expert_capacity)

        if tokens_to_copy == 0:
            start_idx = end_idx
            continue

        # Get the slice of `indices` that belong to this expert, truncated by capacity.
        expert_indices_slice = indices[start_idx : start_idx + tokens_to_copy]

        # The values in the slice are the final token indices, but they may have
        # been replicated if top_k > 1. We divide by top_k to get the
        # original row index in the input tensor `x`.
        original_token_indices = expert_indices_slice // top_k

        # Gather the data from `x` using the calculated indices.
        data_to_copy = x[original_token_indices]

        # If weights are provided, apply them.
        if weights is not None:
            # Slice weights corresponding to the tokens we are actually copying.
            expert_weights = weights[start_idx : start_idx + tokens_to_copy]
            # Reshape weights to be broadcastable for element-wise multiplication.
            data_to_copy = data_to_copy * expert_weights.unsqueeze(1)

        # Place the gathered and weighted data into the output tensor.
        out[i, :tokens_to_copy] = data_to_copy

        # Update the start index for the next expert.
        start_idx = end_idx

    return out

@pytest.mark.parametrize("num_tokens", [16, 128, 1024, 4096])
@pytest.mark.parametrize("hidden_size", [8, 64, 512, 2880])
@pytest.mark.parametrize("num_experts", [2, 8, 32])
@pytest.mark.parametrize("expert_capacity", [4, 16, 128, 1024, 65536])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test(num_tokens, hidden_size, num_experts, expert_capacity, top_k):

    # Some tests will fail with OOM but that's because allocated tensors are too large.
    # This is not a problem with the kernel in itself.
    x = torch.randn(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device='cuda')

    indices = torch.arange(num_tokens * top_k, dtype=torch.int32, device='cuda')

    counts = torch.randint(0, expert_capacity, (num_experts,), device='cuda')
    # Ensure the total number of tokens assigned does not exceed the input tokens.
    counts = (counts.float() / counts.sum() * (num_tokens * top_k * 0.9)).to(torch.int32)
    diff = (num_tokens * top_k) - counts.sum()
    counts[0] += diff
    bins = torch.cumsum(counts, dim=0, dtype=torch.int32).cuda()

    print(f"\nGridsearch: num_tokens={num_tokens}, hidden_size={hidden_size}, num_experts={num_experts}, expert_capacity={expert_capacity}, top_k={top_k}")
    print("Created tensors with shapes:")
    print(f"x: {x.shape}, dtype: {x.dtype}")
    print(f"indices: {indices.shape}, dtype: {indices.dtype}")
    print(f"bins: {bins.shape}, dtype: {bins.dtype}")
    print(f"expert_capacity: {expert_capacity}")

    output_pytorch = binned_gather_pytorch(x, indices, None, bins, expert_capacity, top_k)
    output_triton = binned_gather(x, indices, None, bins, expert_capacity, top_k)

    print("\nComparing Triton kernel output to PyTorch baseline...")
    assert torch.allclose(output_triton, output_pytorch, atol=1e-2, rtol=1e-2), (
        "❌ Failure: Outputs do not match. "
        f"Difference: {torch.abs(output_triton - output_pytorch).max()}"
    )

    print("✅ Success: Outputs match!")
    print(f"\nTriton Output Shape: {output_triton.shape}")
    print(f"PyTorch Output Shape: {output_pytorch.shape}")


# if __name__ == "__main__":
#     test()