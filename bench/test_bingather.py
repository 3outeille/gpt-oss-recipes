import torch
import lovely_tensors as lt; lt.monkey_patch()
import pytest
import os

if os.environ.get("USE_REF", "0") == "1":
    print("Using reference Megablocks implementation")
    from ops_megablock_ref import binned_gather_triton
else:
    print("Using CUSTOM Megablocks implementation")
    from ops_megablock import binned_gather_triton

def set_seeds(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Stress test expert_capacity, especially near and at the upper limit (e.g., 65535 for int16 indexing)
def make_stress_expert_capacity_tests():
    tests = []
    # Small cases for sanity
    # for seq_len, hidden_size, num_experts, top_k in [
    #     (4, 2, 2, 1),
    #     (4, 2, 2, 2),
    #     (4, 2, 2, 4),
    # ]:
    #     for expert_capacity in [1, 2, 4]:
    #         tests.append((seq_len, hidden_size, num_experts, top_k, expert_capacity))
    # Medium cases
    # for seq_len, hidden_size, num_experts, top_k in [
        # (1024, 1536, 4, 1),
        # (1024, 1536, 4, 2),
        # (1024, 1536, 4, 4),
        # (1024, 1536, 64, 1),
        # (1024, 1536, 64, 2),
        # (1024, 1536, 64, 4),
        # (1024, 1536, 128, 1),
        # (1024, 1536, 128, 2),
        # (1024, 1536, 128, 4),
    # ]:
        # for expert_capacity in [1, 2, 4, 128, 1024]:
        #     tests.append((seq_len, hidden_size, num_experts, top_k, expert_capacity))
    # Large cases, stress expert_capacity near 65536 (CUDA second dim grid limit)
    for seq_len, hidden_size, num_experts, top_k in [
        # (16384, 768, 4, 1),
        # (16384, 768, 4, 2),
        # (16384, 768, 4, 4),
        # (16384, 768, 64, 1),
        # (16384, 768, 64, 2),
        # (16384, 768, 64, 4),
        # (16384, 768, 128, 1),
        # (16384, 768, 128, 2),
        (4096, 768, 32, 4),
    ]:
        for expert_capacity in [65535, 70000]:
            tests.append((seq_len, hidden_size, num_experts, top_k, expert_capacity))

    return tuple(tests)

BINNED_GATHER_TESTS = make_stress_expert_capacity_tests()

@pytest.mark.parametrize(('seq_len', 'hidden_size', 'num_experts', 'top_k', 'expert_capacity'), BINNED_GATHER_TESTS)
def test_binned_gather(seq_len: int, hidden_size: int, num_experts: int, top_k: int, expert_capacity: int):
    # NOTE: Capacity factor == 1.
    set_seeds(42)
    # Create the data and indices with gradient tracking
    x = torch.arange(seq_len * hidden_size, device='cuda', dtype=torch.half).view(seq_len, hidden_size)
    x.requires_grad_(True)

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, num_experts, (seq_len * top_k,), device='cuda', dtype=torch.int)
    _, indices = torch.sort(top_expert)
    bins = torch.cumsum(torch.bincount(top_expert, minlength=num_experts), 0).to(torch.int32)
    # Example: counts is [12, 2, 3], the bins tensor will be [12, 14, 20]. This tells the gather function:
    # Expert 0's assignments are in indices[0:12].
    # Expert 1's assignments are in indices[12:14].
    # Expert 2's assignments are in indices[14:20]. (we have num_tokens * 3)

    def binned_gather_pytorch(
        x: torch.Tensor,
        indices: torch.Tensor,
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
                print(f"Expert {i}: indices[{start}:{start + num_tokens}] = {indices[start : start + num_tokens]} -> tokens {idx}")
                out[i, :num_tokens, :] = x[idx, :]
            start = end
        return out

    out = binned_gather_triton(x, indices, bins, expert_capacity, top_k)
    expected_out = binned_gather_pytorch(x, indices, bins, expert_capacity, top_k)
    assert torch.all(torch.eq(out, expected_out))

    # Test backward pass
    grad_output = torch.arange(out.numel(), device=out.device, dtype=out.dtype).view_as(out)
    out.backward(grad_output)

    # Verify gradients were computed
    assert x.grad is not None, "Gradients should be computed for input x"
    assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} should match input shape {x.shape}"
    
    # Reference implementation for backward pass (binned_scatter)
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
            for j in range(num_tokens):
                index = indices[start + j]
                scale = weights[index] if weights is not None else 1.0
                token_pos = index // top_k
                
                out[token_pos, :] += scale * x[i, j, :]
            start = end
        return out
    
    expected_grad = binned_scatter_pytorch(grad_output, indices, None, bins, top_k)
    print(f"x.grad: {x.grad}")
    print(f"expected_grad: {expected_grad}")
    
    # Use torch.allclose instead of exact equality for floating point comparison
    if torch.allclose(x.grad, expected_grad, rtol=1e-3, atol=1e-3):
        print("✅ Success: Gradients match!")
    else:
        print("❌ Gradients don't match")
        # Let's see if it's just a reordering issue
        print("Checking if values match when sorted...")
        grad_sorted = torch.sort(x.grad.flatten())[0]
        expected_sorted = torch.sort(expected_grad.flatten())[0]
        if torch.allclose(grad_sorted, expected_sorted, rtol=1e-3, atol=1e-3):
            print("✅ Same values, different order - routing issue!")
        else:
            print("❌ Different values entirely")
    
    print(f"\nTriton Output Shape: {x.grad.shape}")
    print(f"PyTorch Output Shape: {expected_grad.shape}")


# if __name__ == "__main__":
#     #TODO:
#     # Understand minimal example with torch gather/scatter by hand (hardcoded)
    
#     seq_len = 4
#     hidden_size = 2
#     num_experts = 2
#     top_k = 2
#     expert_capacity = 3

#     test_binned_gather(seq_len, hidden_size, num_experts, top_k, expert_capacity)