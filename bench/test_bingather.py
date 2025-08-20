import torch
import lovely_tensors as lt; lt.monkey_patch()
import pytest
from ops_megablock import binned_gather_triton

# Stress test expert_capacity, especially near and at the upper limit (e.g., 65535 for int16 indexing)
def make_stress_expert_capacity_tests():
    tests = []
    # Small cases for sanity
    for seq_len, hidden_size, num_experts, top_k in [
        (4, 2, 2, 1),
        (4, 2, 2, 2),
        (4, 2, 2, 4),
    ]:
        for expert_capacity in [1, 2, 4]:
            tests.append((seq_len, hidden_size, num_experts, top_k, expert_capacity))
    # Medium cases
    for seq_len, hidden_size, num_experts, top_k in [
        (1024, 1536, 4, 1),
        (1024, 1536, 4, 2),
        (1024, 1536, 4, 4),
        (1024, 1536, 64, 1),
        (1024, 1536, 64, 2),
        (1024, 1536, 64, 4),
        (1024, 1536, 128, 1),
        (1024, 1536, 128, 2),
        (1024, 1536, 128, 4),
    ]:
        for expert_capacity in [1, 2, 4, 128, 1024]:
            tests.append((seq_len, hidden_size, num_experts, top_k, expert_capacity))
    # Large cases, stress expert_capacity near 65536 (CUDA second dim grid limit)
    for seq_len, hidden_size, num_experts, top_k in [
        (16384, 768, 4, 1),
        (16384, 768, 4, 2),
        (16384, 768, 4, 4),
        (16384, 768, 64, 1),
        (16384, 768, 64, 2),
        (16384, 768, 64, 4),
        (16384, 768, 128, 1),
        (16384, 768, 128, 2),
        (16384, 768, 128, 4),
    ]:
        for expert_capacity in [65535, 70000]:
            tests.append((seq_len, hidden_size, num_experts, top_k, expert_capacity))
    return tuple(tests)

BINNED_GATHER_TESTS = make_stress_expert_capacity_tests()
@pytest.mark.gpu
@pytest.mark.parametrize(('seq_len', 'hidden_size', 'num_experts', 'top_k', 'expert_capacity'), BINNED_GATHER_TESTS)
def test_binned_gather(seq_len: int, hidden_size: int, num_experts: int, top_k: int, expert_capacity: int):
    # NOTE: Capacity factor == 1.
    # Create the data and indices with gradient tracking
    x = torch.randn((seq_len, hidden_size), device='cuda', dtype=torch.half, requires_grad=True)

    # Randomly assign tokens to experts.
    top_expert = torch.randint(0, num_experts, (seq_len * top_k,), device='cuda', dtype=torch.int)
    _, indices = torch.sort(top_expert)
    bins = torch.cumsum(torch.bincount(top_expert, minlength=num_experts), 0).to(torch.int32)

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
                out[i, :num_tokens, :] = x[idx, :]
            start = end
        return out

    out = binned_gather_triton(x, indices, bins, expert_capacity, top_k)
    expected_out = binned_gather_pytorch(x, indices, bins, expert_capacity, top_k)
    assert torch.all(torch.eq(out, expected_out))

    # Test backward pass
    grad_output = torch.randn_like(out)
    out.backward(grad_output)
    
    # # Verify gradients were computed
    # assert x.grad is not None, "Gradients should be computed for input x"
    # assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} should match input shape {x.shape}"
    
    # # Reference implementation for backward pass (binned_scatter)
    # def binned_scatter_pytorch(grad_output, indices, bins, top_k):
    #     start = 0
    #     grad_input = torch.zeros((seq_len, hidden_size), dtype=grad_output.dtype, device=grad_output.device)
        
    #     for i in range(num_experts):
    #         end = bins[i]
    #         num_tokens = min(expert_capacity, end - start)
    #         if num_tokens > 0:
    #             expert_indices = indices[start : start + num_tokens]
    #             input_indices = expert_indices // top_k
    #             grad_input.index_add_(0, input_indices, grad_output[i, :num_tokens, :])
    #         start = end
        
    #     return grad_input
    
    # expected_grad = binned_scatter_pytorch(grad_output, indices, bins, top_k)
    # torch.testing.assert_close(x.grad, expected_grad, rtol=1e-3, atol=1e-3)