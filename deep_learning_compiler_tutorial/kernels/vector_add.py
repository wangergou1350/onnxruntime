import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

def vector_add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024):
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    n = a.numel()
    c = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](a, b, c, n, BLOCK_SIZE=block_size)
    return c

if __name__ == '__main__':
    for n in [1_000_000, 4_000_000, 16_000_000]:
        a = torch.randn(n, device='cuda', dtype=torch.float32)
        b = torch.randn(n, device='cuda', dtype=torch.float32)
        # warmup
        for _ in range(5):
            c = vector_add(a, b)
        torch.cuda.synchronize()
        import time
        iters = 20
        start = time.perf_counter()
        for _ in range(iters):
            c = vector_add(a, b)
        torch.cuda.synchronize()
        end = time.perf_counter()
        avg_ms = (end - start) * 1000 / iters
        bytes_moved = (a.element_size() * 3 * n)  # read a,b write c
        bandwidth = bytes_moved / (avg_ms / 1000) / 1e9
        print(f"n={n} avg={avg_ms:.3f}ms effective_bw={bandwidth:.2f} GB/s")
