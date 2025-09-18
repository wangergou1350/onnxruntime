import torch
import triton
import triton.language as tl

# v0: naive matmul (no tiling) - educational, not performant
@triton.jit
def matmul_v0(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m = pid_m
    n = pid_n
    # each program computes one element (inefficient)
    acc = 0.0
    for k in range(0, K):
        a = tl.load(a_ptr + m * stride_am + k * stride_ak, mask=m < M)
        b = tl.load(b_ptr + k * stride_bk + n * stride_bn, mask=n < N)
        acc += a * b
    tl.store(c_ptr + m * stride_cm + n * stride_cn, acc, mask=(m < M) & (n < N))

# v1: blocked matmul
@triton.jit
def matmul_v1(a_ptr, b_ptr, c_ptr, M, N, K,
              stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = k0 + offs_k < K
        a = tl.load(a_ptr + (offs_m[:, None] * stride_am + (k0 + offs_k)[None, :] * stride_ak), mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptr + ((k0 + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn), mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn), acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# convenience python wrappers

def launch_matmul_v0(a: torch.Tensor, b: torch.Tensor):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (M, N)
    matmul_v0[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c

def launch_matmul_v1(a: torch.Tensor, b: torch.Tensor, BLOCK_M=64, BLOCK_N=64, BLOCK_K=32):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_v1[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c

if __name__ == '__main__':
    torch.manual_seed(0)
    M = N = K = 512
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    # v0 warmup
    c0 = launch_matmul_v0(a, b)
    torch.cuda.synchronize()
    # v1 warmup
    c1 = launch_matmul_v1(a, b)
    torch.cuda.synchronize()
    # correctness (allow some fp16 diff)
    ref = a @ b
    max_diff_v0 = (c0 - ref).abs().max().item()
    max_diff_v1 = (c1 - ref).abs().max().item()
    print(f"max diff v0={max_diff_v0:.4f} v1={max_diff_v1:.4f}")
