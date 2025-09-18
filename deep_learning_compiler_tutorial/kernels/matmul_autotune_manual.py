import json, time, itertools
import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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

SEARCH_SPACE = {
    'BLOCK_M': [64, 128],
    'BLOCK_N': [64, 128],
    'BLOCK_K': [32, 64]
}

CACHE_FILE = 'configs/matmul_autotune_cache.json'

def run_config(a, b, BLOCK_M, BLOCK_N, BLOCK_K):
    M,K = a.shape
    K2,N = b.shape
    assert K == K2
    c = torch.empty((M,N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c

def benchmark(a,b,config,iters=10,warmup=5):
    for _ in range(warmup):
        run_config(a,b,**config)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        run_config(a,b,**config)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) * 1000 / iters

def main():
    torch.manual_seed(0)
    M=N=K=1024
    a = torch.randn((M,K), device='cuda', dtype=torch.float16)
    b = torch.randn((K,N), device='cuda', dtype=torch.float16)
    best = None
    results = []
    for combo in itertools.product(SEARCH_SPACE['BLOCK_M'], SEARCH_SPACE['BLOCK_N'], SEARCH_SPACE['BLOCK_K']):
        config = {'BLOCK_M': combo[0], 'BLOCK_N': combo[1], 'BLOCK_K': combo[2]}
        try:
            ms = benchmark(a,b,config)
            gflops = 2*M*N*K/1e9 / (ms/1000)
            record = {**config, 'ms': ms, 'gflops': gflops}
            results.append(record)
            if best is None or ms < best['ms']:
                best = record
            print('config', record)
        except triton.OutOfResources:
            print('skip OOR', config)
    cache = {'shape':[M,N,K], 'best': best, 'all': results}
    with open(CACHE_FILE,'w') as f:
        json.dump(cache, f, indent=2)
    print('Best config:', best)

if __name__ == '__main__':
    main()
