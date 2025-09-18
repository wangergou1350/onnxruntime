import torch, json, os
import triton
import triton.language as tl

CACHE_FILE = 'configs/matmul_autotune_cache.json'

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE,'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE,'w') as f:
        json.dump(cache, f, indent=2)

SEARCH_CONFIGS = [
    {'BLOCK_M':64,'BLOCK_N':64,'BLOCK_K':32,'num_warps':4,'num_stages':2},
    {'BLOCK_M':128,'BLOCK_N':64,'BLOCK_K':32,'num_warps':4,'num_stages':2},
    {'BLOCK_M':64,'BLOCK_N':128,'BLOCK_K':64,'num_warps':8,'num_stages':3},
    {'BLOCK_M':128,'BLOCK_N':128,'BLOCK_K':32,'num_warps':8,'num_stages':3},
]

@triton.jit
@triton.jit
def matmul_autotune_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
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


def _run_kernel(a,b,c, M,N,K, cfg):
    grid = (triton.cdiv(M, cfg['BLOCK_M']), triton.cdiv(N, cfg['BLOCK_N']))
    matmul_autotune_kernel[grid](a,b,c, M,N,K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                                 BLOCK_M=cfg['BLOCK_M'], BLOCK_N=cfg['BLOCK_N'], BLOCK_K=cfg['BLOCK_K'])

def matmul(a: torch.Tensor, b: torch.Tensor):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # try load cached best config shape key: f"{M}x{N}x{K}:{str(a.dtype)}"
    key = f"{M}x{N}x{K}:{str(a.dtype)}"
    cache = _load_cache()
    if key in cache:
        cfg = cache[key]
        _run_kernel(a,b,c,M,N,K,cfg)
    else:
        # search all configs quickly (few iterations)
        torch.cuda.synchronize()
        import time
        best = None
        for cfg in SEARCH_CONFIGS:
            try:
                t0 = time.perf_counter()
                _run_kernel(a,b,c,M,N,K,cfg)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                ms = (t1 - t0)*1000
                if (best is None) or ms < best['ms']:
                    best = {**cfg, 'ms': ms}
            except triton.OutOfResources:
                continue
        chosen = {k:best[k] for k in ['BLOCK_M','BLOCK_N','BLOCK_K']}
        cache[key] = chosen
        _save_cache(cache)
        _run_kernel(a,b,c,M,N,K,chosen)
    return c

if __name__ == '__main__':
    torch.manual_seed(0)
    M = N = K = 1024
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    ref = a @ b
    print('max diff', (c - ref).abs().max().item())
