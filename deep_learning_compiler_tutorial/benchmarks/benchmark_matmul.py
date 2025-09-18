import time
import json
import torch
from kernels.matmul_stepwise import launch_matmul_v1
from kernels.matmul_autotune import matmul as autotune_matmul

def bench(fn, a, b, iters=20, warmup=5):
    for _ in range(warmup):
        _ = fn(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(a, b)
    torch.cuda.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) * 1000 / iters
    return avg_ms

if __name__ == '__main__':
    results = []
    shapes = [(512,512,512), (1024,1024,1024), (2048,2048,2048)]
    for M,N,K in shapes:
        a = torch.randn((M, K), device='cuda', dtype=torch.float16)
        b = torch.randn((K, N), device='cuda', dtype=torch.float16)
        gflops = 2*M*N*K/1e9
        t_v1 = bench(launch_matmul_v1, a, b)
        t_auto = bench(autotune_matmul, a, b)
        res = {
            'shape': (M,N,K),
            'gflops_theoretical': gflops,
            'v1_ms': t_v1,
            'v_auto_ms': t_auto,
            'v1_eff_gflops': gflops / (t_v1/1000),
            'v_auto_eff_gflops': gflops / (t_auto/1000)
        }
        results.append(res)
        print(res)
    with open('matmul_results.json','w') as f:
        json.dump(results, f, indent=2)
