import json, time, math, argparse, os
import torch
from kernels.matmul_stepwise import launch_matmul_v1
from kernels.softmax_layernorm import softmax as triton_softmax, layernorm as triton_layernorm
from kernels.matmul_autotune import matmul as autotune_matmul


def time_fn(fn, *args, iters=20, warmup=5):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize()
    end = time.perf_counter()
    return out, (end - start) * 1000 / iters


def benchmark():
    results = {}
    device = 'cuda'
    torch.manual_seed(0)
    # MatMul
    M = N = K = 1024
    a = torch.randn((M, K), device=device, dtype=torch.float16)
    b = torch.randn((K, N), device=device, dtype=torch.float16)
    ref = a @ b
    out_v1, t_v1 = time_fn(launch_matmul_v1, a, b)
    out_auto, t_auto = time_fn(autotune_matmul, a, b)
    max_diff_v1 = (out_v1 - ref).abs().max().item()
    max_diff_auto = (out_auto - ref).abs().max().item()
    gflops = 2*M*N*K/1e9
    results['matmul'] = {
        'shape': [M,N,K],
        'gflops': gflops,
        'v1_ms': t_v1,
        'auto_ms': t_auto,
        'v1_eff_gflops': gflops / (t_v1/1000),
        'auto_eff_gflops': gflops / (t_auto/1000),
        'v1_max_diff': max_diff_v1,
        'auto_max_diff': max_diff_auto
    }
    # Softmax
    x = torch.randn(512, 768, device=device, dtype=torch.float32)
    ref_sm = torch.softmax(x, dim=-1)
    out_sm, t_sm = time_fn(triton_softmax, x)
    results['softmax'] = {
        'shape':[512,768],
        'ms': t_sm,
        'max_diff': (out_sm - ref_sm).abs().max().item()
    }
    # LayerNorm
    ref_ln = torch.nn.functional.layer_norm(x, (x.shape[1],))
    out_ln, t_ln = time_fn(triton_layernorm, x)
    results['layernorm'] = {
        'shape':[512,768],
        'ms': t_ln,
        'max_diff': (out_ln - ref_ln).abs().max().item()
    }
    return results


def write_report(results):
    os.makedirs('reports', exist_ok=True)
    with open('reports/summary.md','w',encoding='utf-8') as f:
        f.write('# 性能与精度摘要\n\n')
        # MatMul
        mm = results['matmul']
        f.write('## MatMul ({}x{}x{})\n'.format(*mm['shape']))
        f.write('| 版本 | ms | Effective Gflops | Max Diff |\n')
        f.write('|------|----|------------------|----------|\n')
        f.write('| v1 | {:.3f} | {:.1f} | {:.3e} |\n'.format(mm['v1_ms'], mm['v1_eff_gflops'], mm['v1_max_diff']))
        f.write('| autotune | {:.3f} | {:.1f} | {:.3e} |\n'.format(mm['auto_ms'], mm['auto_eff_gflops'], mm['auto_max_diff']))
        # Softmax
        sm = results['softmax']
        f.write('\n## Softmax ({}) shape\n'.format('x'.join(map(str, sm['shape']))))
        f.write('- time: {:.3f} ms, max diff: {:.3e}\n'.format(sm['ms'], sm['max_diff']))
        # LayerNorm
        ln = results['layernorm']
        f.write('\n## LayerNorm ({}) shape\n'.format('x'.join(map(str, ln['shape']))))
        f.write('- time: {:.3f} ms, max diff: {:.3e}\n'.format(ln['ms'], ln['max_diff']))
        f.write('\n---\n生成时间可添加到 CI 以监控性能回归。\n')
    with open('reports/summary.json','w') as jf:
        json.dump(results, jf, indent=2)

if __name__ == '__main__':
    res = benchmark()
    write_report(res)
    print('Report written to reports/summary.md')
