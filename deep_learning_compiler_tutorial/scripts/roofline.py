import json, math, argparse

def compute_ai(gflops, ms, bytes_moved):
    flops = gflops * 1e9
    seconds = ms / 1000
    return flops / bytes_moved, flops / seconds / 1e12  # (AI, achieved TFLOPs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='matmul_results.json')
    parser.add_argument('--bytes-per-element', type=int, default=2)  # fp16
    parser.add_argument('--bandwidth', type=float, default=900)  # GB/s theoretical example
    parser.add_argument('--peak-tflops', type=float, default=125)  # example GPU
    args = parser.parse_args()
    data = json.load(open(args.results))
    print('shape,AI,achieved_TFLOPs,bw_bound_TFLOPs,roofline_TFLOPs')
    for row in data:
        M,N,K = row['shape']
        gflops = row['gflops_theoretical']
        # naive bytes estimation (A+B+C read/write once)
        bytes_moved = (M*K + K*N + M*N) * args.bytes_per_element * 3
        AI, achieved = compute_ai(gflops, row['v1_ms'], bytes_moved)
        bw_bound = (args.bandwidth * 1e9) * AI / 1e12
        roof = min(bw_bound, args.peak_tflops)
        print(f"{M}x{N}x{K},{AI:.2f},{achieved:.2f},{bw_bound:.2f},{roof:.2f}")
