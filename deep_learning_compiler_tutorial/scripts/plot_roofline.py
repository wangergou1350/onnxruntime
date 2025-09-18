import json, argparse, matplotlib.pyplot as plt, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results', default='matmul_results.json')
parser.add_argument('--bandwidth', type=float, default=900)  # GB/s theoretical
parser.add_argument('--peak-tflops', type=float, default=125)
parser.add_argument('--bytes-per-element', type=int, default=2)
parser.add_argument('--output', default='reports/roofline.png')
args = parser.parse_args()

data = json.load(open(args.results))
xs_v1, ys_v1, xs_auto, ys_auto = [], [], [], []
labels = []
for row in data:
    M,N,K = row['shape']
    gflops = row['gflops_theoretical']
    bytes_moved = (M*K + K*N + M*N) * args.bytes_per_element * 3
    flops = gflops * 1e9
    for tag in ['v1','v_auto']:
        ms = row[f'{tag}_ms']
        sec = ms / 1000
        achieved = flops / sec / 1e12
        AI = flops / bytes_moved
        if tag == 'v1':
            xs_v1.append(AI); ys_v1.append(achieved)
        else:
            xs_auto.append(AI); ys_auto.append(achieved)
    labels.append(f"{M}")

ai_line = np.logspace(-2, 3, 400)
compute_line = np.full_like(ai_line, args.peak_tflops)
bw_line = (args.bandwidth * ai_line) / 1000
roof = np.minimum(compute_line, bw_line)

plt.figure(figsize=(7,5))
plt.loglog(ai_line, roof, label='Roofline', color='black')
plt.loglog(ai_line, bw_line, '--', label='Bandwidth bound')
plt.axhline(args.peak_tflops, linestyle='--', color='red', label='Compute peak')
plt.scatter(xs_v1, ys_v1, c='blue', marker='o', label='v1')
plt.scatter(xs_auto, ys_auto, c='green', marker='^', label='autotune')
for x1,y1,x2,y2,l in zip(xs_v1,ys_v1,xs_auto,ys_auto,labels):
    if y2 > y1:
        plt.annotate('', xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color='gray'))
    plt.text(x2*1.05, y2*1.05, l, fontsize=8)
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel('Performance (TFLOPs)')
plt.title('MatMul Roofline v1 vs Autotune')
plt.legend()
plt.grid(True, which='both', ls=':')
plt.tight_layout()
plt.savefig(args.output)
print('Saved', args.output)
