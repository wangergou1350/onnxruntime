import json, os, argparse, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path,'r') as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trend-file', default='reports/fp8_accuracy_trend.json')
    ap.add_argument('--p99-file', default='reports/p99_history.json')
    ap.add_argument('--kl-file', default='reports/kl_history.json')
    ap.add_argument('--out-prefix', default='reports/fp8_trend')
    args = ap.parse_args()

    trend = load_json(args.trend_file) or []
    p99_hist = load_json(args.p99_file) or []
    kl_hist = load_json(args.kl_file) or []

    if not trend and not p99_hist:
        print('No trend data found.')
        return

    os.makedirs('reports', exist_ok=True)

    # Plot p99 vs clip over time
    if trend:
        xs = list(range(len(trend)))
        p99_pc = [t.get('p99_abs_pc') for t in trend]
        clip_pc = [t.get('clip_pc') for t in trend]
        p99_grp = [t.get('p99_abs_group') for t in trend]
        clip_grp = [t.get('clip_group') for t in trend]
        gsize = [t.get('group_size') for t in trend]
        fig, ax1 = plt.subplots(figsize=(7,4))
        ax1.plot(xs, p99_pc, label='p99_abs_pc', color='blue')
        if any(p99_grp):
            ax1.plot(xs, p99_grp, label='p99_abs_group', color='purple')
        ax1.set_ylabel('p99 abs diff')
        ax1.set_xlabel('build index')
        ax2 = ax1.twinx()
        ax2.plot(xs, clip_pc, label='clip_pc', color='red', linestyle='--')
        if any(clip_grp):
            ax2.plot(xs, clip_grp, label='clip_group', color='orange', linestyle='--')
        ax2.set_ylabel('clip ratio')
        # group size as scatter (scaled)
        if any(gsize):
            gs_norm = [ (g if g else 0) for g in gsize]
            ax1.scatter(xs, [min(p, max(p99_pc+p99_grp+[0]))*0.9 if p is not None else 0 for p in p99_pc], s=[(g or 0)*2 for g in gs_norm], alpha=0.3, label='group_size (marker size)')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines+lines2, labels+labels2, loc='upper right')
        plt.tight_layout()
        out_path = args.out_prefix + '_p99_clip.png'
        plt.savefig(out_path)
        print('Saved', out_path)

    # Plot raw p99 history (adaptive set) if available
    if p99_hist:
        plt.figure(figsize=(6,3))
        plt.plot(range(len(p99_hist)), p99_hist, label='p99_abs_pc')
        if len(p99_hist) >= 8:
            import statistics
            mean = statistics.mean(p99_hist)
            std = statistics.pstdev(p99_hist)
            thr = mean + 3*std
            plt.axhline(thr, color='red', linestyle='--', label='adaptive thr')
        plt.xlabel('build index')
        plt.ylabel('p99 abs diff')
        plt.legend()
        plt.tight_layout()
        out2 = args.out_prefix + '_p99_history.png'
        plt.savefig(out2)
        print('Saved', out2)

    # KL history combined (window + fp8)
    if kl_hist:
        kxs = list(range(len(kl_hist)))
        kw = [k.get('window') for k in kl_hist]
        kf = [k.get('fp8') for k in kl_hist]
        plt.figure(figsize=(6,3))
        plt.plot(kxs, kw, label='KL window', color='green')
        plt.plot(kxs, kf, label='KL fp8', color='magenta')
        plt.xlabel('build index')
        plt.ylabel('KL')
        plt.legend()
        plt.tight_layout()
        out3 = args.out_prefix + '_kl.png'
        plt.savefig(out3)
        print('Saved', out3)

if __name__ == '__main__':
    main()
