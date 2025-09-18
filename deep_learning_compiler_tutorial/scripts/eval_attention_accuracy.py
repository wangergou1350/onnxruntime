import json, math, argparse, torch, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernels.flash_attention_multihead import flash_attention
from utils.fp8 import PerChannelScaler, GroupScaler, FP8Format

def ref_attention(q,k,v, causal=False, attn_mask=None, dropout_p=0.0):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p=dropout_p, is_causal=causal)

def kl_divergence(p, q):
    eps = 1e-8
    p = p + eps
    q = q + eps
    return float((p * (p.log() - q.log())).sum().item())

def metric(a,b, hist_bins=50):
    diff = (a-b).abs()
    rel = diff / (b.abs() + 1e-6)
    # histogram of abs diff
    hist_vals = diff.flatten().float()
    max_val = hist_vals.max().item() + 1e-6
    bins = torch.linspace(0, max_val, hist_bins+1, device=hist_vals.device)
    counts = torch.histc(hist_vals, bins=hist_bins, min=0.0, max=max_val)
    prob = counts / counts.sum()
    # reference distribution: assume Laplace-like ~ exp(-x/mean_abs) discretized
    mean_abs = hist_vals.mean().item()
    ref_pdf = torch.exp(- (bins[:-1] + (bins[1]-bins[0])/2) / (mean_abs+1e-6))
    ref_pdf = ref_pdf / ref_pdf.sum()
    kl = kl_divergence(prob, ref_pdf)
    # percentiles
    sorted_vals, _ = torch.sort(hist_vals)
    n = sorted_vals.numel()
    def pct(p):
        if n == 0: return 0.0
        idx = min(int(p/100.0 * (n-1)), n-1)
        return float(sorted_vals[idx].item())
    result = {
        'max_abs': diff.max().item(),
        'mean_abs': diff.mean().item(),
        'p50_abs': pct(50),
        'p90_abs': pct(90),
        'p99_abs': pct(99),
        'max_rel': rel.max().item(),
        'mean_rel': rel.mean().item(),
        'abs_diff_hist_bins': hist_bins,
        'abs_diff_hist_counts': counts.cpu().tolist(),
        'kl_vs_laplace_like': kl
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--H', type=int, default=4)
    parser.add_argument('--N', type=int, default=256)
    parser.add_argument('--D', type=int, default=64)
    parser.add_argument('--window-left', type=int, default=0)
    parser.add_argument('--window-right', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--fp8', action='store_true')
    parser.add_argument('--clip-target', type=float, default=0.01, help='Target max clip ratio for adaptive group search (e.g. 0.01 = 1%)')
    parser.add_argument('--group-candidates', type=str, default='4,8,16,32,64', help='Comma list of group sizes to evaluate for adaptive search')
    parser.add_argument('--out', default='reports/attention_accuracy.json')
    parser.add_argument('--plot', action='store_true', help='Generate histogram PNG')
    args = parser.parse_args()

    torch.manual_seed(0)
    device='cuda'
    q = torch.randn(args.B,args.H,args.N,args.D, device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    full_ref = ref_attention(q.float(), k.float(), v.float(), causal=False)
    results = {}

    base_out = flash_attention(q,k,v, causal=False)
    results['baseline'] = metric(base_out.float(), full_ref)
    # per-head stats (abs mean / max diff per head aggregated)
    bh_base = (base_out.float() - full_ref).abs().mean(dim=(-2,-1))  # (B,H)
    results['baseline']['per_head_mean_abs'] = bh_base.view(-1).tolist()

    if args.window_left>0 or args.window_right>0:
        win_out = flash_attention(q,k,v, window_left=args.window_left, window_right=args.window_right)
        results['window'] = metric(win_out.float(), full_ref)
        results['window']['per_head_mean_abs'] = (win_out.float()-full_ref).abs().mean(dim=(-2,-1)).view(-1).tolist()

    if args.dropout>0:
        drop_out = flash_attention(q,k,v, dropout_p=args.dropout, seed=123)
        results['dropout'] = metric(drop_out.float(), full_ref)
        results['dropout']['per_head_mean_abs'] = (drop_out.float()-full_ref).abs().mean(dim=(-2,-1)).view(-1).tolist()

    if args.fp8:
        with torch.no_grad():
            abs_v = v.float().abs()
            # per-channel stats
            pc_scaler = PerChannelScaler(q.shape[-1], FP8Format.E4M3, device=v.device)
            pc_scaler.update(v.float())
            pc_scale = pc_scaler.scales
            pc_amax = (v.float().view(-1, q.shape[-1]).abs().max(dim=0).values)
            pc_clipped_mask = (abs_v.view(-1, q.shape[-1]) * pc_scale[None,:] > 127)
            pc_clipped = pc_clipped_mask.float().mean().item()
            # adaptive group size search
            chosen_group = None
            candidate_map = {}
            for g in [int(x) for x in args.group_candidates.split(',') if int(x)>0 and q.shape[-1] % int(x) == 0]:
                grp_scaler_tmp = GroupScaler(q.shape[-1], g, FP8Format.E4M3, device=v.device)
                grp_scaler_tmp.update(v.float())
                g_scale_tmp = grp_scaler_tmp.scales.repeat_interleave(g)
                g_clip = (abs_v.view(-1, q.shape[-1]) * g_scale_tmp[None,:] > 127).float().mean().item()
                candidate_map[str(g)] = {'clip_ratio': g_clip}
                if chosen_group is None and g_clip <= args.clip_target:
                    chosen_group = g
            if chosen_group is None:
                # fallback: pick largest (better precision) valid group dividing D
                valid = [int(x) for x in args.group_candidates.split(',') if int(x)>0 and q.shape[-1] % int(x) == 0]
                chosen_group = valid[-1] if valid else (16 if q.shape[-1] % 16 ==0 else 8)
            grp_scaler = GroupScaler(q.shape[-1], chosen_group, FP8Format.E4M3, device=v.device)
            grp_scaler.update(v.float())
            g_scale = grp_scaler.scales.repeat_interleave(chosen_group)
            g_clipped = (abs_v.view(-1, q.shape[-1]) * g_scale[None,:] > 127).float().mean().item()

        # run kernels
        fp8_out = flash_attention(q,k,v, use_fp8=True, fp8_scale=0.02, fp8_per_channel=True)
        results['fp8_per_channel'] = metric(fp8_out.float(), full_ref)
        results['fp8_per_channel']['per_head_mean_abs'] = (fp8_out.float()-full_ref).abs().mean(dim=(-2,-1)).view(-1).tolist()
        results['fp8_per_channel']['clip_ratio'] = pc_clipped
        results['fp8_per_channel']['amax_mean'] = float(pc_amax.mean().item())
        results['fp8_per_channel']['amax_max'] = float(pc_amax.max().item())
        results['fp8_per_channel']['scale_mean'] = float(pc_scale.mean().item())
        results['fp8_per_channel']['scale_max'] = float(pc_scale.max().item())

        fp8_g_out = flash_attention(q,k,v, use_fp8=True, fp8_scale=0.02, fp8_per_channel=True, fp8_group_size=chosen_group)
        results['fp8_group'] = metric(fp8_g_out.float(), full_ref)
        results['fp8_group']['per_head_mean_abs'] = (fp8_g_out.float()-full_ref).abs().mean(dim=(-2,-1)).view(-1).tolist()
        results['fp8_group']['clip_ratio'] = g_clipped
        results['fp8_group']['group_size'] = chosen_group
        results['fp8_group']['scale_mean'] = float(g_scale.mean().item())
        results['fp8_group']['scale_max'] = float(g_scale.max().item())
        results['fp8_group']['candidates'] = candidate_map
        results['fp8_group']['target_clip'] = args.clip_target

    os.makedirs('reports', exist_ok=True)
    with open(args.out,'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    if args.plot:
        # Build histogram over actual diff magnitude bins (log X) and overlay CDF
        plt.figure(figsize=(7,4))
        ax = plt.gca()
        plotted = []
        for name,color in [('baseline','blue'),('window','orange'),('fp8_per_channel','green'),('fp8_group','purple'),('dropout','red')]:
            if name in results:
                counts = torch.tensor(results[name]['abs_diff_hist_counts'], dtype=torch.float32)
                total = counts.sum()
                if total == 0: continue
                cdf = counts.cumsum(0)/total
                xs = torch.arange(len(counts)) + 1
                ax.plot(xs, counts, label=f'{name}-hist', color=color)
                ax2 = ax.twinx()
                ax2.plot(xs, cdf, linestyle='--', color=color, alpha=0.6)
                plotted.append(name)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('abs diff bin index (log)')
        ax.set_ylabel('count (log)')
        ax.set_title('Attention abs diff distribution & CDF')
        ax.legend()
        plt.tight_layout()
        plt.savefig('reports/attention_error_hist.png')
        print('Saved reports/attention_error_hist.png with CDF (names:', ','.join(plotted),')')

if __name__ == '__main__':
    import os
    main()
