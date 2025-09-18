import subprocess, json, sys, os, torch, argparse, math

THRESHOLDS = {
    'matmul_auto_speedup': 1.05,  # autotune >=5% faster than v1
    'softmax_max_diff': 1e-5,
    'layernorm_max_diff': 1e-5,
    'attention_baseline_max_abs': 2e-3,
    'attention_window_max_abs': 5e-3,
    'attention_fp8_max_abs': 1e-2,
    'attention_fp8_max_kl': 0.50,
    'attention_window_max_kl': 0.50,
    # new metrics
    'attention_fp8_p99_abs': 2.5e-2,
    'attention_fp8_clip_ratio': 0.02,  # 2% clipping upper bound
}

THRESHOLD_CONFIG = 'configs/ci_thresholds.json'

BASELINE_FILE = 'reports/perf_baseline.json'
KL_HISTORY_FILE = 'reports/kl_history.json'
FP8_TREND_FILE = 'reports/fp8_accuracy_trend.json'
P99_HISTORY_FILE = 'reports/p99_history.json'
TOLERANCE = 0.03  # ±3%

def run(cmd):
    print('>',' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"Command failed: {' '.join(cmd)}")
    return r.stdout

def load_baseline():
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE,'r') as f:
            return json.load(f)
    return None

def save_baseline(report):
    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    with open(BASELINE_FILE,'w') as f:
        json.dump(report, f, indent=2)

def within_tolerance(curr, base, tol):
    if base == 0:
        return True
    return abs(curr - base) / base <= tol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update-baseline', action='store_true', help='Update stored performance baseline')
    parser.add_argument('--fp8-p99-threshold', type=float, default=None, help='Override p99 abs diff threshold for FP8 variants (static)')
    parser.add_argument('--fp8-clip-threshold', type=float, default=None, help='Override clip ratio threshold for FP8 variants')
    parser.add_argument('--no-adaptive-p99', action='store_true', help='Disable adaptive p99 thresholding')
    parser.add_argument('--p99-history-len', type=int, default=50, help='History length for adaptive p99 computation')
    parser.add_argument('--write-adaptive-thresholds', action='store_true', help='Persist adaptive thresholds back to config file')
    args = parser.parse_args()
    run([sys.executable, 'benchmarks/benchmark_matmul.py'])
    data = json.load(open('matmul_results.json'))
    mm = data[-1]
    speedup = mm['v1_ms'] / mm['v_auto_ms']
    from kernels.softmax_layernorm import softmax, layernorm
    x = torch.randn(64,128, device='cuda', dtype=torch.float32)
    ref_sm = torch.softmax(x, dim=-1)
    out_sm = softmax(x)
    sm_diff = (ref_sm - out_sm).abs().max().item()
    ref_ln = torch.nn.functional.layer_norm(x, (x.shape[1],))
    out_ln = layernorm(x)
    ln_diff = (ref_ln - out_ln).abs().max().item()
    # Attention accuracy small shape
    run([sys.executable, 'scripts/eval_attention_accuracy.py', '--N','128','--D','64','--window-left','32','--fp8','--out','reports/attention_accuracy.json'])
    attn_data = json.load(open('reports/attention_accuracy.json'))
    attn_base = attn_data['baseline']['max_abs']
    attn_win = attn_data.get('window', {}).get('max_abs', 0.0)
    # updated keys: fp8_per_channel / fp8_group (prefer group if exists)
    fp8_pc = attn_data.get('fp8_per_channel', {})
    fp8_grp = attn_data.get('fp8_group', {})
    attn_fp8 = fp8_pc.get('max_abs', 0.0)
    kl_win = attn_data.get('window', {}).get('kl_vs_laplace_like', 0.0)
    kl_fp8 = fp8_pc.get('kl_vs_laplace_like', 0.0)
    p99_fp8 = fp8_pc.get('p99_abs', 0.0)
    clip_fp8 = fp8_pc.get('clip_ratio', 0.0)
    # group metrics (optional)
    p99_fp8_grp = fp8_grp.get('p99_abs', None)
    clip_fp8_grp = fp8_grp.get('clip_ratio', None)
    group_size = fp8_grp.get('group_size', None)
    report = {
      'matmul_v1_ms': mm['v1_ms'],
      'matmul_v_auto_ms': mm['v_auto_ms'],
      'matmul_auto_speedup': speedup,
      'softmax_max_diff': sm_diff,
      'layernorm_max_diff': ln_diff
    ,'attention_baseline_max_abs': attn_base,
    'attention_window_max_abs': attn_win,
    'attention_fp8_max_abs': attn_fp8,
    'attention_window_kl': kl_win,
    'attention_fp8_kl': kl_fp8,
    'attention_fp8_p99_abs': p99_fp8,
    'attention_fp8_clip_ratio': clip_fp8,
    'attention_fp8_group_p99_abs': p99_fp8_grp,
    'attention_fp8_group_clip_ratio': clip_fp8_grp,
    'attention_fp8_group_size': group_size
    }
    print(json.dumps(report, indent=2))
    baseline = load_baseline()
    if args.update_baseline or baseline is None:
        save_baseline(report)
        print('Baseline updated.')
    else:
        # Compare with tolerance (latency metrics should not increase > +3%)
        regressions = []
        for k in ['matmul_v1_ms','matmul_v_auto_ms']:
            if not within_tolerance(report[k], baseline[k]*(1+TOLERANCE), 0):
                # allow slower up to +3%; if more, regression
                if report[k] > baseline[k]*(1+TOLERANCE):
                    regressions.append(f'{k} regression {report[k]:.3f}ms > {(baseline[k]*(1+TOLERANCE)):.3f}ms (baseline {baseline[k]:.3f})')
        # speedup should not drop below baseline*(1 - TOLERANCE)
        if report['matmul_auto_speedup'] < baseline['matmul_auto_speedup']*(1-TOLERANCE):
            regressions.append(f'speedup dropped {report["matmul_auto_speedup"]:.3f} < {(baseline["matmul_auto_speedup"]*(1-TOLERANCE)):.3f}')
        if regressions:
            print('Baseline comparison failures:')
            for r in regressions:
                print(' -', r)
            raise SystemExit('Performance regression beyond tolerance')
        else:
            print('Performance within baseline tolerance.')
    # Adaptive KL thresholding: maintain moving average + 3*std
    def update_kl_history(win, fp8):
        hist = []
        if os.path.exists(KL_HISTORY_FILE):
            try:
                hist = json.load(open(KL_HISTORY_FILE,'r'))
            except Exception:
                hist = []
        hist.append({'window': win, 'fp8': fp8})
        # keep last 50 entries
        hist = hist[-50:]
        with open(KL_HISTORY_FILE,'w') as f:
            json.dump(hist, f, indent=2)
        import statistics
        window_vals = [h['window'] for h in hist if h['window'] is not None]
        fp8_vals = [h['fp8'] for h in hist if h['fp8'] is not None]
        def stats(vals):
            if len(vals) < 5:
                return None
            m = statistics.mean(vals)
            sd = statistics.pstdev(vals)
            return m + 3*sd
        return stats(window_vals), stats(fp8_vals)

    adaptive_win, adaptive_fp8 = update_kl_history(kl_win, kl_fp8)
    if adaptive_win is not None and kl_win > adaptive_win:
        raise SystemExit(f'Attention window KL {kl_win:.4f} > adaptive limit {adaptive_win:.4f}')
    if adaptive_fp8 is not None and kl_fp8 > adaptive_fp8:
        raise SystemExit(f'Attention FP8 KL {kl_fp8:.4f} > adaptive limit {adaptive_fp8:.4f}')

    # allow CLI overrides
    p99_thresh = args.fp8_p99_threshold or THRESHOLDS['attention_fp8_p99_abs']
    clip_thresh = args.fp8_clip_threshold or THRESHOLDS['attention_fp8_clip_ratio']

    if speedup < THRESHOLDS['matmul_auto_speedup']:
        raise SystemExit('Autotune speedup below threshold')
    if sm_diff > THRESHOLDS['softmax_max_diff']:
        raise SystemExit('Softmax diff too large')
    if ln_diff > THRESHOLDS['layernorm_max_diff']:
        raise SystemExit('LayerNorm diff too large')
    if attn_base > THRESHOLDS['attention_baseline_max_abs']:
        raise SystemExit('Attention baseline diff too large')
    if attn_win > THRESHOLDS['attention_window_max_abs']:
        raise SystemExit('Attention window diff too large')
    if attn_fp8 > THRESHOLDS['attention_fp8_max_abs']:
        raise SystemExit('Attention FP8 diff too large')
    if kl_win > THRESHOLDS['attention_window_max_kl']:
        raise SystemExit('Attention window KL too large')
    # Merge external threshold overrides if present
    if os.path.exists(THRESHOLD_CONFIG):
        try:
            ext = json.load(open(THRESHOLD_CONFIG,'r'))
            for k,v in ext.items():
                if k in THRESHOLDS and isinstance(v,(int,float)):
                    THRESHOLDS[k]=v
        except Exception as e:
            print('Failed loading threshold config:', e)

    if kl_fp8 > THRESHOLDS['attention_fp8_max_kl']:
        raise SystemExit('Attention FP8 KL too large')

    # Adaptive p99 tracking (per-channel FP8 primary)
    def update_p99_history(p99_val, history_len):
        hist = []
        if os.path.exists(P99_HISTORY_FILE):
            try:
                hist = json.load(open(P99_HISTORY_FILE,'r'))
            except Exception:
                hist = []
        hist.append(p99_val)
        hist = hist[-history_len:]
        with open(P99_HISTORY_FILE,'w') as f:
            json.dump(hist, f, indent=2)
        import statistics
        if len(hist) < 8:  # need minimum samples
            return None
        mean = statistics.mean(hist)
        std = statistics.pstdev(hist)
        return mean + 3*std
    adaptive_p99 = None if args.no_adaptive_p99 else update_p99_history(p99_fp8, args.p99_history_len)
    # Append trend record (keep last 100)
    try:
        trend = []
        if os.path.exists(FP8_TREND_FILE):
            trend = json.load(open(FP8_TREND_FILE,'r'))
        trend.append({
            'p99_abs_pc': p99_fp8,
            'clip_pc': clip_fp8,
            'p99_abs_group': p99_fp8_grp,
            'clip_group': clip_fp8_grp,
            'group_size': group_size,
            'kl_pc': kl_fp8
        })
        trend = trend[-100:]
        os.makedirs('reports', exist_ok=True)
        with open(FP8_TREND_FILE,'w') as f:
            json.dump(trend, f, indent=2)
    except Exception as e:
        print('Trend logging failed:', e)
    # Final p99 threshold selection: adaptive (if available) else static
    eff_p99_thresh = adaptive_p99 if adaptive_p99 is not None else p99_thresh
    if p99_fp8 > eff_p99_thresh:
        raise SystemExit(f'Attention FP8 p99 abs diff {p99_fp8:.4e} > threshold {eff_p99_thresh:.4e} (adaptive={adaptive_p99 is not None})')
    if clip_fp8 > clip_thresh:
        raise SystemExit(f'Attention FP8 clip ratio {clip_fp8:.4f} > threshold {clip_thresh:.4f}')
    if p99_fp8_grp is not None and p99_fp8_grp > p99_thresh:
        raise SystemExit(f'Attention FP8 group p99 abs diff {p99_fp8_grp:.4e} > threshold {p99_thresh:.4e}')
    if clip_fp8_grp is not None and clip_fp8_grp > clip_thresh:
        raise SystemExit(f'Attention FP8 group clip ratio {clip_fp8_grp:.4f} > threshold {clip_thresh:.4f}')
    # Optionally write back updated adaptive thresholds (only if adaptive used and passed)
    if args.write_adaptive_thresholds and adaptive_p99 is not None:
        os.makedirs('configs', exist_ok=True)
        try:
            cfg = {}
            if os.path.exists(THRESHOLD_CONFIG):
                cfg = json.load(open(THRESHOLD_CONFIG,'r'))
            # store new adaptive p99 as static baseline (tightened if lower)
            current = cfg.get('attention_fp8_p99_abs', THRESHOLDS['attention_fp8_p99_abs'])
            new_val = min(current, adaptive_p99)
            cfg['attention_fp8_p99_abs'] = new_val
            with open(THRESHOLD_CONFIG,'w') as f:
                json.dump(cfg, f, indent=2)
            print(f'Adaptive p99 threshold written: {new_val:.4e}')
        except Exception as e:
            print('Failed writing adaptive thresholds:', e)
    print('CI check passed.')

if __name__ == '__main__':
    main()
