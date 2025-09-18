import json, os, argparse, statistics, datetime

TREND_FILE = 'reports/fp8_accuracy_trend.json'
P99_FILE = 'reports/p99_history.json'
KL_FILE = 'reports/kl_history.json'
THRESHOLD_CONFIG = 'configs/ci_thresholds.json'
REPORT_OUT = 'reports/fp8_report.md'


def load(path, default=None):
    if os.path.exists(path):
        try:
            with open(path,'r') as f:
                return json.load(f)
        except Exception:
            return default
    return default

def summarise_series(name, series):
    if not series:
        return f'- {name}: no data' 
    return (f"- {name}: n={len(series)} mean={statistics.mean(series):.4e} "
            f"p50={statistics.median(series):.4e} p90={percentile(series,90):.4e} "
            f"p99={percentile(series,99):.4e} max={max(series):.4e}")

def percentile(series, p):
    if not series:
        return 0.0
    s = sorted(series)
    k = int((p/100.0)*(len(s)-1))
    return s[k]

def recommend(p99_hist, clip_hist, thresholds):
    rec = []
    if p99_hist:
        p99_curr = p99_hist[-1]
        if p99_curr > thresholds.get('attention_fp8_p99_abs', 1e9):
            rec.append('p99 超阈值：检查最近提交的量化逻辑或者窗口/分组变化')
        elif len(p99_hist) >= 5 and p99_curr < min(p99_hist[:-1]):
            rec.append('p99 创新低：可考虑收紧静态 p99 阈值')
    if clip_hist:
        clip_curr = clip_hist[-1]
        if clip_curr > thresholds.get('attention_fp8_clip_ratio', 0.02):
            rec.append('clip_ratio 高：尝试减小 group_size 或调整 scale EMA')
        elif clip_curr < 0.002:
            rec.append('clip_ratio 很低：可能 scale 偏保守，可探索更大 group_size 节省元数据')
    if not rec:
        rec.append('无显著异常')
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=REPORT_OUT)
    args = ap.parse_args()

    trend = load(TREND_FILE, [])
    p99_hist = load(P99_FILE, [])
    kl_hist = load(KL_FILE, [])
    thresholds = load(THRESHOLD_CONFIG, {}) or {}

    p99_pc_series = [t.get('p99_abs_pc') for t in trend if t.get('p99_abs_pc') is not None]
    clip_pc_series = [t.get('clip_pc') for t in trend if t.get('clip_pc') is not None]
    p99_group_series = [t.get('p99_abs_group') for t in trend if t.get('p99_abs_group') is not None]
    clip_group_series = [t.get('clip_group') for t in trend if t.get('clip_group') is not None]
    kl_fp8_series = [t.get('kl_pc') for t in trend if t.get('kl_pc') is not None]

    recs = recommend(p99_pc_series, clip_pc_series, thresholds)

    lines = []
    lines.append(f"# FP8 Precision Report ({datetime.datetime.utcnow().isoformat()} UTC)")
    lines.append('')
    lines.append('## Thresholds')
    for k,v in thresholds.items():
        lines.append(f'- {k}: {v}')
    lines.append('')
    lines.append('## Recent Metrics Summary')
    lines.append(summarise_series('p99_abs_pc', p99_pc_series))
    lines.append(summarise_series('clip_pc', clip_pc_series))
    if p99_group_series:
        lines.append(summarise_series('p99_abs_group', p99_group_series))
    if clip_group_series:
        lines.append(summarise_series('clip_group', clip_group_series))
    if kl_fp8_series:
        lines.append(summarise_series('kl_fp8', kl_fp8_series))
    lines.append('')
    lines.append('## Adaptive History (raw p99)')
    if p99_hist:
        lines.append(summarise_series('p99_history', p99_hist))
    else:
        lines.append('- no p99 history captured')
    lines.append('')
    lines.append('## Recommendations')
    for r in recs:
        lines.append(f'- {r}')
    lines.append('')
    lines.append('## Last 5 Records (chronological)')
    for entry in trend[-5:]:
        lines.append(f'- p99_pc={entry.get("p99_abs_pc"):.4e} clip_pc={entry.get("clip_pc"):.4f} p99_group={entry.get("p99_abs_group")} clip_group={entry.get("clip_group")} gsize={entry.get("group_size")}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out,'w',encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('Report written to', args.out)

if __name__ == '__main__':
    main()
