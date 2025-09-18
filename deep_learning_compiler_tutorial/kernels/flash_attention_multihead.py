import math, torch, triton, triton.language as tl
from utils.fp8 import (DynamicScaler, PerChannelScaler, GroupScaler, FP8Format,
                       fp8_quantize, fp8_dequantize, fp8_quantize_per_channel, fp8_dequantize_per_channel,
                       fp8_quantize_groupwise, fp8_dequantize_groupwise,
                       choose_format, load_per_channel_state, save_per_channel_state, save_group_state, load_group_state)

# Simplified multi-head FlashAttention (added causal/padding mask, dropout, optional fp8 scale)
# Input shapes: Q,K,V: (B, H, N, D)
# Optional: attn_mask (B, 1, N, N) broadcast or (B,H,N,N); True/1 indicates keep, False/0 masked.
# Dropout: standard training-time scaling (inverted), deterministic seed optional.

@triton.jit
def flash_mh_kernel(
    Q, K, V, O,
    B, H, N, D,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    DROPOUT_P: tl.constexpr,
    USE_FP8: tl.constexpr,
    INV_S: tl.constexpr,
    SEED: tl.constexpr,
    MASK_PTR, stride_mask_b, stride_mask_h, stride_mask_m, stride_mask_n,
    WINDOW_LEFT: tl.constexpr, WINDOW_RIGHT: tl.constexpr,
    USE_WINDOW: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    INV_S_PTR,
):
    bh = tl.program_id(0)
    row_block = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_ptr = Q + b*stride_qb + h*stride_qh
    k_ptr = K + b*stride_kb + h*stride_kh
    v_ptr = V + b*stride_vb + h*stride_vh
    o_ptr = O + b*stride_ob + h*stride_oh
    m_i = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if USE_WINDOW:
            # restrict valid keys for each query row according to sliding window [m-WL, m+WR]
            # Build a mask matrix for this block to early-continue if fully outside
            valid_window = (offs_n[None, :] >= (offs_m[:, None] - WINDOW_LEFT)) & (offs_n[None, :] <= (offs_m[:, None] + WINDOW_RIGHT))
            # If all rows have all keys invalid, skip this BLOCK_N entirely
            if tl.all(valid_window == 0):
                continue
        q = tl.load(q_ptr + (offs_m[:, None]*stride_qn + offs_d[None, :]*stride_qd),
                    mask=(offs_m[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        k = tl.load(k_ptr + (offs_n[None, :]*stride_kn + offs_d[:, None]*stride_kd),
                    mask=(offs_n[None, :] < N) & (offs_d[:, None] < D), other=0.0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        if CAUSAL:
            causal_mask = offs_n[None, :] <= offs_m[:, None]
            scores = tl.where(causal_mask, scores, -float('inf'))
        if USE_WINDOW:
            scores = tl.where(valid_window, scores, -float('inf'))
        if MASK_PTR != 0:
            # load boolean/binary mask
            mask_vals = tl.load(MASK_PTR + b*stride_mask_b + h*stride_mask_h + offs_m[:, None]*stride_mask_m + offs_n[None, :]*stride_mask_n,
                                mask=(offs_m[:, None] < N) & (offs_n[None, :] < N), other=0)
            scores = tl.where(mask_vals > 0, scores, -float('inf'))
        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        scores_exp = tl.exp(scores - m_new[:, None])
        if DROPOUT_P > 0:
            # simple counter-based RNG: hash(row, col, seed)
            rnd = (offs_m[:, None]*1315423911 + offs_n[None, :]*2654435761 + SEED) & 0xffffffff
            keep = (rnd.astype(tl.float32) / 2**32) > DROPOUT_P
            scores_exp = scores_exp * keep / (1 - DROPOUT_P)
        l_curr = tl.sum(scores_exp, axis=1)
        l_new = tl.exp(m_i - m_new) * l_i + l_curr
        v = tl.load(v_ptr + (offs_n[:, None]*stride_vn + offs_d[None, :]*stride_vd),
                    mask=(offs_n[:, None] < N) & (offs_d[None, :] < D), other=0.0)
        if USE_FP8:
            if PER_CHANNEL:
                # load per-channel inverse scales (length D)
                inv_s = tl.load(INV_S_PTR + offs_d, mask=offs_d < D, other=1.0)
                v = v.to(tl.float32) * inv_s[None, :]
            else:
                v = v.to(tl.float32) * INV_S
        acc = acc * (tl.exp(m_i - m_new) * (l_i / l_new))[:, None] + tl.dot(scores_exp, v) * (1.0 / l_new)[:, None]
        m_i = m_new
        l_i = l_new
    if USE_FP8:
        if PER_CHANNEL:
            inv_s = tl.load(INV_S_PTR + offs_d, mask=offs_d < D, other=1.0)
            q_out = tl.clip(acc * inv_s[None, :], -127, 127)
        else:
            q_out = tl.clip(acc / INV_S, -127, 127)
        tl.store(o_ptr + (offs_m[:, None]*stride_on + offs_d[None, :]*stride_od), q_out.to(tl.int8),
                 mask=(offs_m[:, None] < N) & (offs_d[None, :] < D))
    else:
        tl.store(o_ptr + (offs_m[:, None]*stride_on + offs_d[None, :]*stride_od), acc,
                 mask=(offs_m[:, None] < N) & (offs_d[None, :] < D))


def flash_attention(
    q, k, v,
    attn_mask=None,
    causal=False,
    dropout_p=0.0,
    seed=0,
    use_fp8=False,
    fp8_scale=1.0,
    window_left=None,
    window_right=None,
    fp8_dynamic=False,
    fp8_format: str = None,
    v_scale_state: dict = None,
    fp8_per_channel: bool = False,
    fp8_state_path: str = None,
    fp8_group_size: int = None,
    save_updated_state: bool = False,
):
    B,H,N,D = q.shape
    if use_fp8:
        o = torch.empty((B,H,N,D), device=q.device, dtype=torch.int8)
    else:
        o = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = D
    grid = (B*H, triton.cdiv(N, BLOCK_M))
    mask_ptr = 0
    stride_mb = stride_mh = stride_mm = stride_mn = 0
    if attn_mask is not None:
        assert attn_mask.shape[-2:] == (N,N)
        if attn_mask.dtype != torch.int32:
            mask = attn_mask.to(torch.int32)
        else:
            mask = attn_mask
        mask_ptr = mask
        stride_mb, stride_mh, stride_mm, stride_mn = mask.stride()
    use_window = (window_left is not None) and (window_right is not None)
    wl = int(window_left) if use_window else 0
    wr = int(window_right) if use_window else 0
    inv_s_ptr = 0
    per_channel_flag = False
    scaler = None
    if use_fp8 and fp8_per_channel and (fp8_group_size is None):
        if fp8_state_path and os.path.exists(fp8_state_path):
            scaler = load_per_channel_state(fp8_state_path, device=v.device)
        else:
            scaler = PerChannelScaler(D, fp8_format or FP8Format.E4M3, ema_decay=0.9, device=v.device)
        scaler.update(v.float())
        inv_scales = (1.0 / scaler.scales).to(torch.float32)
        inv_s_ptr = inv_scales
        per_channel_flag = True
    elif use_fp8 and fp8_group_size is not None:
        # group-wise path (override per-channel flag)
        if fp8_state_path and os.path.exists(fp8_state_path):
            scaler = load_group_state(fp8_state_path, device=v.device)
        else:
            scaler = GroupScaler(D, fp8_group_size, fp8_format or FP8Format.E4M3, ema_decay=0.9, device=v.device)
        scaler.update(v.float())
        # expand group inverse scales to channel vector
        group_inv = (1.0 / scaler.scales).repeat_interleave(fp8_group_size).to(torch.float32)
        inv_s_ptr = group_inv
        per_channel_flag = True  # reuse same kernel branch
    flash_mh_kernel[grid](
        q, k, v, o,
        B, H, N, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        SCALE=scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_D,
        CAUSAL=causal, DROPOUT_P=dropout_p, USE_FP8=use_fp8, INV_S=1.0/fp8_scale, SEED=seed,
        MASK_PTR=mask_ptr if mask_ptr != 0 else 0,
        stride_mask_b=stride_mb, stride_mask_h=stride_mh, stride_mask_m=stride_mm, stride_mask_n=stride_mn,
    WINDOW_LEFT=wl, WINDOW_RIGHT=wr, USE_WINDOW=use_window,
    PER_CHANNEL=per_channel_flag,
    INV_S_PTR=inv_s_ptr if per_channel_flag else 0,
    )
    if save_updated_state and scaler is not None and fp8_state_path:
        if isinstance(scaler, PerChannelScaler):
            save_per_channel_state(fp8_state_path, scaler)
        else:
            save_group_state(fp8_state_path, scaler)
    return o

if __name__ == '__main__':
    torch.manual_seed(0)
    B,H,N,D = 2, 2, 128, 64
    q = torch.randn(B,H,N,D, device='cuda', dtype=torch.float16)
    k = torch.randn(B,H,N,D, device='cuda', dtype=torch.float16)
    v = torch.randn(B,H,N,D, device='cuda', dtype=torch.float16)
    ref = torch.nn.functional.scaled_dot_product_attention(q.float(), k.float(), v.float(), dropout_p=0.0, is_causal=True)
    out = flash_attention(q,k,v, causal=True)
    print('causal max diff', (ref.half()-out.half()).abs().max().item())
    # FP8 path demo (static scale)
    scale = 0.02
    out_fp8 = flash_attention(q,k,v, causal=True, use_fp8=True, fp8_scale=scale)
    print('fp8 stored int8 range', out_fp8.min().item(), out_fp8.max().item())
    # sliding window demo (look back 32, look ahead 0)
    sw_out = flash_attention(q,k,v, causal=False, window_left=32, window_right=0)
    print('sliding window output dtype', sw_out.dtype)
