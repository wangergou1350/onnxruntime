import torch, math
import triton
import triton.language as tl

@triton.jit
def qk_matmul_kernel(q_ptr, k_ptr, scores_ptr, M, N, K, stride_qm, stride_qk, stride_kn, stride_kk, stride_sm, stride_sn,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SCALE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = k0 + offs_k < K
        q_tile = tl.load(q_ptr + (offs_m[:, None] * stride_qm + (k0 + offs_k)[None, :] * stride_qk),
                         mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        k_tile = tl.load(k_ptr + ((k0 + offs_k)[:, None] * stride_kk + offs_n[None, :] * stride_kn),
                         mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(q_tile, tl.trans(k_tile))
    acc = acc * SCALE
    tl.store(scores_ptr + (offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn), acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def softmax_block_kernel(scores_ptr, out_ptr, M, N, stride_sm, stride_sn, stride_om, stride_on, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_scores = scores_ptr + row * stride_sm + offs * stride_sn
    mask = offs < N
    x = tl.load(row_scores, mask=mask, other=-float('inf'))
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row * stride_om + offs * stride_on, y, mask=mask)

# simplified flash-like kernel (single stage, demonstration only) omitted advanced online updates for brevity
@triton.jit
def flash_step_kernel(q_ptr, k_ptr, v_ptr, o_ptr, M, N, K, D,
                      stride_qm, stride_qk,
                      stride_kn, stride_kk,
                      stride_vn, stride_vk,
                      stride_om, stride_od,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                      SCALE: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # accumulator for output rows
    o_acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    m_prev = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n_mask = n0 + offs_n < N
        # compute scores block (offs_m x offs_n)
        scores_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_mask = k0 + offs_k < K
            q_tile = tl.load(q_ptr + (offs_m[:, None] * stride_qm + (k0 + offs_k)[None, :] * stride_qk),
                             mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            k_tile = tl.load(k_ptr + ((n0 + offs_n)[None, :] * stride_kn + (k0 + offs_k)[:, None] * stride_kk),
                             mask=n_mask[None, :] & k_mask[:, None], other=0.0)
            scores_block += tl.dot(q_tile, k_tile)
        scores_block *= SCALE
        # online softmax merge
        m_curr = tl.max(scores_block, axis=1)
        m_new = tl.maximum(m_prev, m_curr)
        scores_block = tl.exp(scores_block - m_new[:, None])
        l_curr = tl.sum(scores_block, axis=1)
        l_new = tl.exp(m_prev - m_new) * l_prev + l_curr
        # load V and accumulate
        v_tile = tl.load(v_ptr + ((n0 + offs_n)[:, None] * stride_vn + tl.arange(0, D)[None, :] * stride_vk),
                         mask=n_mask[:, None], other=0.0)
        o_update = tl.dot(scores_block, v_tile)
        o_acc = o_acc * (tl.exp(m_prev - m_new) * (l_prev / l_new))[:, None] + o_update * (1.0 / l_new[:, None])
        m_prev = m_new
        l_prev = l_new
    tl.store(o_ptr + (offs_m[:, None] * stride_om + tl.arange(0, D)[None, :] * stride_od), o_acc,
             mask=(offs_m[:, None] < M))

if __name__ == '__main__':
    torch.manual_seed(0)
    B = 1; H = 1; N = 256; D = 64
    scale = 1.0 / math.sqrt(D)
    q = torch.randn((N, D), device='cuda', dtype=torch.float16)
    k = torch.randn((N, D), device='cuda', dtype=torch.float16)
    v = torch.randn((N, D), device='cuda', dtype=torch.float16)
    # reference
    ref = torch.softmax((q @ k.t()) * scale, dim=-1) @ v
    # simplified flash
    o = torch.empty_like(ref)
    flash_step_kernel[( ( (N + 63)//64), )](q, k, v, o,
        N, N, D, D,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
        v.stride(0), v.stride(1),
        o.stride(0), o.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, SCALE=scale)
    print('flash diff', (o - ref).abs().max().item())
