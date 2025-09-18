import torch
import triton
import triton.language as tl

@triton.jit
def softmax_row_kernel(x_ptr, y_ptr, n_cols, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    row_start_x = x_ptr + row_id * stride_x
    row_start_y = y_ptr + row_id * stride_y
    mask = offs < n_cols
    x = tl.load(row_start_x + offs, mask=mask, other=-float('inf'))
    row_max = tl.maximum(x, 0)
    row_max = tl.max(row_max, axis=0)  # broadcast reduce
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    out = num / denom
    tl.store(row_start_y + offs, out, mask=mask)

@triton.jit
def layernorm_stats_kernel(x_ptr, mean_ptr, rstd_ptr, n_cols, stride, BLOCK_SIZE: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    ptr = x_ptr + row * stride + offs
    mask = offs < n_cols
    x = tl.load(ptr, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / n_cols
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + EPS)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)

@triton.jit
def layernorm_apply_kernel(x_ptr, y_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr,
                           n_cols, stride_x, stride_y, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offs, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + offs, mask=mask, other=0.0)
    y = (x - mean) * rstd * gamma + beta
    tl.store(y_ptr + row * stride_y + offs, y, mask=mask)


def softmax(x: torch.Tensor):
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    BLOCK = min(1024, BLOCK)
    softmax_row_kernel[(M,)](x, y, N, x.stride(0), y.stride(0), BLOCK_SIZE=BLOCK)
    return y


def layernorm(x: torch.Tensor, eps=1e-5):
    assert x.is_cuda and x.dim() == 2
    M, N = x.shape
    mean = torch.empty((M,), device=x.device, dtype=x.dtype)
    rstd = torch.empty_like(mean)
    y = torch.empty_like(x)
    gamma = torch.ones((N,), device=x.device, dtype=x.dtype)
    beta = torch.zeros_like(gamma)
    BLOCK = triton.next_power_of_2(N)
    BLOCK = min(1024, BLOCK)
    layernorm_stats_kernel[(M,)](x, mean, rstd, N, x.stride(0), BLOCK_SIZE=BLOCK, EPS=eps)
    layernorm_apply_kernel[(M,)](x, y, mean, rstd, gamma, beta, N, x.stride(0), y.stride(0), BLOCK_SIZE=BLOCK)
    return y

if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(512, 768, device='cuda', dtype=torch.float32)
    y_ref = torch.softmax(x, dim=-1)
    y = softmax(x)
    print('softmax max diff', (y - y_ref).abs().max().item())
    ln_ref = torch.nn.functional.layer_norm(x, (x.shape[1],))
    ln = layernorm(x)
    print('layernorm max diff', (ln - ln_ref).abs().max().item())
