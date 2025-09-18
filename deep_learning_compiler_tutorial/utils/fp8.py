import torch
import math

# FP8 formats: E4M3 (4-bit exponent, 3-bit mantissa, bias=7), E5M2 (5-bit exponent, 2-bit mantissa, bias=15)
# We implement conversion float32 <-> simulated FP8 (stored as uint8) with saturate-to-inf semantics.
# This is a reference utility; real hardware may have different rounding modes.

E4M3_MAX = 240.0  # approximate max finite value (not exact IEEE-like, for demo)
E5M2_MAX = 57344.0

class FP8Format:
    E4M3 = 'E4M3'
    E5M2 = 'E5M2'

def _calc_scale(amax, fp8_format: str):
    if fp8_format == FP8Format.E4M3:
        max_val = E4M3_MAX
    else:
        max_val = E5M2_MAX
    amax = max(amax, 1e-8)
    return max_val / amax

class DynamicScaler:
    def __init__(self, fp8_format: str, ema_decay=0.9):
        self.format = fp8_format
        self.ema_decay = ema_decay
        self.amax = 0.0

    def update(self, tensor: torch.Tensor):
        amax_local = tensor.abs().max().item()
        if self.amax == 0.0:
            self.amax = amax_local
        else:
            self.amax = self.ema_decay * self.amax + (1 - self.ema_decay) * amax_local
        return self.scale

    @property
    def scale(self):
        return _calc_scale(self.amax, self.format)

class PerChannelScaler:
    """Track amax per channel (last dimension) with optional EMA."""
    def __init__(self, channels: int, fp8_format: str, ema_decay=0.9, device='cuda'):
        self.channels = channels
        self.format = fp8_format
        self.ema_decay = ema_decay
        self.amax = torch.zeros(channels, device=device, dtype=torch.float32)

    def update(self, tensor: torch.Tensor):
        # tensor shape [..., C]; reduce over all dims except last
        assert tensor.shape[-1] == self.channels
        amax_local, _ = tensor.abs().view(-1, self.channels).max(dim=0)
        if (self.amax == 0).all():
            self.amax = amax_local
        else:
            self.amax = self.ema_decay * self.amax + (1 - self.ema_decay) * amax_local
        return self.scales

    @property
    def scales(self):  # scale factors (one per channel)
        # compute scale per channel to map amax -> max representable
        if self.format == FP8Format.E4M3:
            max_val = E4M3_MAX
        else:
            max_val = E5M2_MAX
        safe_amax = torch.clamp(self.amax, min=1e-8)
        return max_val / safe_amax


def fp8_quantize(x: torch.Tensor, scaler: DynamicScaler, fmt: str):
    s = scaler.scale
    q = torch.clamp((x * s).round(), -128, 127)  # symmetric quant approx
    return q.to(torch.int8), 1.0 / s  # store inv scale for dequant


def fp8_dequantize(q: torch.Tensor, inv_scale: float):
    return q.float() * inv_scale

def fp8_quantize_per_channel(x: torch.Tensor, scaler: PerChannelScaler):
    scales = scaler.scales  # (C,)
    # Broadcast: x [..., C] * scales[C]
    q = (x * scales).round().clamp(-128, 127).to(torch.int8)
    inv_scales = (1.0 / scales).to(torch.float32)
    return q, inv_scales

def fp8_dequantize_per_channel(q: torch.Tensor, inv_scales: torch.Tensor):
    return q.float() * inv_scales  # broadcasting along channel


def choose_format(tensor: torch.Tensor, threshold=1000.0):
    # simplistic heuristic: large dynamic range -> E5M2 else E4M3
    amax = tensor.abs().max().item()
    if amax > threshold:
        return FP8Format.E5M2
    return FP8Format.E4M3

__all__ = [
    'FP8Format', 'DynamicScaler', 'PerChannelScaler',
    'fp8_quantize', 'fp8_dequantize', 'fp8_quantize_per_channel', 'fp8_dequantize_per_channel',
    'choose_format'
]

# Serialization helpers for external EMA cache
def save_per_channel_state(path: str, scaler: PerChannelScaler):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'channels': scaler.channels,
        'format': scaler.format,
        'ema_decay': scaler.ema_decay,
        'amax': scaler.amax.cpu().tolist()
    }
    with open(path,'w') as f:
        json.dump(payload, f, indent=2)

def load_per_channel_state(path: str, device='cuda') -> PerChannelScaler:
    import json
    with open(path,'r') as f:
        payload = json.load(f)
    scaler = PerChannelScaler(payload['channels'], payload['format'], payload['ema_decay'], device=device)
    scaler.amax = torch.tensor(payload['amax'], device=device, dtype=torch.float32)
    return scaler

__all__ += ['save_per_channel_state','load_per_channel_state']

class GroupScaler:
    """Group-wise EMA amax & scaling (groups partition last dim)."""
    def __init__(self, channels: int, group_size: int, fp8_format: str, ema_decay=0.9, device='cuda'):
        assert channels % group_size == 0, 'channels must be divisible by group_size'
        self.channels = channels
        self.group_size = group_size
        self.groups = channels // group_size
        self.format = fp8_format
        self.ema_decay = ema_decay
        self.amax = torch.zeros(self.groups, device=device, dtype=torch.float32)

    def update(self, tensor: torch.Tensor):
        assert tensor.shape[-1] == self.channels
        x = tensor.abs().view(-1, self.channels)
        # reshape to (..., groups, group_size) then max over group_size
        grouped = x.view(-1, self.groups, self.group_size)
        amax_local, _ = grouped.max(dim=2)  # (B_flat, groups)
        amax_local, _ = amax_local.max(dim=0)  # (groups,)
        if (self.amax == 0).all():
            self.amax = amax_local
        else:
            self.amax = self.ema_decay * self.amax + (1 - self.ema_decay) * amax_local
        return self.scales

    @property
    def scales(self):
        if self.format == FP8Format.E4M3:
            max_val = E4M3_MAX
        else:
            max_val = E5M2_MAX
        safe = torch.clamp(self.amax, min=1e-8)
        return max_val / safe  # (groups,)

def fp8_quantize_groupwise(x: torch.Tensor, scaler: GroupScaler):
    scales = scaler.scales  # (groups,)
    groups = scaler.groups
    gs = scaler.group_size
    x_view = x.view(-1, groups, gs)
    q = (x_view * scales[None, :, None]).round().clamp(-128,127).to(torch.int8)
    inv_scales = (1.0 / scales).to(torch.float32)
    return q.view_as(x), inv_scales

def fp8_dequantize_groupwise(q: torch.Tensor, inv_scales: torch.Tensor, group_size: int):
    groups = inv_scales.shape[0]
    qv = q.view(-1, groups, group_size).float()
    return (qv * inv_scales[None,:,None]).view_as(q)

def save_group_state(path: str, scaler: GroupScaler):
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'channels': scaler.channels,
        'group_size': scaler.group_size,
        'format': scaler.format,
        'ema_decay': scaler.ema_decay,
        'amax': scaler.amax.cpu().tolist()
    }
    with open(path,'w') as f:
        json.dump(payload, f, indent=2)

def load_group_state(path: str, device='cuda') -> GroupScaler:
    import json
    with open(path,'r') as f:
        payload = json.load(f)
    scaler = GroupScaler(payload['channels'], payload['group_size'], payload['format'], payload['ema_decay'], device=device)
    scaler.amax = torch.tensor(payload['amax'], device=device, dtype=torch.float32)
    return scaler

__all__ += ['GroupScaler','fp8_quantize_groupwise','fp8_dequantize_groupwise','save_group_state','load_group_state']
