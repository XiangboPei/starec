"""Fixed wavelet packet transform for StaRec (orthogonal filters via PyWavelets)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

SUPPORTED_FIXED_WAVELETS = ("haar", "db2", "db4", "sym4", "sym6")


def dwt_forward(x, h0, h1):
    h0_flip = h0.flip(dims=[0]).reshape(1, 1, -1)
    h1_flip = h1.flip(dims=[0]).reshape(1, 1, -1)
    filters = torch.cat([h0_flip, h1_flip], dim=0)

    bsz, channels, signal_len = x.shape
    filt_len = len(h0)
    filters = filters.repeat(channels, 1, 1)

    out_len = (signal_len + filt_len - 1) // 2
    total_pad = 2 * (out_len - 1) - signal_len + filt_len

    if total_pad % 2 == 1:
        x = F.pad(x, (0, 1))

    conv_pad = total_pad // 2
    res = F.conv1d(x, filters, stride=2, padding=conv_pad, groups=channels)
    res = res.view(bsz, channels, 2, -1)
    return res[:, :, 0, :], res[:, :, 1, :]


def dwt_inverse(low, high, h0, h1, target_len=None, output_padding=0):
    h0_flip = h0.flip(dims=[0]).reshape(1, 1, -1)
    h1_flip = h1.flip(dims=[0]).reshape(1, 1, -1)
    filters = torch.cat([h0_flip, h1_flip], dim=0)

    channels = low.shape[1]
    filters = filters.repeat(channels, 1, 1)
    inputs = torch.stack([low, high], dim=2).view(low.shape[0], -1, low.shape[-1])

    filt_len = len(h0)
    pad = filt_len - 2
    output_padding = int(max(0, min(1, output_padding)))
    recon = F.conv_transpose1d(
        inputs,
        filters,
        stride=2,
        padding=pad,
        output_padding=output_padding,
        groups=channels,
    )

    if target_len is not None:
        if recon.shape[-1] > target_len:
            recon = recon[..., :target_len]
        elif recon.shape[-1] < target_len:
            recon = F.pad(recon, (0, target_len - recon.shape[-1]))
    return recon


def get_fixed_wavelet_filters(wave, dtype=torch.float32):
    if wave not in SUPPORTED_FIXED_WAVELETS:
        raise ValueError(f"Unsupported fixed wavelet '{wave}'. Choose from {SUPPORTED_FIXED_WAVELETS}.")
    wavelet = pywt.Wavelet(wave)
    h0 = torch.tensor(wavelet.dec_lo, dtype=dtype)
    h1 = torch.tensor(wavelet.dec_hi, dtype=dtype)
    return h0, h1


class WaveletPacketDecomposition(nn.Module):
    """Depth-`J` wavelet packet decomposition using fixed orthogonal filters."""

    def __init__(self, J: int = 3, wave: str = "sym4"):
        super().__init__()
        self.J = int(J)
        self.wave = wave
        h0, h1 = get_fixed_wavelet_filters(wave)
        self.register_buffer("fixed_h0", h0)
        self.register_buffer("fixed_h1", h1)
        self.filter_length = len(h0)
        self.cached_shapes = []
        self.all_levels = []

    def wavelet_filters(self):
        return self.fixed_h0, self.fixed_h1

    def forward(self, x):
        h0, h1 = self.wavelet_filters()
        h0 = h0.to(device=x.device, dtype=x.dtype)
        h1 = h1.to(device=x.device, dtype=x.dtype)
        nodes = [x]
        self.cached_shapes = [x.shape[-1]]
        self.all_levels = [nodes.copy()]

        for _ in range(self.J):
            next_nodes = []
            for node in nodes:
                low, high = dwt_forward(node, h0, h1)
                next_nodes.append(low)
                next_nodes.append(high)
            nodes = next_nodes
            self.all_levels.append(nodes.copy())
            if len(nodes) > 0:
                self.cached_shapes.append(nodes[0].shape[-1])

        ca = nodes[0]
        cd = nodes[1:]
        return ca, cd

    def inverse(self, coeffs):
        h0, h1 = self.wavelet_filters()
        ca, cd = coeffs
        h0 = h0.to(device=ca.device, dtype=ca.dtype)
        h1 = h1.to(device=ca.device, dtype=ca.dtype)
        nodes = [ca] + list(cd)

        for j in range(self.J, 0, -1):
            prev_nodes = []
            target_len = self.cached_shapes[j - 1]
            for i in range(0, len(nodes), 2):
                low_child = nodes[i]
                high_child = nodes[i + 1]
                min_len = min(low_child.shape[-1], high_child.shape[-1])
                low_child = low_child[..., :min_len]
                high_child = high_child[..., :min_len]

                filt_len = len(h0)
                raw_out_len = 2 * min_len - filt_len
                op = target_len - raw_out_len
                op = max(0, min(1, op))
                parent = dwt_inverse(
                    low_child,
                    high_child,
                    h0,
                    h1,
                    target_len=target_len,
                    output_padding=op,
                )
                prev_nodes.append(parent)
            nodes = prev_nodes
        return nodes[0]
