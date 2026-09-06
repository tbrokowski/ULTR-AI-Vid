"""Temporal style parameters are shared over the whole clip; inputs are [T,C,H,W]."""
import torch
from torch.nn import functional as F


def fourier_style(clip, donor, fraction=0.05, blend=0.5):
    # Shift only amplitude; use temporally averaged donor amplitude, preserve phase.
    source = torch.fft.fft2(clip.float(), dim=(-2, -1))
    target = torch.fft.fft2(donor.float(), dim=(-2, -1))
    amp = torch.fft.fftshift(source.abs(), dim=(-2, -1))
    donor_amp = torch.fft.fftshift(target.abs().mean(0, keepdim=True), dim=(-2, -1))
    h, w = clip.shape[-2:]
    radius = max(1, int(min(h, w) * fraction))
    mask = torch.zeros_like(amp)
    mask[..., h // 2 - radius:h // 2 + radius + 1, w // 2 - radius:w // 2 + radius + 1] = 1
    amplitude = amp * (1 - blend * mask) + donor_amp * blend * mask
    amplitude = torch.fft.ifftshift(amplitude, dim=(-2, -1))
    return torch.fft.ifft2(amplitude * torch.exp(1j * source.angle()), dim=(-2, -1)).real


def synthetic_style(clip, generator, switches, donor=None):
    out = clip
    def uniform(lo, hi):
        return lo + (hi - lo) * torch.rand((), generator=generator).item()
    if switches.get("gain", False):
        out = out * uniform(0.8, 1.2)
    if switches.get("speckle", False):
        field = torch.randn((1, 1, *out.shape[-2:]), generator=generator) * uniform(0.03, 0.12)
        out = out * (1 + field)
    if switches.get("resampling", False):
        h, w = out.shape[-2:]
        scale = uniform(0.65, 0.95)
        out = F.interpolate(out, size=(int(h * scale), int(w * scale)), mode="bilinear", align_corners=False)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    if switches.get("fourier", False):
        if donor is None:
            raise ValueError("Fourier style requires a verified training-patient donor")
        out = fourier_style(out, donor, switches.get("fourier_fraction", 0.05), switches.get("fourier_blend", 0.5))
    return out.clamp(0, 1)
