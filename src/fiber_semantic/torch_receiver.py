"""Optional differentiable CPU/CUDA receiver using the same offline calibration.

Requires the `attack` extra (PyTorch). Offline PCA/table construction remains
on CPU; this module performs batched online response projection and recovery.
"""
import torch
from torch import nn


class TorchReceiver(nn.Module):
    def __init__(self, calibration, *, dtype=torch.float64):
        super().__init__()
        for name in ['mean', 'basis', 'lower', 'upper']:
            self.register_buffer(name, torch.tensor(getattr(calibration, name).copy(), dtype=dtype))
        self.gain = float(calibration.gain)
        self.eps = float(calibration.eps)

    def forward(self, responses):
        """Map (..., 2P, 7) responses to (..., P, 2) values in [-1, 1]."""
        if responses.ndim < 2 or responses.shape[-1] != 7 or responses.shape[-2] == 0 or responses.shape[-2] % 2:
            raise ValueError('Expected (..., 2P, 7) responses with P > 0')
        if responses.device != self.mean.device:
            raise ValueError('Move receiver and responses to the same device')
        responses = responses.to(dtype=self.mean.dtype)
        if not torch.isfinite(responses).all():
            raise ValueError('Responses must be finite')
        b = (responses - self.mean) @ self.basis
        k = 2 * (b - self.lower) / (self.upper - self.lower + self.eps) - 1
        return torch.clamp(self.gain * (k[..., 0::2, :] - k[..., 1::2, :]), -1, 1)

    def recover_bytes(self, responses):
        q = self(responses)
        return torch.clamp(torch.round((q + 1) * 255 / 2), 0, 255).to(torch.uint8).flatten(-2)

