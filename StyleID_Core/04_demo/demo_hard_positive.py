"""Load the frozen Hard-Positive adapter used by the final Colab LFW row."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from demo_leace import file_sha256


class ResidualMetric(nn.Module):
    def __init__(self, dimension=512, rank=32, limit=0.25):
        super().__init__()
        self.limit = limit
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)

    def forward(self, x):
        delta = self.up(torch.tanh(self.down(x)))
        norm = torch.clamp(delta.norm(dim=1, keepdim=True), min=1e-12)
        delta = delta * torch.clamp(self.limit / norm, max=1.0)
        return F.normalize(x + delta, dim=1)


def load_hard_positive(weights_path, metadata_path, device):
    metadata = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    if file_sha256(weights_path) != metadata['weights_sha256']:
        raise ValueError('Hard-Positive adapter checksum mismatch')
    with np.load(weights_path, allow_pickle=False) as weights:
        if set(weights.files) != {'w_down', 'w_up'}:
            raise ValueError('Unexpected Hard-Positive adapter fields')
        down = weights['w_down'].copy()
        up = weights['w_up'].copy()
    if down.shape != (32, 512) or up.shape != (512, 32):
        raise ValueError('Expected rank-32 ID512 Hard-Positive adapter')
    if not np.isfinite(down).all() or not np.isfinite(up).all():
        raise ValueError('Nonfinite Hard-Positive adapter')
    model = ResidualMetric(limit=metadata['residual_limit'])
    with torch.no_grad():
        model.down.weight.copy_(torch.from_numpy(down))
        model.up.weight.copy_(torch.from_numpy(up))
    return model.to(device=device, dtype=torch.float32).eval(), metadata
