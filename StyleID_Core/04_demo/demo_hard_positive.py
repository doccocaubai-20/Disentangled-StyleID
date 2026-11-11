"""Load the frozen Hard-Positive head used by the final Colab LFW row."""
import torch
from torch import nn
from torch.nn import functional as F


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


def load_hard_positive(path, device, *, soup_sha256, base_revision, styleid_revision):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    expected = {
        'dimension': 512,
        'rank': 32,
        'residual_limit': 0.25,
        'checkpoint_sha256': soup_sha256,
        'base_revision': base_revision,
        'styleid_revision': styleid_revision,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError('Hard-Positive checkpoint metadata differs from the final LFW configuration')
    model = ResidualMetric(limit=checkpoint['residual_limit'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.requires_grad_(False).to(device=device, dtype=torch.float32).eval()
