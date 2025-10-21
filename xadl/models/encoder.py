import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(128,128), emb_dim=64):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.BatchNorm1d(h)]
            last = h
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(last, emb_dim)
    def forward(self, x):
        h = self.mlp(x)
        z = self.proj(h)
        return F.normalize(z, dim=-1)

class CNN1DEncoder(nn.Module):
    def __init__(self, in_channels=3, emb_dim=64, channels=(32,64)):
        super().__init__()
        c1, c2 = channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, c1, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(c1),
            nn.Conv1d(c1, c2, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(c2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(c2, emb_dim)
    def forward(self, x):  # x: [B,C,L]
        h = self.net(x).squeeze(-1)
        z = self.proj(h)
        return F.normalize(z, dim=-1)

def build_encoder(cfg, in_dim=None, in_channels=None):
    enc = cfg["model"]["encoder"]
    emb_dim = cfg["model"]["emb_dim"]
    if enc == "mlp":
        assert in_dim is not None
        return MLPEncoder(in_dim, tuple(cfg["model"]["hidden_dims"]), emb_dim)
    elif enc == "cnn1d":
        return CNN1DEncoder(in_channels=in_channels, emb_dim=emb_dim, channels=tuple(cfg["model"]["cnn_channels"]))
    else:
        raise ValueError(f"Unknown encoder: {enc}")
