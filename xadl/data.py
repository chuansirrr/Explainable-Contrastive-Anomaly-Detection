import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def _augment_tabular(x, noise_std=0.02, drop_p=0.1):
    x1 = x + np.random.normal(0, noise_std, size=x.shape)
    mask = np.random.rand(*x.shape) < drop_p
    x2 = x.copy()
    x2[mask] = x2[mask] * 0.9
    return x1.astype(np.float32), x2.astype(np.float32)

def _augment_timeseries(x):
    # simple jitter + scaling
    jitter = np.random.normal(0, 0.01, size=x.shape)
    scale = np.random.uniform(0.9, 1.1, size=(x.shape[0], 1, 1))
    return (x + jitter).astype(np.float32), (x * scale).astype(np.float32)

class TabularContrastiveDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        x1, x2 = _augment_tabular(x)
        return torch.from_numpy(x1), torch.from_numpy(x2)

class TabularPlainDataset(Dataset):
    def __init__(self, X): self.X = X.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx])

class TimeSeriesContrastiveDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]
        x1, x2 = _augment_timeseries(x)
        return torch.from_numpy(x1), torch.from_numpy(x2)

class TimeSeriesPlainDataset(Dataset):
    def __init__(self, X): self.X = X.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.from_numpy(self.X[idx])

def synthetic_tabular(n_samples=6000, n_features=20, anomaly_fraction=0.05, seed=42):
    rng = np.random.default_rng(seed)
    mean = np.zeros(n_features); cov = np.eye(n_features)
    X = rng.multivariate_normal(mean, cov, size=n_samples)
    n_anom = int(n_samples * anomaly_fraction)
    if n_anom>0:
        idx = rng.choice(n_samples, n_anom, replace=False)
        X[idx] += rng.normal(5, 1.0, size=(n_anom, n_features))  # shift anomalies
    return X

def synthetic_timeseries(n_series=800, length=128, n_channels=3, anomaly_fraction=0.08, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_series, n_channels, length))
    n_anom = int(n_series * anomaly_fraction)
    if n_anom>0:
        idx = rng.choice(n_series, n_anom, replace=False)
        for i in idx:
            ch = rng.integers(0, n_channels)
            pos = rng.integers(length//4, 3*length//4)
            X[i, ch, pos:pos+5] += 6.0  # spike anomaly
    return X

def get_dataloaders(cfg, for_contrastive=True):
    kind = cfg["dataset"]["kind"]
    if kind == "synthetic_tabular":
        X = synthetic_tabular(
            n_samples=cfg["dataset"]["n_samples"],
            n_features=cfg["dataset"]["n_features"],
            anomaly_fraction=cfg["dataset"]["anomaly_fraction"],
            seed=cfg.get("seed", 42))
        n = len(X)
        tr = int(n*cfg["dataset"]["train_frac"]); va = int(n*cfg["dataset"]["val_frac"])
        train_X, val_X, test_X = X[:tr], X[tr:tr+va], X[tr+va:]
        if for_contrastive:
            ds_tr = TabularContrastiveDataset(train_X)
        else:
            ds_tr = TabularPlainDataset(train_X)
        ds_va = TabularPlainDataset(val_X); ds_te = TabularPlainDataset(test_X)
    elif kind == "synthetic_timeseries":
        X = synthetic_timeseries(
            n_series=cfg["dataset"]["n_series"],
            length=cfg["dataset"]["length"],
            n_channels=cfg["dataset"]["n_channels"],
            anomaly_fraction=cfg["dataset"]["anomaly_fraction"],
            seed=cfg.get("seed", 7))
        n = len(X)
        tr = int(n*cfg["dataset"]["train_frac"]); va = int(n*cfg["dataset"]["val_frac"])
        train_X, val_X, test_X = X[:tr], X[tr:tr+va], X[tr+va:]
        if for_contrastive:
            ds_tr = TimeSeriesContrastiveDataset(train_X)
        else:
            ds_tr = TimeSeriesPlainDataset(train_X)
        ds_va = TimeSeriesPlainDataset(val_X); ds_te = TimeSeriesPlainDataset(test_X)
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    bs = cfg["train"]["batch_size"]
    from torch.utils.data import DataLoader
    return (
        DataLoader(ds_tr, batch_size=bs, shuffle=True),
        DataLoader(ds_va, batch_size=bs, shuffle=False),
        DataLoader(ds_te, batch_size=bs, shuffle=False)
    )
