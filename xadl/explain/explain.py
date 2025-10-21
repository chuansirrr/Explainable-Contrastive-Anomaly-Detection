import numpy as np
import torch

def gradient_saliency(encoder, x):
    """
    Computes gradient w.r.t input for the L2 norm of embedding (proxy for anomaly score).
    x: torch tensor [B, ...] with requires_grad=False
    Returns numpy array of absolute gradients aggregated over trailing dims if present.
    """
    encoder.eval()
    x = x.clone().detach().requires_grad_(True)
    z = encoder(x)
    obj = (z**2).sum(dim=1).mean()
    obj.backward()
    g = x.grad.detach().cpu().numpy()
    if g.ndim > 2:
        # average across channels/length for a per-feature per-sample score
        axes = tuple(range(2, g.ndim))
        return np.abs(g).mean(axis=axes)
    return np.abs(g)

def permutation_importance(score_fn, x, n_repeats=8, random_state=42):
    """
    score_fn: callable that maps batch numpy array -> anomaly scores (higher worse)
    x: numpy array [B, F] (tabular)
    """
    rng = np.random.default_rng(random_state)
    base = score_fn(x).mean()
    B, F = x.shape
    importances = np.zeros(F)
    for f in range(F):
        v = x.copy()
        for _ in range(n_repeats):
            rng.shuffle(v[:, f])
        s = score_fn(v).mean()
        importances[f] = s - base
    return importances
