import torch
import torch.nn.functional as F

def nt_xent(z_i, z_j, temperature: float = 0.2):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).
    z_i, z_j: [B, D] positive pairs
    """
    z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T)  # [2B,2B]
    B = z_i.size(0)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim = sim[~mask].view(2*B, 2*B-1)
    positives = torch.cat([torch.diag(sim, B-1), torch.diag(sim, -(B-1))], dim=0)
    logits = sim / temperature
    labels = torch.arange(2*B, device=z.device)
    labels = (labels + B) % (2*B)  # positive index
    loss = F.cross_entropy(logits, labels)
    return loss
