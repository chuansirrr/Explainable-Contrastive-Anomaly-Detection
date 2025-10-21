import yaml, torch, numpy as np, os, pickle
from xadl.utils import set_seed, get_device
from xadl.data import get_dataloaders
from xadl.models.encoder import build_encoder
from xadl.models.heads import MahalanobisHead, KNNHead, OneClassSVMHead

def _embeddings(encoder, loader, device):
    encoder.eval()
    Z = []
    with torch.no_grad():
        for xb in loader:
            if isinstance(xb, (list, tuple)): xb = xb[0]
            xb = xb.to(device)
            z = encoder(xb).cpu().numpy()
            Z.append(z)
    return np.concatenate(Z, axis=0)

def _head_from_cfg(cfg, override=None):
    head = override or cfg["detector"]["head"]
    if head == "mahalanobis":
        return MahalanobisHead()
    elif head == "knn":
        return KNNHead(k=cfg["detector"].get("knn_k", 10))
    elif head == "ocsvm":
        return OneClassSVMHead(nu=cfg["detector"].get("ocsvm_nu", 0.05))
    else:
        raise ValueError(f"Unknown head: {head}")

def train_detector(cfg_path: str, encoder_path="artifacts/encoder.pt", head_name=None):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))
    device = get_device()
    tr_loader, va_loader, te_loader = get_dataloaders(cfg, for_contrastive=False)

    # build & load encoder
    xb0 = next(iter(tr_loader))
    if isinstance(xb0, (list, tuple)): xb0 = xb0[0]
    if cfg["dataset"]["kind"].endswith("tabular"):
        enc = build_encoder(cfg, in_dim=xb0.shape[-1]).to(device)
    else:
        enc = build_encoder(cfg, in_channels=xb0.shape[1]).to(device)
    state = torch.load(encoder_path, map_location=device)
    enc.load_state_dict(state["state_dict"])

    Z_tr = _embeddings(enc, tr_loader, device)
    head = _head_from_cfg(cfg, head_name)
    head.fit(Z_tr)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/head.pkl", "wb") as f: pickle.dump(head, f)
    print("Saved detector head to artifacts/head.pkl")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default="artifacts/encoder.pt")
    ap.add_argument("--head", default=None, help="mahalanobis|knn|ocsvm")
    args = ap.parse_args()
    train_detector(args.config, args.encoder, args.head)
