import argparse, yaml, numpy as np, pickle, torch
from xadl.utils import get_device
from xadl.data import get_dataloaders
from xadl.models.encoder import build_encoder

def main(cfg_path, encoder_path, head_path, head_name=None):
    cfg = yaml.safe_load(open(cfg_path))
    device = get_device()
    _, val_loader, test_loader = get_dataloaders(cfg, for_contrastive=False)
    xb0 = next(iter(test_loader))
    if isinstance(xb0, (list, tuple)): xb0 = xb0[0]
    if cfg["dataset"]["kind"].endswith("tabular"):
        enc = build_encoder(cfg, in_dim=xb0.shape[-1]).to(device)
    else:
        enc = build_encoder(cfg, in_channels=xb0.shape[1]).to(device)
    state = torch.load(encoder_path, map_location=device)
    enc.load_state_dict(state["state_dict"])
    enc.eval()

    with open(head_path, "rb") as f: head = pickle.load(f)

    def embed(loader):
        Z=[]; X=[]
        with torch.no_grad():
            for xb in loader:
                if isinstance(xb, (list, tuple)): xb = xb[0]
                X.append(xb.numpy())
                xb = xb.to(device)
                Z.append(enc(xb).cpu().numpy())
        return (np.concatenate(X), np.concatenate(Z))

    Xv, Zv = embed(val_loader)
    Xt, Zt = embed(test_loader)
    sv = head.score(Zv); st = head.score(Zt)
    print("Validation score mean/std:", sv.mean(), sv.std())
    print("Test score mean/std:", st.mean(), st.std())
    np.save("artifacts/test_scores.npy", st)
    print("Saved artifacts/test_scores.npy")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default="artifacts/encoder.pt")
    ap.add_argument("--head_path", default="artifacts/head.pkl")
    ap.add_argument("--head", default=None)
    args = ap.parse_args()
    main(args.config, args.encoder, args.head_path, args.head)
