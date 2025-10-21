import argparse, yaml, numpy as np, pickle, torch
from xadl.utils import get_device
from xadl.data import get_dataloaders
from xadl.models.encoder import build_encoder
from xadl.explain.explain import gradient_saliency, permutation_importance

def main(cfg_path, encoder_path, head_path, method="saliency", n_samples=32):
    cfg = yaml.safe_load(open(cfg_path))
    device = get_device()
    _, _, test_loader = get_dataloaders(cfg, for_contrastive=False)
    xb0 = next(iter(test_loader))
    if isinstance(xb0, (list, tuple)): xb0 = xb0[0]
    if cfg["dataset"]["kind"].endswith("tabular"):
        enc = build_encoder(cfg, in_dim=xb0.shape[-1]).to(device)
    else:
        enc = build_encoder(cfg, in_channels=xb0.shape[1]).to(device)
    state = torch.load(encoder_path, map_location=device)
    enc.load_state_dict(state["state_dict"])

    batch = next(iter(test_loader))
    if isinstance(batch, (list, tuple)): batch = batch[0]
    batch = batch[:n_samples]

    if method == "saliency":
        sal = gradient_saliency(enc, batch.to(device))
        np.save("artifacts/saliency.npy", sal)
        print("Saved artifacts/saliency.npy (per-sample saliency).")
    elif method == "pfi":
        with open(head_path, "rb") as f: head = pickle.load(f)
        def score_fn(x_np):
            x = torch.from_numpy(x_np).to(device)
            z = enc(x).detach().cpu().numpy()
            return head.score(z)
        sal = permutation_importance(score_fn, batch.numpy(), n_repeats=8)
        np.save("artifacts/pfi.npy", sal)
        print("Saved artifacts/pfi.npy (global feature importances).")
    else:
        raise ValueError("Unknown method")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default="artifacts/encoder.pt")
    ap.add_argument("--head_path", default="artifacts/head.pkl")
    ap.add_argument("--method", default="saliency", help="saliency|pfi")
    ap.add_argument("--n_samples", type=int, default=32)
    args = ap.parse_args()
    main(args.config, args.encoder, args.head_path, args.method, args.n_samples)
