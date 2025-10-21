import os, yaml, torch
from tqdm import tqdm
from xadl.utils import set_seed, get_device
from xadl.data import get_dataloaders
from xadl.models.encoder import build_encoder
from xadl.losses import nt_xent

def train(cfg_path: str, save_path: str = "artifacts/encoder.pt"):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("seed", 42))
    device = get_device()
    train_loader, val_loader, _ = get_dataloaders(cfg, for_contrastive=True)

    # infer input shapes
    first_batch = next(iter(train_loader))
    x1 = first_batch[0]
    if cfg["dataset"]["kind"].endswith("tabular"):
        in_dim = x1.shape[-1]
        enc = build_encoder(cfg, in_dim=in_dim).to(device)
    else:
        in_channels = x1.shape[1]
        enc = build_encoder(cfg, in_channels=in_channels).to(device)

    opt = torch.optim.Adam(enc.parameters(), lr=cfg["train"]["lr"])
    epochs = cfg["train"]["epochs"]
    temperature = cfg["train"]["temperature"]

    enc.train()
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        total = 0.0
        for a, b in pbar:
            a = a.to(device); b = b.to(device)
            z1 = enc(a); z2 = enc(b)
            loss = nt_xent(z1, z2, temperature=temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {ep+1}: avg loss = {total/len(train_loader):.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save({"state_dict": enc.state_dict(), "cfg": cfg}, save_path)
    print(f"Saved encoder to {save_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--save", default="artifacts/encoder.pt")
    args = ap.parse_args()
    train(args.config, args.save)
