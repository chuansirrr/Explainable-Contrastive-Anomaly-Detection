import argparse
from xadl.training.contrastive_trainer import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--save", default="artifacts/encoder.pt")
    args = ap.parse_args()
    train(args.config, args.save)
