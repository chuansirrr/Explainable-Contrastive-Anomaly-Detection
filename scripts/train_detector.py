import argparse
from xadl.training.detector_trainer import train_detector

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--encoder", default="artifacts/encoder.pt")
    ap.add_argument("--head", default=None, help="mahalanobis|knn|ocsvm")
    args = ap.parse_args()
    train_detector(args.config, args.encoder, args.head)
