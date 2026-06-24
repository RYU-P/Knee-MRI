"""
Train an MRNet ACL model on the Stanford MRNet dataset, then save weights that
predict_acl.py can use on your own scan.

This is the legitimate way to get real weights. You need the MRNet dataset:
    https://stanfordmlgroup.github.io/competitions/mrnet/
(register, accept the research-use agreement, download ~6 GB, unzip).

Expected layout after download:
    MRNet-v1.0/
        train/axial/  train/coronal/  train/sagittal/   (0000.npy ... each (s,256,256))
        valid/...
        train-acl.csv   valid-acl.csv                    (case_id,label)

Usage (train the sagittal ACL model — the most useful one):
    python train_mrnet.py --data /path/to/MRNet-v1.0 --plane sagittal --epochs 15

A GPU helps a lot but it will run on CPU (slowly). Output -> weights/acl_<plane>.pth
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mrnet_model import MRNet
from mri_pipeline import IMAGENET_MEAN, IMAGENET_STD, TARGET_SIZE


class MRNetDataset(Dataset):
    def __init__(self, data_root: str, split: str, plane: str, limit: int | None = None):
        self.dir = os.path.join(data_root, split, plane)
        labels_csv = os.path.join(data_root, f"{split}-acl.csv")
        if not os.path.isdir(self.dir) or not os.path.isfile(labels_csv):
            raise FileNotFoundError(
                f"Expected MRNet layout under {data_root!r}: a '{split}/{plane}/' "
                f"folder of .npy stacks and a '{split}-acl.csv'. Got dir="
                f"{os.path.isdir(self.dir)}, csv={os.path.isfile(labels_csv)}. "
                f"Point --data at the unzipped MRNet-v1.0 folder."
            )
        self.labels = {}
        with open(labels_csv) as f:
            for row in csv.reader(f):
                if row:
                    self.labels[row[0]] = int(row[1])
        self.ids = sorted(self.labels)
        if limit:  # smoke-test: use a balanced-ish handful of cases
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        cid = self.ids[i]
        vol = np.load(os.path.join(self.dir, f"{cid}.npy")).astype(np.float32)
        # vol: (slices, 256, 256). Scale per-slice 0-255, 3ch, ImageNet-normalize.
        out = np.empty((vol.shape[0], 3, TARGET_SIZE, TARGET_SIZE), np.float32)
        for s, sl in enumerate(vol):
            lo, hi = sl.min(), sl.max()
            sl = (sl - lo) / (hi - lo + 1e-6) * 255.0
            rgb = np.stack([sl, sl, sl], 0)
            rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
            out[s] = rgb
        return torch.from_numpy(out), torch.tensor([self.labels[cid]], dtype=torch.float32)


def evaluate(model, loader, device) -> float:
    """Return AUC on a validation loader."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.sigmoid(model(x.to(device))).cpu().item()
            ps.append(p); ys.append(y.item())
    # simple AUC without sklearn
    pos = [p for p, y in zip(ps, ys) if y == 1]
    neg = [p for p, y in zip(ps, ys) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(1 for a in pos for b in neg if a == b)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to MRNet-v1.0 folder")
    ap.add_argument("--plane", default="sagittal", choices=["sagittal", "coronal", "axial"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "trained"))
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N cases per split (smoke test; "
                         "produces a throwaway, non-trustworthy model)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = MRNetDataset(args.data, "train", args.plane, args.limit)
    valid_ds = MRNetDataset(args.data, "valid", args.plane, args.limit)
    train_ld = DataLoader(train_ds, batch_size=1, shuffle=True)
    valid_ld = DataLoader(valid_ds, batch_size=1)

    pos = sum(train_ds.labels.values())
    pos_weight = torch.tensor([(len(train_ds) - pos) / max(pos, 1)], device=device)
    model = MRNet(pretrained_backbone=True).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)

    os.makedirs(args.out, exist_ok=True)
    best_auc, best_path = 0.0, os.path.join(args.out, f"acl_{args.plane}.pth")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_ld:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            total += loss.item()
        auc = evaluate(model, valid_ld, device)
        print(f"epoch {epoch:2d}  loss {total/len(train_ld):.4f}  val_auc {auc:.4f}")
        if auc == auc and auc > best_auc:  # auc==auc filters NaN
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"   saved {best_path} (auc {auc:.4f})")

    print(f"\nDone. Best val AUC {best_auc:.4f}. Weights -> {best_path}")
    print("Now run:  python predict_acl.py")


if __name__ == "__main__":
    main()
