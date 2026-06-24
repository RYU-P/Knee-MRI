"""
Train a 3-class ACL *grade* model — healthy / partial / complete — on the
KneeMRI (Rijeka) dataset. Unlike the Stanford MRNet labels (binary tear/no-tear),
KneeMRI distinguishes a PARTIAL tear from a COMPLETE rupture, which is the
distinction that actually matters for the surgery question.

Data: https://zenodo.org/records/14789903   (CC BY-NC-ND 4.0 — NON-COMMERCIAL)
Expected layout (default: <project>/data/kneemri):
    metadata.csv
    volumetric_data/<examId>-<seriesNo>.pck     numpy (slices,320,320) uint16

Usage:
    python src/train_acl_grade.py                 # full data, default epochs
    python src/train_acl_grade.py --limit 60      # quick smoke test
    python src/train_acl_grade.py --epochs 15

Output -> models/trained/acl_grade.pth

NOTE: only 55 of 917 exams are complete ruptures (6%). The loss is class-weighted
and the split is stratified, but the 'complete' class is still data-poor, so its
predictions are the least reliable. Research estimate only — NOT a diagnosis.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from mri_pipeline import to_mrnet_tensor
from mrnet_model import MRNet

HERE = os.path.dirname(os.path.abspath(__file__))   # acl_detector/src
PROJECT = os.path.dirname(HERE)                      # acl_detector
DATA_DIR = os.path.join(PROJECT, "data", "kneemri")
OUT_PATH = os.path.join(PROJECT, "models", "trained", "acl_grade.pth")
CLASSES = ["healthy", "partial", "complete"]         # aclDiagnosis 0 / 1 / 2


def read_metadata(data_dir: str) -> list[tuple[str, int]]:
    """Return [(volume_path, label), ...] for volumes that exist on disk."""
    vol_dir = os.path.join(data_dir, "volumetric_data")
    rows = []
    with open(os.path.join(data_dir, "metadata.csv")) as f:
        for r in csv.DictReader(f):
            path = os.path.join(vol_dir, r["volumeFilename"])
            if os.path.isfile(path):
                rows.append((path, int(r["aclDiagnosis"])))
    return rows


def stratified_split(rows, val_frac=0.15, seed=0):
    """Split per-class so every class (esp. the rare 'complete') is in both sets."""
    rng = random.Random(seed)
    by_cls = {0: [], 1: [], 2: []}
    for item in rows:
        by_cls[item[1]].append(item)
    train, val = [], []
    for cls, items in by_cls.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac)))
        val += items[:n_val]
        train += items[n_val:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


class KneeMRIDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        with open(path, "rb") as fh:
            vol = pickle.load(fh).astype(np.float32)   # (slices, 320, 320)
        tensor = to_mrnet_tensor(vol)                  # (slices, 3, 256, 256)
        return torch.from_numpy(tensor), torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device):
    """Return (overall_acc, balanced_acc, confusion 3x3)."""
    model.eval()
    conf = np.zeros((3, 3), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))               # (1, 3)
            pred = int(logits.argmax(dim=1).item())
            conf[int(y.item()), pred] += 1
    correct = np.trace(conf)
    total = conf.sum()
    per_cls_recall = [conf[c, c] / conf[c].sum() if conf[c].sum() else float("nan")
                      for c in range(3)]
    bal = np.nanmean(per_cls_recall)
    return correct / total, bal, conf, per_cls_recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None, help="use only N exams (smoke test)")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = read_metadata(args.data)
    if not rows:
        raise SystemExit(
            f"No volumes found under {args.data!r}. Expected metadata.csv and "
            f"volumetric_data/*.pck — download/extract the KneeMRI dataset first."
        )
    if args.limit:
        # keep it stratified-ish even when limiting
        rows = stratified_split(rows, val_frac=0.0, seed=args.seed)[0][:args.limit]
    train_rows, val_rows = stratified_split(rows, seed=args.seed)

    counts = np.bincount([l for _, l in train_rows], minlength=3)
    print(f"Device: {device}.  Train {len(train_rows)}  Val {len(val_rows)}")
    print(f"Train class counts {dict(zip(CLASSES, counts.tolist()))}\n")

    # Balanced oversampling: each epoch draws roughly equal healthy/partial/
    # complete exams, so the rare 'complete' class can't be ignored. (Loss
    # weighting alone let the model collapse to always predicting 'healthy'.)
    per_class_w = 1.0 / np.maximum(counts, 1)
    sample_w = [per_class_w[l] for _, l in train_rows]
    sampler = WeightedRandomSampler(sample_w, num_samples=len(train_rows), replacement=True)
    train_ld = DataLoader(KneeMRIDataset(train_rows), batch_size=1, sampler=sampler)
    val_ld = DataLoader(KneeMRIDataset(val_rows), batch_size=1)

    model = MRNet(pretrained_backbone=True, num_classes=3).to(device)
    crit = nn.CrossEntropyLoss()                      # sampler already balances
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_bal = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_ld:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
            total += loss.item()
        acc, bal, conf, recall = evaluate(model, val_ld, device)
        print(f"epoch {epoch:2d}  loss {total/len(train_ld):.4f}  "
              f"val_acc {acc:.3f}  bal_acc {bal:.3f}  "
              f"recall(h/p/c) {recall[0]:.2f}/{recall[1]:.2f}/{recall[2]:.2f}")
        if bal == bal and bal > best_bal:           # bal==bal filters NaN
            best_bal = bal
            torch.save({"state_dict": model.state_dict(),
                        "classes": CLASSES, "balanced_acc": float(bal)}, args.out)
            print(f"   saved {os.path.relpath(args.out, PROJECT)} (bal_acc {bal:.3f})")

    print(f"\nDone. Best balanced accuracy {best_bal:.3f}. Weights -> {args.out}")
    print("Run:  python src/predict_grade.py")


if __name__ == "__main__":
    main()
