"""
Run the 3-class ACL *grade* model (healthy / partial / complete) on a knee MRI
DICOM study and print a probability for each grade.

    python src/predict_grade.py
    python src/predict_grade.py --dicom "D:/path/to/DICOM/PATxxx/STUDYxxx"

Needs weights from train_acl_grade.py at models/trained/acl_grade.pth.

==========================  PLEASE READ  ===================================
This is NOT a medical diagnosis, and it especially does NOT tell you whether
you need surgery. The 'complete' class was trained on only 55 examples, so its
probability is the least reliable number here. It uses the SAGITTAL plane only
(that is what the KneeMRI training data is). Your real answer is the
radiologist's report on this exact scan.
============================================================================
"""
from __future__ import annotations

import argparse
import os

import torch

from mri_pipeline import scan_study, load_volume, pick_series, to_mrnet_tensor
from mrnet_model import load_mrnet

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REPO = os.path.dirname(PROJECT)
DEFAULT_DICOM = os.path.join(REPO, "knee-mri-copy", "DICOM", "PAT001", "STUDY001")
WEIGHTS = os.path.join(PROJECT, "models", "trained", "acl_grade.pth")
CLASSES = ["healthy", "partial tear", "complete tear"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", default=DEFAULT_DICOM)
    ap.add_argument("--weights", default=WEIGHTS)
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        raise SystemExit(
            f"No grade weights at {args.weights}. Train them first:\n"
            f"  python src/train_acl_grade.py   (after downloading KneeMRI)"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_mrnet(args.weights, device, num_classes=3)

    study = scan_study(args.dicom)
    series = pick_series(study, "Sagittal")
    if series is None:
        raise SystemExit("No sagittal series found in this study.")
    vol = load_volume(series)
    x = torch.from_numpy(to_mrnet_tensor(vol)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze(0).tolist()

    print(f"Sagittal series: {series.description}  ({len(series.files)} slices)\n")
    print("=" * 56)
    top = int(max(range(3), key=lambda i: probs[i]))
    for i, (name, p) in enumerate(zip(CLASSES, probs)):
        print(f"  {name:14}: {p:.3f}{'   <-- most likely' if i == top else ''}")
    print("=" * 56)
    print(f"\n  Most likely grade: {CLASSES[top]} ({probs[top]:.1%})")
    print("\n  RESEARCH ESTIMATE ONLY, NOT A DIAGNOSIS. The 'complete' class is")
    print("  data-poor (55 training cases) and this does NOT decide surgery.")
    print("  Confirm with the radiologist's report on this exact scan.")


if __name__ == "__main__":
    main()
