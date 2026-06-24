"""
Run ACL-tear inference on your knee MRI.

    python predict_acl.py

It loads your DICOM study, builds the sagittal/coronal/axial stacks, runs an
MRNet model on each plane that has trained weights, and combines them into a
single ACL-tear probability.

==========================  PLEASE READ  ===================================
This is NOT a medical diagnosis. The model is research code trained on a
different population/scanner than yours, so its output can be wrong in either
direction. Your real answer is the radiologist's report on THIS scan. Use this
only to learn and explore.
============================================================================

Weights: place trained files in the ./weights folder as:
    weights/acl_sagittal.pth
    weights/acl_coronal.pth
    weights/acl_axial.pth
See README.md for how to obtain or train them. Any subset works (e.g. sagittal
only); planes without weights are skipped.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from mri_pipeline import scan_study, load_volume, pick_series, to_mrnet_tensor
from mrnet_model import load_mrnet

HERE = os.path.dirname(os.path.abspath(__file__))   # acl_detector/src
PROJECT = os.path.dirname(HERE)                      # acl_detector
REPO = os.path.dirname(PROJECT)                      # Knee-MRI
DEFAULT_DICOM = os.path.join(REPO, "knee-mri-copy", "DICOM", "PAT001", "STUDY001")
WEIGHT_DIR = os.path.join(PROJECT, "models", "trained")
# How much to trust each plane when combining (sagittal is most informative
# for the ACL). Used only as a simple weighted average of probabilities; if you
# train a proper logistic-regression combiner, swap it in here.
PLANE_WEIGHTS = {"Sagittal": 0.6, "Coronal": 0.25, "Axial": 0.15}


def run_plane(study, plane: str, weight_path: str, device: str) -> float | None:
    series = pick_series(study, plane)
    if series is None:
        print(f"  {plane:9}: no series in study — skipped")
        return None
    vol = load_volume(series)
    tensor = to_mrnet_tensor(vol)               # (slices, 3, 256, 256)
    x = torch.from_numpy(tensor).unsqueeze(0).to(device)  # (1, slices, 3, 256, 256)
    model = load_mrnet(weight_path, device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    print(f"  {plane:9}: {series.description:24} -> ACL-tear prob {prob:.3f}")
    return prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", default=DEFAULT_DICOM)
    ap.add_argument("--weights", default=WEIGHT_DIR)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    study = scan_study(args.dicom)
    print(f"Loaded study: {len(study)} series. Device: {device}\n")

    weight_files = {
        "Sagittal": os.path.join(args.weights, "acl_sagittal.pth"),
        "Coronal": os.path.join(args.weights, "acl_coronal.pth"),
        "Axial": os.path.join(args.weights, "acl_axial.pth"),
    }
    available = {p: f for p, f in weight_files.items() if os.path.isfile(f)}

    if not available:
        print("No trained ACL weights found in ./%s - cannot produce a score.\n"
              "I won't print a fake number. To get a real score you need weights:\n"
              "  - train them with train_mrnet.py on the Stanford MRNet dataset, or\n"
              "  - drop a community .pth into ./%s (see README.md).\n"
              "\nIn the meantime, run  python export_images.py  and look at the\n"
              "sagittal slices yourself — the README explains what an ACL tear\n"
              "looks like." % (args.weights, args.weights))
        return

    print("Running ACL inference on planes with weights:\n")
    probs, wsum, combined = {}, 0.0, 0.0
    for plane, wfile in available.items():
        p = run_plane(study, plane, wfile, device)
        if p is not None:
            probs[plane] = p
            combined += PLANE_WEIGHTS[plane] * p
            wsum += PLANE_WEIGHTS[plane]

    if wsum:
        combined /= wsum
        print("\n" + "=" * 60)
        print(f"  Combined ACL-tear probability: {combined:.3f}")
        verdict = "SUGGESTS a tear" if combined >= 0.5 else "suggests NO tear"
        print(f"  Model {verdict} (threshold 0.5).")
        print("=" * 60)
        print("\n  REMINDER: research estimate only, NOT a diagnosis. Confirm with\n"
              "  your radiologist's report on this exact scan.")


if __name__ == "__main__":
    main()
