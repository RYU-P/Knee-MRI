"""
Export your knee MRI as ordinary PNG images so you can look at them yourself.

The ACL (anterior cruciate ligament) is seen best on the SAGITTAL series: it
runs as a dark band from the back of the femur to the front of the tibia. A
normal ACL is a continuous straight dark stripe; a tear often shows up as a
wavy/absent/blurred band with bright (fluid) signal around it.

Usage:
    python export_images.py                     # exports all selected planes
    python export_images.py --plane Sagittal    # one plane only
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from mri_pipeline import scan_study, load_volume, pick_series, _resize_center

HERE = os.path.dirname(os.path.abspath(__file__))   # acl_detector/src
PROJECT = os.path.dirname(HERE)                      # acl_detector
REPO = os.path.dirname(PROJECT)                      # Knee-MRI
DEFAULT_DICOM = os.path.join(REPO, "knee-mri-copy", "DICOM", "PAT001", "STUDY001")
OUT_DIR = os.path.join(PROJECT, "outputs", "exported_images")


def export_series(series, out_subdir: str) -> int:
    os.makedirs(out_subdir, exist_ok=True)
    vol = load_volume(series)
    for i, sl in enumerate(vol):
        sl = _resize_center(sl, size=512)  # bigger for human viewing
        lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
        sl = np.clip((sl - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
        Image.fromarray(sl.astype(np.uint8)).save(
            os.path.join(out_subdir, f"slice_{i:02d}.png"))
    return len(vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", default=DEFAULT_DICOM)
    ap.add_argument("--plane", choices=["Sagittal", "Coronal", "Axial"], default=None)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    study = scan_study(args.dicom)
    planes = [args.plane] if args.plane else ["Sagittal", "Coronal", "Axial"]
    for plane in planes:
        series = pick_series(study, plane)
        if not series:
            print(f"  {plane}: no series found")
            continue
        sub = os.path.join(args.out, f"{plane}_{series.description.replace(' ', '_')}")
        n = export_series(series, sub)
        print(f"  {plane}: exported {n} slices -> {sub}")
    print(f"\nDone. Open the '{args.out}' folder and scroll the Sagittal slices to find the ACL.")
    print("Tip: the ACL is usually clearest on the middle sagittal slices.")


if __name__ == "__main__":
    main()
