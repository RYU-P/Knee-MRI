 """
Pull pretrained MRNet weights (or the MRNet dataset) from Kaggle.

There is no single official pretrained ACL weight file, but some Kaggle users
publish trained MRNet weights as Kaggle *datasets*. Quality is unverified, so
treat any resulting score with extra skepticism.

Setup (one time):
    1. Make a free Kaggle account.
    2. Account -> Settings -> "Create New API Token" -> downloads kaggle.json.
    3. Put kaggle.json at  C:\\Users\\<you>\\.kaggle\\kaggle.json
    pip install kaggle    (already in requirements.txt)

Find a weights dataset on kaggle.com (search e.g. "mrnet weights" / "mrnet acl
pretrained"), copy its slug (owner/dataset-name), then:

    python fetch_kaggle_weights.py --dataset <owner>/<dataset-name>

It downloads + unzips into ./kaggle_download. Move any acl_*.pth files it
contains into ./weights and run predict_acl.py. Rename them to match:
    weights/acl_sagittal.pth  weights/acl_coronal.pth  weights/acl_axial.pth
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

OUT = "kaggle_download"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="Kaggle dataset slug, e.g. someuser/mrnet-weights")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    if not args.dataset:
        print(__doc__)
        print("\nNo --dataset given. Search kaggle.com for a weights dataset first.")
        return

    try:
        import kaggle  # noqa: F401  (validates credentials on import)
    except Exception as e:
        print("Kaggle API not ready:", e)
        print("Install with `pip install kaggle` and place kaggle.json in ~/.kaggle/")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", args.out, "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDownloaded to {args.out}. Move any acl_*.pth into ./weights/ then run predict_acl.py")


if __name__ == "__main__":
    main()
