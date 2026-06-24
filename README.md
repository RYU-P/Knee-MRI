# Knee MRI — ACL tear explorer
A small Python project for **anyone who has a knee MRI on disk as a DICOM study**.
It reads the study, prepares it for an MRNet-style model, lets you view the
images as ordinary pictures, and runs ACL-tear inference.

### Why this exists
I recently tore my ACL, and waiting on the radiologist's report was taking
forever — so I built this to take a look myself in the meantime. The catch I
only realized later: the available trained models predict ACL tear **yes/no**,
but the MRNet labels don't distinguish a **partial** tear from a **complete**
one — and complete-vs-partial is basically the thing that decides whether you
need surgery. So this can hint that *something* is wrong with the ACL, but it
**can't** tell you how bad it is or whether to operate. Lesson learned: it's a
curiosity tool, not a substitute for the report. 🙃

If you got a CD/DVD copy from your clinic, the scan is almost certainly in
**DICOM** format (the medical-imaging standard — it's what these discs use),
which is exactly what this reads. Some discs also bundle a viewer or a few
exported JPEGs, but the real image data is the DICOM.

> ## This is not a diagnosis, use only for your satisfaction

## What you need
A knee MRI as a **DICOM study** — a folder of image files (named `1`, `2`, `3`…
or `*.dcm`) from a hospital CD/DVD or download. JPEG2000-compressed is supported.

Files don't need sorting: the pipeline reads each file's DICOM tags, groups them
into series, detects the plane from image orientation, and auto-picks the most
useful series per plane (favoring fat-suppressed, fluid-sensitive sequences).

### Point it at your data

Put your study under `knee-mri-copy/` (replacing the bundled example), **or** pass
its path on the command line — the scripts that read DICOM accept `--dicom`:

```bash
python src/export_images.py --dicom "D:/path/to/your/DICOM/PATxxx/STUDYxxx"
python src/predict_acl.py   --dicom "D:/path/to/your/DICOM/PATxxx/STUDYxxx"
```

## Project layout

```
Knee-MRI/
├── README.md
├── knee-mri-copy/              a knee DICOM study (swap in your own)
└── acl_detector/
    ├── requirements.txt
    ├── src/                    all Python (run scripts from here)
    │   ├── mri_pipeline.py       load DICOM, group series, build tensors
    │   ├── mrnet_model.py        the MRNet network (Bien et al., 2018)
    │   ├── export_images.py      save slices as PNGs for human viewing
    │   ├── predict_acl.py        run your own trained weights, combine planes
    │   ├── train_mrnet.py        train weights on the Stanford MRNet dataset
    │   ├── run_alberto.py        run a community 3-plane model (the quick route)
    │   └── fetch_kaggle_weights.py
    ├── models/
    │   ├── trained/            your own weights from train_mrnet.py (gitignored)
    │   ├── community/          alberto_mrnet.pth — third-party model (gitignored)
    │   └── downloads/          anything pulled from Kaggle
    └── outputs/                generated PNGs and result text files (gitignored)
```

Scripts compute paths relative to themselves, so they work from any directory.
The examples below assume you're in `acl_detector/`.

## Setup

```bash
cd acl_detector
pip install -r requirements.txt
```

(PyTorch, pydicom, and a JPEG2000 decoder — most knee DICOM is JPEG2000-compressed.)

## Use it

```bash
# 1. See your series and which ones were auto-selected per plane
python src/mri_pipeline.py

# 2. Export PNGs you can actually look at (start here!)
python src/export_images.py
#    -> outputs/exported_images/Sagittal_.../slice_00.png ... open and scroll

# 3a. Quick score from a community model (no training needed)
python src/run_alberto.py        # prints to screen AND writes outputs/acl_result.txt

# 3b. Score from your OWN trained weights (needs train_mrnet.py first)
python src/predict_acl.py
```

> Note: `run_alberto.py` currently runs on the study in `knee-mri-copy/`. To use a
> different study with it, swap your data into `knee-mri-copy/` or edit the
> `DICOM` path near the top of the file.

## Looking at the ACL yourself (no model needed)

Open `outputs/exported_images/Sagittal_*/` and scroll through the slices toward
the **middle/inner** part of the knee. The ACL is a **dark band running
diagonally** from the back-top (femur) to the front-bottom (tibia).

- **Normal ACL:** a continuous, taut, dark straight stripe.
- **Possible tear:** the band looks **wavy, blurry, thickened, or missing**,
  often with **bright (white) fluid signal** where the fibers should be.

A trained eye is far more reliable than any model below.

## Two ways to get a score

### A. Community model — fast, unverified (`run_alberto.py`)

`models/community/alberto_mrnet.pth` is a third-party 3-plane model
([AlbertoUAH](https://github.com/AlbertoUAH/Knee-Lesions-Classification-via-Deep-Learning)),
trained on the Stanford MRNet data. `run_alberto.py` runs it on all three planes
and prints abnormal / ACL / meniscus probabilities.

Caveats: unverified quality, trained on specific MRI sequences that may differ
from yours, and it's a *pickled full model* loaded via a restricted unpickler
(only audited torch classes are allowed). A toy sanity-check, **not** a
trustworthy result.

### B. Train your own — legitimate, defensible (`train_mrnet.py`)

Get the Stanford MRNet dataset (free, requires research-use registration at
https://stanfordmlgroup.github.io/competitions/mrnet/), then:

```bash
python src/train_mrnet.py --data /path/to/MRNet-v1.0 --plane sagittal --epochs 15
# tip: add --limit 20 for a ~1-minute smoke test before the full (slow CPU) run
```

Weights land in `models/trained/acl_<plane>.pth`. Then `python src/predict_acl.py`
loads them. If no weights are present, `predict_acl.py` refuses to invent a number.
`mrnet_model.py` also **rejects** any checkpoint whose classifier head doesn't fit,
so a mismatched `.pth` can never silently produce a fake score.

## How plane/series selection works (for any study)

`mri_pipeline.py` decides the plane of each series from its
`ImageOrientationPatient` tag (no reliance on series names), then `pick_series`
ranks candidates per plane, preferring fat-suppressed / fluid-sensitive images.
As a concrete example, on the bundled study it selects:

| Plane | Selected series | Why |
|-------|-----------------|-----|
| Sagittal | `SAG PD FS` | **Best plane for the ACL** — the ligament is seen along its length |
| Coronal | `COR PD DIXON FS_W` | Fat-suppressed water image — shows fluid/edema |
| Axial | `AX PD FS` | Cross-section, helps confirm |

Your study's series names will differ; the selection logic is the same.

## Credit / further reading

- MRNet: Bien et al., *PLOS Medicine* 2018 —
  https://stanfordmlgroup.github.io/projects/mrnet/
- Community model: https://github.com/AlbertoUAH/Knee-Lesions-Classification-via-Deep-Learning
