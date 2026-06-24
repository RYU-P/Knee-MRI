# Knee MRI — ACL tear grader
A small Python project for **anyone who has a knee MRI on disk as a DICOM study**.
It reads the study, prepares it for an MRNet-style model, lets you view the
images as ordinary pictures, and grades the ACL **no tear / partial / complete**.

### Why this exists
After my own knee MRI, I was stuck waiting forever on the radiologist's report
to find out whether I'd need surgery — so I went looking for a model that could
give me an early read. The problem: every pretrained knee-MRI model I could find
only predicts ACL tear **yes/no** — including the popular
[MRNet](https://stanfordmlgroup.github.io/projects/mrnet/) (Bien et al.,
Stanford). None of them separate a **partial** tear from a **complete** one,
which is basically the distinction that decides whether you need surgery. So I
built the model that didn't exist: a 3-class classifier that grades
**no tear / partial / complete**, trained on the
[KneeMRI dataset](https://zenodo.org/records/14789903) (Štajduhar et al.,
Clinical Hospital Centre Rijeka), the one source that actually labels each exam
healthy / partial / complete. That said, the grade is still the shakiest part
(very little complete-tear data for the model to learn from). So please use this
tool for satisfaction — THIS IS NOT A DIAGNOSIS! LOL.

If you got a CD/DVD copy from your clinic, the scan is almost certainly in
DICOM format (the medical-imaging standard — it's what these discs use),
which is exactly what this reads. Some discs also bundle a viewer or a few
exported JPEGs, but the real image data is the DICOM.

> ## This is not a diagnosis, use only for your satisfaction

## Highlights

| Area | What's here |
|------|-------------|
| **Real DICOM pipeline** | Reads a JPEG2000-compressed knee study, groups files into series, detects each plane from `ImageOrientationPatient`, and picks the best sequence per plane |
| **Partial vs complete grading** | `MRNet(num_classes=3)` trained on **KneeMRI (Rijeka)** — 917 sagittal PD-FS volumes labeled healthy / partial / complete |
| **Trainer + predictor** | `train_acl_grade.py` (stratified split, **balanced oversampling** for the 690/172/55 imbalance) and `predict_grade.py` (per-grade probabilities on any DICOM study) |
| **Honest engineering** | best balanced acc **0.449** (epoch 2); caught and fixed a mid-run collapse to all-healthy by switching from loss-weighting to balanced sampling |
| **Safe weight loading** | refuses checkpoints whose classifier head doesn't fit — no silent fake scores |
| **Example result** | grade model on the bundled scan → healthy **0.02** / partial **0.28** / complete **0.70** |

> Numbers above are a research/portfolio estimate, **not** a diagnosis — see the
> caveats under **Grade the tear** below.

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
python src/predict_grade.py  --dicom "D:/path/to/your/DICOM/PATxxx/STUDYxxx"
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
    │   ├── train_acl_grade.py    train the 3-class grade model on KneeMRI
    │   └── predict_grade.py      grade a study: healthy / partial / complete
    ├── models/
    │   └── trained/            acl_grade.pth from train_acl_grade.py (gitignored)
    ├── data/                   KneeMRI dataset (gitignored, CC BY-NC-ND)
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

# 3. Grade the tear (needs a trained model — see "Grade the tear" below)
python src/predict_grade.py
```

## Looking at the ACL yourself (no model needed)

Open `outputs/exported_images/Sagittal_*/` and scroll through the slices toward
the **middle/inner** part of the knee. The ACL is a **dark band running
diagonally** from the back-top (femur) to the front-bottom (tibia).

- **Normal ACL:** a continuous, taut, dark straight stripe.
- **Possible tear:** the band looks **wavy, blurry, thickened, or missing**,
  often with **bright (white) fluid signal** where the fibers should be.

A trained eye is far more reliable than the model below.

## Grade the tear: partial vs complete (`train_acl_grade.py` + `predict_grade.py`)

To get at the question that actually matters — **partial vs complete** — this
trains a 3-class model on the
[KneeMRI (Rijeka) dataset](https://zenodo.org/records/14789903), which labels
each exam `healthy` / `partial` / `complete`. KneeMRI is sagittal PD fat-sat,
matching a `SAG PD FS` series well.

```bash
# 1. download + extract KneeMRI into data/kneemri/  (metadata.csv + volumetric_data/*.pck)
# 2. train (CPU ~8 min/epoch; balanced oversampling handles the 690/172/55 imbalance)
python src/train_acl_grade.py --epochs 8
# 3. grade your scan
python src/predict_grade.py
```

`predict_grade.py` prints a probability for each grade, e.g. on the bundled scan: (my actual results lol I'm gonna need surgery)
```
  healthy       : 0.018
  partial tear  : 0.283
  complete tear : 0.699   <-- most likely
```

> ⚠️ **Read the limits.** The `complete` class has only **55 examples** in the
> whole dataset (~47 after the split), the model's balanced accuracy is a modest


**Step 1 — see what's in the scan.** The pipeline groups the loose DICOM files
into series and picks the best one per plane:

```bash
$ python src/mri_pipeline.py
Found 7 series in .../knee-mri-copy/DICOM/PAT001/STUDY001

   Series 8  [Axial]    'AX PD FS'            (50 slices)
   Series 13 [Coronal]  'COR PD DIXON FS_W'   (32 slices)
   Series 17 [Sagittal] 'SAG PD FS'           (35 slices)
   ...
Selected for ACL model:
  Sagittal : 'SAG PD FS'           (best plane for the ACL)
  Coronal  : 'COR PD DIXON FS_W'
  Axial    : 'AX PD FS'
```

**Step 2 — look at the images yourself** (always worth doing first):

```bash
$ python src/export_images.py
  Sagittal: exported 35 slices -> outputs/exported_images/Sagittal_SAG_PD_FS
  Coronal : exported 32 slices -> outputs/exported_images/Coronal_COR_PD_DIXON_FS_W
  Axial   : exported 50 slices -> outputs/exported_images/Axial_AX_PD_FS
```

**Step 3 — train the grade model on KneeMRI** (download + extract into
`data/kneemri/` first):

```bash
$ python src/train_acl_grade.py --epochs 8
Device: cpu.  Train 779  Val 138
Train class counts {'healthy': 586, 'partial': 146, 'complete': 47}
epoch  2  loss 0.8961  val_acc 0.601  bal_acc 0.449  recall(h/p/c) 0.73/0.12/0.50
   saved models/trained/acl_grade.pth (bal_acc 0.449)
...
Done. Best balanced accuracy 0.449.
```

**Step 4 — grade your scan:**

```bash
$ python src/predict_grade.py
Sagittal series: SAG PD FS  (35 slices)
========================================================
  healthy       : 0.018
  partial tear  : 0.283
  complete tear : 0.699   <-- most likely
========================================================
  Most likely grade: complete tear (69.9%)
```

So on this scan the model leans **complete**. But the grade is the least-reliable
step (see the limits above), and none of this replaces the radiologist's report.

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

- MRNet (architecture): Bien et al., *PLOS Medicine* 2018 —
  https://stanfordmlgroup.github.io/projects/mrnet/
- KneeMRI (partial/complete grades): Štajduhar et al., *CMPB* 2017 —
  https://zenodo.org/records/14789903
