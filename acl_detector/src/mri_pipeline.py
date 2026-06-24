"""
MRI pipeline: load a DICOM knee-MRI study, group it into series, and turn the
sagittal / coronal / axial stacks into the format an MRNet-style ACL model wants.

Nothing in here is a diagnosis. It just reads your scan and prepares the data.
"""
from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# MRNet-style models expect square slices at this size.
TARGET_SIZE = 256
# ImageNet mean/std used by the AlexNet/ResNet backbone, scaled to 0-255.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255.0


@dataclass
class Series:
    number: int
    description: str
    plane: str  # "Sagittal" | "Coronal" | "Axial" | "Unknown"
    files: list[str] = field(default_factory=list)
    # populated by load_volume()
    volume: np.ndarray | None = None  # (slices, H, W) float32

    def __repr__(self) -> str:
        return (f"Series {self.number} [{self.plane}] '{self.description}' "
                f"({len(self.files)} slices)")


def _plane_from_orientation(iop) -> str:
    """Classify a slice's plane from ImageOrientationPatient."""
    if not iop or len(iop) < 6:
        return "Unknown"
    row = np.array(iop[:3], dtype=float)
    col = np.array(iop[3:6], dtype=float)
    normal = np.abs(np.cross(row, col))
    return ["Sagittal", "Coronal", "Axial"][int(normal.argmax())]


def scan_study(dicom_dir: str) -> dict[int, Series]:
    """Walk a DICOM folder and group files into series, sorted by slice position."""
    by_series: dict[int, Series] = {}
    pos: dict[int, list[tuple[float, str]]] = defaultdict(list)

    for name in os.listdir(dicom_dir):
        path = os.path.join(dicom_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if "PixelData" not in ds and not hasattr(ds, "Rows"):
            continue
        num = int(getattr(ds, "SeriesNumber", -1))
        if num not in by_series:
            by_series[num] = Series(
                number=num,
                description=str(getattr(ds, "SeriesDescription", "?")),
                plane=_plane_from_orientation(getattr(ds, "ImageOrientationPatient", None)),
            )
        # Sort key: position along slice normal, fall back to InstanceNumber.
        ipp = getattr(ds, "ImagePositionPatient", None)
        key = float(ipp[2]) if ipp else float(getattr(ds, "InstanceNumber", 0))
        pos[num].append((key, path))

    for num, series in by_series.items():
        series.files = [p for _, p in sorted(pos[num], key=lambda t: t[0])]
    return by_series


def load_volume(series: Series) -> np.ndarray:
    """Decode every slice (handles JPEG2000) into a (slices, H, W) float32 array."""
    slices = []
    for path in series.files:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)
        try:
            arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception:
            pass
        slices.append(arr)
    series.volume = np.stack(slices, axis=0)
    return series.volume


def _resize_center(img: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """Center-crop to square then resize to `size` using Pillow (no SciPy dep)."""
    from PIL import Image

    h, w = img.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    img = img[top:top + side, left:left + side]
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32)


def to_mrnet_tensor(volume: np.ndarray) -> np.ndarray:
    """
    Convert a raw (slices, H, W) volume into MRNet input: (slices, 3, 256, 256),
    per-slice min-max scaled to 0-255, replicated to 3 channels, then
    ImageNet-normalized. This matches the standard MRNet preprocessing.
    """
    out = np.empty((volume.shape[0], 3, TARGET_SIZE, TARGET_SIZE), dtype=np.float32)
    for i, sl in enumerate(volume):
        sl = _resize_center(sl)
        lo, hi = sl.min(), sl.max()
        sl = (sl - lo) / (hi - lo + 1e-6) * 255.0
        rgb = np.stack([sl, sl, sl], axis=0)  # (3, H, W)
        rgb = (rgb - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        out[i] = rgb
    return out


def pick_series(study: dict[int, Series], plane: str) -> Series | None:
    """
    Choose the best series for a plane for ACL work:
    prefer fat-saturated PD/T2 (best for ligament/edema signal).
    """
    candidates = [s for s in study.values() if s.plane == plane]
    if not candidates:
        return None
    # Prefer fat-suppressed fluid-sensitive images (best for ligament/edema).
    # For Dixon series, the water image (FS_W) is the useful one; demote the
    # fat (FS_F) and in/opposed-phase (FS_IN/FS_OPP) reconstructions.
    def score(s: Series) -> tuple:
        d = s.description.upper()
        is_fatsat = "FS" in d or "STIR" in d
        is_water = "FS_W" in d or "_W" in d
        is_bad_dixon = "FS_F" in d or "FS_IN" in d or "FS_OPP" in d
        return (is_fatsat, is_water, not is_bad_dixon, len(s.files))
    return sorted(candidates, key=score, reverse=True)[0]


if __name__ == "__main__":
    import sys
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        _repo, "knee-mri-copy", "DICOM", "PAT001", "STUDY001")
    study = scan_study(root)
    print(f"Found {len(study)} series in {root}\n")
    for num in sorted(study):
        print("  ", study[num])
    print("\nSelected for ACL model:")
    for plane in ("Sagittal", "Coronal", "Axial"):
        print(f"  {plane:9}:", pick_series(study, plane))
