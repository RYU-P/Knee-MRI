"""
Run AlbertoUAH's pretrained 3-plane knee model on Ryuji's DICOM study.

RESEARCH ESTIMATE ONLY — NOT A DIAGNOSIS. These are an unofficial third-party's
weights of unverified quality, trained on a different population/scanner, using
PD sequences here that differ from the model's training sequences. Treat the
number as a toy sanity-check, not a clinical result.
"""
import sys, pickle, os, warnings
warnings.filterwarnings("ignore")   # hush torch's benign version/source warnings
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# this dir on path so we can reuse the verified DICOM pipeline regardless of CWD.
HERE = os.path.dirname(os.path.abspath(__file__))   # acl_detector/src
PROJECT = os.path.dirname(HERE)                      # acl_detector
REPO = os.path.dirname(PROJECT)                      # Knee-MRI
sys.path.insert(0, HERE)
from mri_pipeline import scan_study, pick_series, load_volume, _resize_center, TARGET_SIZE

MODEL_PATH = os.path.join(PROJECT, "models", "community", "alberto_mrnet.pth")
DICOM = os.path.join(REPO, "knee-mri-copy", "DICOM", "PAT001", "STUDY001")
OUT_FILE = os.path.join(PROJECT, "outputs", "acl_result.txt")
LABELS = ["Abnormal", "ACL Tear", "Meniscus Tear"]  # their LABEL_DICT order


# --- their CNNModel, vendored verbatim so the pickle can rebuild it ----------
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.axial = models.alexnet(weights=None).features
        self.sagittal = models.alexnet(weights=None).features
        self.coronal = models.alexnet(weights=None).features
        self.features_conv_axial = self.axial[:12]
        self.features_conv_sagittal = self.sagittal[:12]
        self.features_conv_coronal = self.coronal[:12]
        self.max_pool = nn.MaxPool2d(3, 2)
        self.avg_pool_axial = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_sagittal = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_coronal = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(3 * 256, 3))

    def forward(self, x):
        imgs = [torch.squeeze(i, dim=0) for i in x]
        f = [self.features_conv_axial(imgs[0]),
             self.features_conv_sagittal(imgs[1]),
             self.features_conv_coronal(imgs[2])]
        pools = [self.avg_pool_axial, self.avg_pool_sagittal, self.avg_pool_coronal]
        outs = []
        for img, pool in zip(f, pools):
            img = self.max_pool(img)
            img = pool(img).view(img.size(0), -1)
            outs.append(torch.max(img, dim=0, keepdim=True)[0])
        return self.fc(torch.cat(outs, dim=1))


# --- defense-in-depth: only allow the globals we statically audited ----------
ALLOWED = {
    ("__main__", "CNNModel"), ("collections", "OrderedDict"),
    ("torch", "FloatStorage"),
    ("torch._utils", "_rebuild_parameter"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch.nn.modules.activation", "ReLU"), ("torch.nn.modules.container", "Sequential"),
    ("torch.nn.modules.conv", "Conv2d"), ("torch.nn.modules.linear", "Linear"),
    ("torch.nn.modules.pooling", "AdaptiveAvgPool2d"),
    ("torch.nn.modules.pooling", "MaxPool2d"),
}


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"BLOCKED non-audited global: {module}.{name}")
        return super().find_class(module, name)


class _PickleShim:            # torch.load(pickle_module=...) wants Unpickler+load
    Unpickler = RestrictedUnpickler
    load = staticmethod(pickle.load)
    @staticmethod
    def loads(*a, **k):
        return pickle.loads(*a, **k)


def prep_plane(study, plane):
    """Build their model input for one plane: (slices,3,256,256), 0..1, 3ch.
    Matches their prepare_data (img/255, 3-channel) — NO ImageNet normalization."""
    s = pick_series(study, plane)
    vol = load_volume(s)
    out = np.empty((vol.shape[0], 3, TARGET_SIZE, TARGET_SIZE), np.float32)
    for i, sl in enumerate(vol):
        sl = _resize_center(sl)
        lo, hi = sl.min(), sl.max()
        sl = (sl - lo) / (hi - lo + 1e-6)          # 0..1 (their /255 of a 0..255 img)
        out[i] = np.stack([sl, sl, sl], 0)
    return torch.from_numpy(out), s.description


def main():
    print("Loading AlbertoUAH weights via restricted unpickler...")
    model = torch.load(MODEL_PATH, map_location="cpu",
                       weights_only=False, pickle_module=_PickleShim)
    model.eval()
    print("  loaded OK (only audited classes were instantiated)\n")

    study = scan_study(DICOM)
    order = ["Axial", "Sagittal", "Coronal"]       # their forward() input order
    x, descs = [], {}
    for plane in order:
        t, d = prep_plane(study, plane)
        x.append(t); descs[plane] = d
        print(f"  {plane:9}: {d:24} {tuple(t.shape)}")

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits.squeeze())

    print("\n" + "=" * 60)
    for name, p in zip(LABELS, probs.tolist()):
        star = "  <-- ACL" if name == "ACL Tear" else ""
        print(f"  {name:14}: {p:.3f}{star}")
    acl = probs[1].item()
    print("=" * 60)
    print(f"\n  ACL-tear probability (AlbertoUAH model): {acl:.3f}  "
          f"-> model {'SUGGESTS a tear' if acl >= 0.5 else 'suggests NO tear'}")
    print("\n  RESEARCH ESTIMATE ONLY, NOT A DIAGNOSIS. Unverified third-party")
    print("  weights, sequence/scanner mismatch. Confirm with the radiologist's")
    print("  report on this exact scan.")


class _Tee:
    """Send everything printed to both the screen and the output file."""
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        for st in self.streams: st.write(s)
    def flush(self):
        for st in self.streams: st.flush()


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        real_stdout = sys.stdout
        sys.stdout = _Tee(real_stdout, f)
        try:
            main()
        finally:
            sys.stdout = real_stdout
    print(f"\n(results also written to {os.path.relpath(OUT_FILE, PROJECT)})")
