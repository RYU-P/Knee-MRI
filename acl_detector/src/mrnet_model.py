"""
MRNet model (Bien et al., Stanford 2018) — the standard architecture for
detecting ACL tears / meniscus tears / abnormality from a single-plane knee
MRI stack.

One MRNet instance handles one plane (sagittal, coronal, or axial). The full
system runs three of them and combines their outputs. This file defines the
network; weights are loaded separately (see predict_acl.py / train_mrnet.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    """AlexNet feature extractor + slice-wise max pooling + linear classifier.

    num_classes=1 (default) -> a single logit for binary tear/no-tear
    (use torch.sigmoid). num_classes=3 -> grade logits for
    [healthy, partial, complete] (use softmax); see train_acl_grade.py.
    """

    def __init__(self, pretrained_backbone: bool = True, num_classes: int = 1):
        super().__init__()
        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        self.features = models.alexnet(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, slices, 3, 256, 256) — batch size is always 1 (one exam).
        x = x.squeeze(0)                  # (slices, 3, 256, 256)
        x = self.features(x)              # (slices, 256, 6, 6)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (slices, 256)
        x = torch.max(x, dim=0, keepdim=True)[0]  # (1, 256) max across slices
        return self.classifier(x)        # (1, num_classes) logits


def load_mrnet(weight_path: str | None, device: str = "cpu",
               num_classes: int = 1) -> MRNet:
    """Build an MRNet and optionally load trained weights from a .pth file.

    Loading is *verified*: a foreign checkpoint whose layer names don't match
    our architecture would otherwise be silently ignored by strict=False,
    leaving a randomly-initialised classifier head that still emits a
    confident-looking probability. That is worse than no number, so we check
    that the trained classifier weights actually made it in and raise if not.
    """
    model = MRNet(pretrained_backbone=weight_path is None, num_classes=num_classes)
    if weight_path:
        state = torch.load(weight_path, map_location=device)
        # Accept either a raw state_dict or a {'state_dict': ...} checkpoint.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # Tolerate common prefixes from other MRNet repos (e.g. "model.",
        # "module.", "pretrained_model.") by stripping them when the suffix
        # matches one of our parameter names.
        own = set(model.state_dict().keys())
        remapped = {}
        for k, v in state.items():
            if k in own:
                remapped[k] = v
                continue
            for prefix in ("model.", "module.", "pretrained_model.", "net."):
                if k.startswith(prefix) and k[len(prefix):] in own:
                    remapped[k[len(prefix):]] = v
                    break
        result = model.load_state_dict(remapped, strict=False)
        loaded = own - set(result.missing_keys)
        # The classifier head is what turns features into the ACL logit. If it
        # didn't load, the score would be meaningless — refuse rather than fake.
        head = {"classifier.weight", "classifier.bias"}
        if not head <= loaded:
            raise RuntimeError(
                f"Refusing to run: '{weight_path}' does not contain MRNet "
                f"classifier weights that fit this architecture "
                f"(matched {len(loaded)}/{len(own)} layers; classifier head "
                f"missing). These weights are not compatible — do not trust "
                f"any score from them. Retrain with train_mrnet.py."
            )
    model.to(device).eval()
    return model
