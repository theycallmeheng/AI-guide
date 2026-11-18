"""Local SegFormer inference used to estimate walkable sidewalk pixels."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


WALKABLE_LABELS = {
    "sidewalk",
    "walkway",
    "footpath",
    "pedestrian path",
    "pedestrian walkway",
}


class SegFormerWalkway:
    """Loads one Hugging Face SegFormer and returns a per-pixel sidewalk mask."""

    def __init__(self, model_dir: str | Path, device: str | int | None = None,
                 confidence: float = 0.45) -> None:
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.device = self._resolve_device(device, torch)
        self.model.to(self.device).eval()
        self.confidence = float(confidence)

        labels = self.model.config.id2label or {}
        self.labels = {int(class_id): str(label).strip().lower().replace("_", " ")
                       for class_id, label in labels.items()}
        self.walkable_ids = tuple(
            class_id for class_id, label in self.labels.items()
            if label in WALKABLE_LABELS
        )
        if not self.walkable_ids:
            raise ValueError(
                "SegFormer không có nhãn sidewalk/walkway/footpath trong id2label. "
                "Không thể suy ra vùng đi được an toàn từ model này."
            )

    @staticmethod
    def _resolve_device(device: str | int | None, torch: Any):
        if device is None or str(device).lower() == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        value = str(device).lower()
        if value == "cpu":
            return torch.device("cpu")
        if value.isdigit():
            value = f"cuda:{value}"
        if value.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(value)

    def predict(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a bool mask matching the input frame; True means sidewalk class."""
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = self.torch.nn.functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
            probabilities = logits.softmax(dim=1)
            walkable_probability = probabilities[:, self.walkable_ids].amax(dim=1)[0]
            predicted_class = probabilities.argmax(dim=1)[0]
            is_walkable_class = self.torch.zeros_like(predicted_class, dtype=self.torch.bool)
            for class_id in self.walkable_ids:
                is_walkable_class |= predicted_class == class_id
            mask = is_walkable_class & (walkable_probability >= self.confidence)
        return mask.to("cpu").numpy().astype(np.uint8)
