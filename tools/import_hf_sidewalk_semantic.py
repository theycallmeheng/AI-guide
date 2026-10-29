"""Import the gated sidewalk-semantic dataset as YOLO-seg training data.

This converts mapped semantic mask regions into approximate polygons. The source
is semantic segmentation, so touching instances of the same class can merge.
Review the generated overlays before using them for research training.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_data import read_manifest, write_manifest_row  # noqa: E402

DATASET_ID = "segments/sidewalk-semantic"
# Source IDs follow the dataset card's semantic taxonomy. Unmapped classes are ignored.
SOURCE_TO_TARGET = {
    1: 9,   # flat-road -> road
    2: 8,   # flat-sidewalk -> walkway
    7: 10,  # flat-curb -> curb_step
    8: 0,   # human-person -> person
    9: 0,   # human-rider -> person
    10: 3,  # vehicle-car -> car
    11: 4,  # vehicle-truck -> bus_truck
    12: 4,  # vehicle-bus -> bus_truck
    14: 2,  # vehicle-motorcycle -> motorcycle
    15: 1,  # vehicle-bicycle -> bicycle
    21: 7,  # construction-fenceguardrail -> barrier
    24: 10, # construction-stairs -> curb_step
    25: 6,  # object-pole -> pole_tree
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert segments/sidewalk-semantic into train-only YOLO segmentation labels."
    )
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "data" / "training")
    parser.add_argument("--split", default="train", help="The HF dataset has one split; use train only.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 imports all rows.")
    parser.add_argument("--min-area", type=int, default=24, help="Ignore smaller connected mask regions.")
    parser.add_argument("--epsilon-ratio", type=float, default=0.002, help="Polygon simplification ratio.")
    return parser.parse_args()


def image_array(value: Any, mode: str) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.asarray(value.convert(mode))
    if isinstance(value, dict):
        if value.get("path"):
            with Image.open(value["path"]) as image:
                return np.asarray(image.convert(mode))
        if value.get("bytes"):
            import io

            with Image.open(io.BytesIO(value["bytes"])) as image:
                return np.asarray(image.convert(mode))
    return np.asarray(value)


def mask_index_array(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        raw = np.asarray(value)
    elif isinstance(value, dict):
        if value.get("path"):
            with Image.open(value["path"]) as image:
                raw = np.asarray(image)
        elif value.get("bytes"):
            import io

            with Image.open(io.BytesIO(value["bytes"])) as image:
                raw = np.asarray(image)
        else:
            raise ValueError("Không đọc được ảnh mask từ trường label.")
    else:
        raw = np.asarray(value)
    if raw.ndim == 3:
        if raw.shape[2] < 3 or not np.array_equal(raw[..., 0], raw[..., 1]) or not np.array_equal(raw[..., 0], raw[..., 2]):
            raise ValueError("Mask không phải ảnh chỉ số lớp đơn kênh; cần kiểm tra cách mã hóa mask.")
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"Kích thước mask không hợp lệ: {raw.shape}")
    return raw.astype(np.int32, copy=False)


def polygons_from_mask(
    mask: np.ndarray, class_id: int, min_area: int, epsilon_ratio: float
) -> list[str]:
    binary = np.ascontiguousarray(mask.astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = binary.shape
    lines: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = max(0.5, epsilon_ratio * cv2.arcLength(contour, True))
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(polygon) < 3:
            continue
        values: list[str] = [str(class_id)]
        for x, y in polygon:
            values.extend((f"{min(max(float(x) / width, 0.0), 1.0):.6f}", f"{min(max(float(y) / height, 0.0), 1.0):.6f}"))
        lines.append(" ".join(values))
    return lines


def main() -> None:
    args = parse_args()
    if args.max_samples < 0 or args.min_area < 1 or args.epsilon_ratio <= 0:
        raise SystemExit("max-samples, min-area và epsilon-ratio phải dương hoặc bằng 0 phù hợp.")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Thiếu thư viện datasets. Cài bằng: pip install datasets") from exc

    token = os.environ.get("HF_TOKEN")
    try:
        dataset = load_dataset(DATASET_ID, split=args.split, token=token)
    except Exception as exc:
        raise SystemExit(
            "Không tải được sidewalk-semantic. Hãy đăng nhập Hugging Face, chấp nhận điều kiện truy cập "
            "CC BY-NC 4.0 trên trang dataset, rồi đặt HF_TOKEN trong môi trường nếu cần. "
            f"Chi tiết: {exc}"
        ) from exc

    columns = set(dataset.column_names)
    image_column = "pixel_values" if "pixel_values" in columns else "image" if "image" in columns else None
    mask_column = "label" if "label" in columns else "segmentation_mask" if "segmentation_mask" in columns else None
    if image_column is None or mask_column is None:
        raise SystemExit(f"Schema dataset chưa được hỗ trợ; các cột hiện có: {sorted(columns)}")

    root = args.dataset_root.resolve()
    image_dir = root / "images" / "train" / "hf_sidewalk_semantic"
    label_dir = root / "labels" / "train" / "hf_sidewalk_semantic"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "session_manifest.csv"
    existing_rows = read_manifest(manifest_path)
    existing_paths = {row.get("image_path", "") for row in existing_rows}

    total = len(dataset) if args.max_samples == 0 else min(len(dataset), args.max_samples)
    imported = 0
    total_polygons = 0
    session_id = "hf_sidewalk_semantic_train"
    location_id = "hf_sidewalk_semantic"
    for index in range(total):
        sample = dataset[index]
        image = image_array(sample[image_column], "RGB")
        mask = mask_index_array(sample[mask_column])
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        relative_image = Path("images") / "train" / "hf_sidewalk_semantic" / f"frame_{index:06d}.jpg"
        relative_label = Path("labels") / "train" / "hf_sidewalk_semantic" / f"frame_{index:06d}.txt"
        image_path = root / relative_image
        label_path = root / relative_label
        if relative_image.as_posix() in existing_paths:
            continue

        polygons: list[str] = []
        for source_id, target_id in SOURCE_TO_TARGET.items():
            polygons.extend(
                polygons_from_mask(mask == source_id, target_id, args.min_area, args.epsilon_ratio)
            )

        if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Không ghi được ảnh: {image_path}")
        label_path.write_text("\n".join(polygons) + ("\n" if polygons else ""), encoding="utf-8")
        write_manifest_row(
            manifest_path,
            {
                "session_id": session_id,
                "location_id": location_id,
                "split": "train",
                "source_video": f"hf://datasets/{DATASET_ID}/{args.split}",
                "frame_index": str(index),
                "image_path": relative_image.as_posix(),
                "object_count": str(len(polygons)),
            },
        )
        imported += 1
        total_polygons += len(polygons)

    print(f"Đã thêm {imported} ảnh train, {total_polygons} polygon từ {DATASET_ID}.")
    print("Không thay đổi validation/test; kiểm tra overlay mask trước khi train.")
    print("Ghi công và điều kiện CC BY-NC 4.0 phải được giữ trong bản nghiên cứu.")


if __name__ == "__main__":
    main()
