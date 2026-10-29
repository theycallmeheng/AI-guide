"""Shared class taxonomy and split checks for the training workflow."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable


CLASS_NAMES = [
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus_truck",
    "dog",
    "pole_tree",
    "barrier",
    "walkway",
    "road",
    "curb_step",
    "pothole_drain",
]

CLASS_NAMES_VI = [
    "người",
    "xe đạp",
    "xe máy",
    "ô tô",
    "xe buýt hoặc xe tải",
    "chó",
    "cột hoặc cây",
    "vật chắn",
    "lối đi",
    "đường xe chạy",
    "lề đường hoặc bậc",
    "hố hoặc miệng cống",
]

SPLITS = {"train", "val", "test"}
MANIFEST_FIELDS = [
    "session_id",
    "location_id",
    "split",
    "source_video",
    "frame_index",
    "image_path",
    "object_count",
]


def safe_identifier(value: str, field: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    if not cleaned:
        raise ValueError(f"{field} không hợp lệ; hãy dùng mã ngắn không chứa đường dẫn.")
    return cleaned


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def validate_group_splits(records: Iterable[dict[str, str]]) -> None:
    """Reject session or location identifiers appearing in multiple splits."""
    session_splits: dict[str, tuple[str, str]] = {}
    location_splits: dict[str, str] = {}
    for row_number, record in enumerate(records, start=2):
        split = record.get("split", "").strip()
        session_id = record.get("session_id", "").strip()
        location_id = record.get("location_id", "").strip()
        if split not in SPLITS:
            raise ValueError(f"Manifest dòng {row_number}: split phải là train, val hoặc test.")
        if not session_id or not location_id:
            raise ValueError(f"Manifest dòng {row_number}: thiếu session_id hoặc location_id.")
        previous = session_splits.setdefault(session_id, (split, location_id))
        if previous[0] != split:
            raise ValueError(
                f"Rò rỉ dữ liệu: phiên quay {session_id!r} nằm trong cả {previous[0]} và {split}."
            )
        if previous[1] != location_id:
            raise ValueError(
                f"Phiên quay {session_id!r} được gán cho hai địa điểm khác nhau: "
                f"{previous[1]!r} và {location_id!r}."
            )
        previous = location_splits.setdefault(location_id, split)
        if previous != split:
            raise ValueError(
                f"Rò rỉ dữ liệu: địa điểm {location_id!r} nằm trong cả {previous} và {split}. "
                "Hãy giữ toàn bộ video của một địa điểm trong cùng một tập."
            )


def write_manifest_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    has_rows = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        if not has_rows:
            writer.writeheader()
        writer.writerow(row)


def write_dataset_yaml(dataset_root: Path, output_path: Path) -> None:
    """Write a small YAML file without requiring a YAML library at setup time."""
    root = dataset_root.resolve().as_posix()
    lines = [
        f'path: "{root}"',
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f'  {index}: "{name}"' for index, name in enumerate(CLASS_NAMES))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
