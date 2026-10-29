"""Validate the grouped video dataset and fine-tune a YOLO segmentation model."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_data import (  # noqa: E402
    CLASS_NAMES,
    read_manifest,
    validate_group_splits,
    write_dataset_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO segmentation for video-based walkway hazard alerts."
    )
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "training")
    parser.add_argument("--weights", default="yolo11n-seg.pt", help="Pretrained segmentation weights")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="", help="Empty means let Ultralytics choose CPU/GPU")
    parser.add_argument("--run-name", help="Optional training run name")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "models" / "training_runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "walkway_obstacles.pt",
        help="Where to copy the selected best checkpoint",
    )
    return parser.parse_args()


def inspect_dataset(dataset_root: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = dataset_root / "session_manifest.csv"
    if not manifest_path.is_file():
        raise ValueError(
            f"Không thấy {manifest_path}. Hãy tạo video và gán nhãn bằng "
            "tools/annotate_training_video.py trước."
        )
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError("Manifest chưa có khung hình được gán nhãn.")
    validate_group_splits(records)

    image_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    location_counts: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row_number, row in enumerate(records, start=2):
        split = row["split"]
        image_rel = Path(row["image_path"])
        if (
            image_rel.is_absolute()
            or ".." in image_rel.parts
            or len(image_rel.parts) < 3
            or image_rel.parts[0] != "images"
            or image_rel.parts[1] != split
        ):
            raise ValueError(
                f"Manifest dòng {row_number}: image_path phải nằm trong images/{split}/."
            )
        image_path = dataset_root / image_rel
        label_rel = Path("labels") / Path(*image_rel.parts[1:]).with_suffix(".txt")
        label_path = dataset_root / label_rel
        if not image_path.is_file() or not label_path.is_file():
            raise ValueError(
                f"Manifest dòng {row_number}: thiếu ảnh hoặc nhãn cho {image_rel.as_posix()}."
            )
        image_counts[split] += 1
        location_counts[split].add(row["location_id"])
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                raise ValueError(
                    f"Nhãn {label_path}, dòng {line_number}: cần class_id và ít nhất 3 cặp tọa độ."
                )
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Nhãn không phải số hợp lệ: {label_path}, dòng {line_number}.") from exc
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(f"Nhãn có class_id ngoài danh sách: {label_path}, dòng {line_number}.")
            if any(value < 0 or value > 1 for value in coordinates):
                raise ValueError(
                    f"Tọa độ phải nằm trong khoảng 0..1: {label_path}, dòng {line_number}."
                )
            object_counts[split] += 1

    for split in ("train", "val"):
        if not image_counts[split]:
            raise ValueError(f"Tập {split} đang trống.")
        if not object_counts[split]:
            raise ValueError(
                f"Tập {split} chưa có đa giác dương tính; cần gán nhãn ít nhất một vùng nguy hiểm/ngữ cảnh."
            )

    dataset_yaml = dataset_root / "dataset.yaml"
    write_dataset_yaml(dataset_root, dataset_yaml)
    summary = {
        split: {
            "images": image_counts[split],
            "polygons": object_counts[split],
            "locations": sorted(location_counts[split]),
        }
        for split in ("train", "val", "test")
    }
    return dataset_yaml, summary


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        raise SystemExit("epochs, imgsz, batch và workers phải là các giá trị hợp lệ.")
    dataset_root = args.dataset.resolve()
    try:
        dataset_yaml, summary = inspect_dataset(dataset_root)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Dữ liệu chưa sẵn sàng: {exc}") from exc

    print("Tóm tắt dữ liệu theo địa điểm:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["test"]["images"]:
        print("Chưa có tập test độc lập; sau khi huấn luyện, hãy bổ sung video ở địa điểm giữ riêng.")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Thiếu Ultralytics/PyTorch. Kích hoạt .venv và chạy: pip install -r requirements.txt"
        ) from exc

    run_name = args.run_name or f"walkway_seg_{datetime.now():%Y%m%d_%H%M%S}"
    runs_dir = args.runs_dir.resolve()
    run_dir = runs_dir / run_name
    if run_dir.exists():
        raise SystemExit(f"Thư mục lần train đã tồn tại: {run_dir}. Hãy chọn --run-name khác.")
    runs_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device or None,
        project=str(runs_dir),
        name=run_name,
        exist_ok=False,
        patience=15,
        seed=42,
    )

    best_checkpoint = run_dir / "weights" / "best.pt"
    if not best_checkpoint.is_file():
        raise SystemExit(f"Không tìm thấy trọng số tốt nhất sau huấn luyện: {best_checkpoint}")
    shutil.copy2(best_checkpoint, args.output.resolve())
    print(f"Đã lưu mô hình: {args.output.resolve()}")

    if summary["test"]["images"]:
        print("Đánh giá một lần trên tập test giữ riêng…")
        test_model = YOLO(str(args.output.resolve()))
        metrics = test_model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device or None,
            workers=args.workers,
        )
        result_values = getattr(metrics, "results_dict", {})
        print("Kết quả test:")
        print(json.dumps(result_values, ensure_ascii=False, indent=2, default=str))
    print("Prototype nghiên cứu: kết quả mô hình không bảo đảm lối đi an toàn.")


if __name__ == "__main__":
    main()
