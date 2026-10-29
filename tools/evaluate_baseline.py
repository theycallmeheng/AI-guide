"""Evaluate the app's detector and alert policy on a video with event annotations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from obstacle_detector import FrameAnalysis, Obstacle, analyze_frame


def load_annotations(path: Path, video: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    resolved_video = video.resolve()
    total_rows = 0
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        for index, row in enumerate(csv.DictReader(file)):
            total_rows += 1
            try:
                row_video = Path(row["video"]).resolve()
                start = float(row["start_sec"])
                end = float(row["end_sec"])
            except (KeyError, ValueError, OSError) as exc:
                raise ValueError(f"Dòng nhãn {index + 2} không hợp lệ: {exc}") from exc
            if row_video != resolved_video and row_video.name != resolved_video.name:
                continue
            if end <= start:
                raise ValueError(f"Dòng nhãn {index + 2}: end_sec phải lớn hơn start_sec.")
            row["_id"] = index
            row["_start"] = start
            row["_end"] = end
            rows.append(row)
    if total_rows and not rows:
        raise ValueError(
            f"CSV không có nhãn cho video {resolved_video.name}; hãy chạy label_video.py với đúng tệp."
        )
    return rows


def event_matches(event: dict[str, Any], obstacle: Obstacle) -> bool:
    if event["class_name"].strip().lower() != obstacle.class_name.strip().lower():
        return False
    direction = event.get("direction", "").strip()
    band = event.get("distance_band", "").strip()
    if direction and direction != obstacle.direction:
        return False
    if band and band != obstacle.distance_band:
        return False
    return True


def closest_obstacle(analysis: FrameAnalysis) -> Obstacle | None:
    priority = {"gần": 0, "vừa": 1, "xa": 2}
    return min(
        analysis.obstacles,
        key=lambda item: (priority[item.distance_band], -item.confidence),
        default=None,
    )


def percentile95(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, 95))


def evaluate(
    video_path: Path,
    annotation_path: Path,
    model_path: str,
    confidence: float,
    sample_interval: float,
) -> dict[str, Any]:
    from ultralytics import YOLO

    video_path = video_path.resolve()
    annotations = load_annotations(annotation_path, video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if frame_count > 0 else 0.0
    model = YOLO(model_path)
    event_hits: set[int] = set()
    last_alert: tuple[str, float] | None = None
    last_unclear_at = 0.0
    alerts: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    decoded_frames = 0
    sampled_frames = 0
    unclear_frames = 0
    next_sample_at = 0.0
    last_video_sec = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            decoded_frames += 1
            timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp <= 0.0:
                timestamp = (decoded_frames - 1) / max(fps, 1.0)
            last_video_sec = max(last_video_sec, timestamp)
            if timestamp + 1e-6 < next_sample_at:
                continue
            next_sample_at = timestamp + sample_interval
            sampled_frames += 1

            started = time.perf_counter()
            analysis = analyze_frame(frame, model, confidence)
            inference_ms.append((time.perf_counter() - started) * 1000.0)
            if analysis.unclear:
                unclear_frames += 1

            active_events = [
                event
                for event in annotations
                if event["_start"] <= timestamp <= event["_end"]
            ]
            candidate = closest_obstacle(analysis)
            alert_type = ""
            emitted_obstacle: Obstacle | None = None
            if candidate is not None and (not analysis.unclear or candidate.distance_band == "gần"):
                key = f"{candidate.class_name}|{candidate.distance_band}|{candidate.direction}"
                cooldown = 3.0 if candidate.distance_band == "gần" else 5.0
                if last_alert is None or last_alert[0] != key or timestamp - last_alert[1] >= cooldown:
                    last_alert = (key, timestamp)
                    alert_type = "obstacle"
                    emitted_obstacle = candidate
            if not alert_type and analysis.unclear and timestamp - last_unclear_at >= 5.0:
                last_unclear_at = timestamp
                alert_type = "unclear"

            if alert_type == "obstacle" and emitted_obstacle is not None:
                matching = [event for event in active_events if event_matches(event, emitted_obstacle)]
                for event in matching:
                    event_hits.add(event["_id"])
                alerts.append(
                    {
                        "time_sec": round(timestamp, 3),
                        "type": "obstacle",
                        "class_name": emitted_obstacle.class_name,
                        "direction": emitted_obstacle.direction,
                        "distance_band": emitted_obstacle.distance_band,
                        "confidence": round(emitted_obstacle.confidence, 4),
                        "matched_event_ids": [event["_id"] for event in matching],
                        "false_warning": not bool(matching),
                    }
                )
            elif alert_type == "unclear":
                alerts.append(
                    {
                        "time_sec": round(timestamp, 3),
                        "type": "unclear",
                        "reason": analysis.unclear_reason,
                    }
                )
    finally:
        cap.release()

    effective_duration = duration or last_video_sec
    class_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"events": 0, "hits": 0})
    for event in annotations:
        class_name = event["class_name"].strip().lower()
        class_totals[class_name]["events"] += 1
        if event["_id"] in event_hits:
            class_totals[class_name]["hits"] += 1

    false_alerts = sum(1 for alert in alerts if alert["type"] == "obstacle" and alert["false_warning"])
    object_alerts = sum(1 for alert in alerts if alert["type"] == "obstacle")
    result = {
        "video": str(video_path),
        "annotations": str(annotation_path.resolve()),
        "model": model_path,
        "confidence": confidence,
        "sample_interval_sec": sample_interval,
        "duration_sec": round(effective_duration, 3),
        "decoded_frames": decoded_frames,
        "sampled_frames": sampled_frames,
        "unclear_frames": unclear_frames,
        "event_count": len(annotations),
        "event_hits": len(event_hits),
        "event_recall": round(len(event_hits) / len(annotations), 4) if annotations else None,
        "per_class_event_recall": {
            name: {
                "events": values["events"],
                "hits": values["hits"],
                "recall": round(values["hits"] / values["events"], 4) if values["events"] else None,
            }
            for name, values in sorted(class_totals.items())
        },
        "object_alerts": object_alerts,
        "false_object_alerts": false_alerts,
        "false_alerts_per_5_min": round(false_alerts / effective_duration * 300, 4) if effective_duration else None,
        "p95_inference_ms": round(percentile95(inference_ms), 2),
        "alerts": alerts,
        "notes": [
            "Mỗi nhãn là một khoảng thời gian có vật cản; các khoảng không có nhãn được xem là đã rà soát và không có mục tiêu.",
            "Độ trễ ở đây là thời gian chạy nhận diện trên khung hình đã lấy mẫu; đo độ trễ TTS trực tiếp trong ứng dụng.",
            "Chỉ số này là baseline nghiên cứu, không chứng minh ứng dụng an toàn khi đi bộ.",
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Đo baseline YOLO trên video có nhãn sự kiện.")
    parser.add_argument("--video", type=Path, required=True, help="Video đã gắn nhãn")
    parser.add_argument("--annotations", type=Path, required=True, help="CSV do label_video.py tạo")
    parser.add_argument("--model", required=True, help="Tệp trọng số Ultralytics, ví dụ yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--sample-interval", type=float, default=0.5, help="Khoảng cách lấy mẫu khung hình theo giây")
    parser.add_argument("--report", type=Path, help="JSON đầu ra; mặc định reports/baseline_<video>.json")
    args = parser.parse_args()
    if args.sample_interval <= 0:
        parser.error("--sample-interval phải lớn hơn 0")
    result = evaluate(args.video, args.annotations, args.model, args.confidence, args.sample_interval)
    report_path = args.report or Path("reports") / f"baseline_{args.video.stem}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "alerts"}, ensure_ascii=False, indent=2))
    print(f"Báo cáo đầy đủ: {report_path.resolve()}")


if __name__ == "__main__":
    main()
