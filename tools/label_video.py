"""Review a walking video and save time intervals containing target hazards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


LABELS = [
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "backpack",
    "suitcase",
    "curb",
    "stairs",
    "pothole",
    "manhole",
    "drain",
    "pole",
    "barrier",
    "cone",
    "obstacle",
]
DIRECTIONS = ["bất kỳ", "phía trước", "bên trái", "bên phải"]
BANDS = ["bất kỳ", "gần", "vừa", "xa"]
FIELDS = ("video", "start_sec", "end_sec", "class_name", "direction", "distance_band")


def save_annotations(output: Path, records: list[dict[str, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def label_video(video_path: Path, output: Path) -> None:
    video_path = video_path.resolve()
    output = output.resolve()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    delay_ms = max(1, round(1000 / min(max(fps, 1.0), 60.0)))
    label_index = 0
    direction_index = 1
    band_index = 0
    paused = True
    ok, current_frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Video không có khung hình đọc được: {video_path}")
    current_sec = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    pending: dict[str, str] | None = None
    notice = "Pause video; [ = start hazard, ] = end and save"
    records: list[dict[str, str]] = []

    if output.exists():
        with output.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                try:
                    existing_video = Path(row["video"]).resolve()
                except (KeyError, OSError):
                    continue
                if existing_video == video_path:
                    records.append({field: row.get(field, "") for field in FIELDS})

    window = "Gắn nhãn video — phím tắt hiện trên khung hình"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    notice = "End of video — press s to save or q to quit"
                    paused = True
                else:
                    current_frame = frame
                    current_sec = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)

            if current_frame is not None:
                display = current_frame.copy()
                current_direction = DIRECTIONS[direction_index]
                current_band = BANDS[band_index]
                direction_ui = {
                    "bất kỳ": "any",
                    "phía trước": "front",
                    "bên trái": "left",
                    "bên phải": "right",
                }[current_direction]
                band_ui = {"bất kỳ": "any", "gần": "near", "vừa": "mid", "xa": "far"}[current_band]
                lines = [
                    f"Time {current_sec:7.2f}s | {'PAUSED' if paused else 'PLAYING'} | {notice}",
                    f"Class [{label_index + 1}/{len(LABELS)}]: {LABELS[label_index]} | d direction: {direction_ui} | b band: {band_ui}",
                    f"Events saved: {len(records)} | n/p class | j/l +/-1s | J/L +/-5s | u undo | s save | q quit",
                ]
                for row_index, line in enumerate(lines):
                    y = 28 + row_index * 27
                    cv2.rectangle(display, (0, y - 21), (display.shape[1], y + 5), (15, 15, 15), -1)
                    cv2.putText(display, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                if pending is not None:
                    cv2.putText(
                        display,
                        f"OPEN: {pending['class_name']} from {pending['start_sec']}s",
                        (10, display.shape[0] - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 180, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window, display)

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
            if key == ord("q"):
                save_annotations(output, records)
                break
            if key == ord("s"):
                save_annotations(output, records)
                notice = f"Saved {len(records)} events to {output.name}"
            elif key == ord(" "):
                paused = not paused
                notice = "Paused" if paused else "Playing"
            elif key == ord("n"):
                label_index = (label_index + 1) % len(LABELS)
            elif key == ord("p"):
                label_index = (label_index - 1) % len(LABELS)
            elif key == ord("d"):
                direction_index = (direction_index + 1) % len(DIRECTIONS)
            elif key == ord("b"):
                band_index = (band_index + 1) % len(BANDS)
            elif key == ord("j"):
                paused = True
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, current_sec - 1.0) * 1000.0)
                ok, sought_frame = cap.read()
                if ok:
                    current_frame = sought_frame
                    current_sec = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                notice = "Seeked back 1 second"
            elif key == ord("l"):
                paused = True
                cap.set(cv2.CAP_PROP_POS_MSEC, (current_sec + 1.0) * 1000.0)
                ok, sought_frame = cap.read()
                if ok:
                    current_frame = sought_frame
                    current_sec = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                notice = "Seeked forward 1 second"
            elif key == ord("J"):
                paused = True
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, current_sec - 5.0) * 1000.0)
                ok, sought_frame = cap.read()
                if ok:
                    current_frame = sought_frame
                    current_sec = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                notice = "Seeked back 5 seconds"
            elif key == ord("L"):
                paused = True
                cap.set(cv2.CAP_PROP_POS_MSEC, (current_sec + 5.0) * 1000.0)
                ok, sought_frame = cap.read()
                if ok:
                    current_frame = sought_frame
                    current_sec = max(0.0, float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
                notice = "Seeked forward 5 seconds"
            elif key == ord("["):
                paused = True
                if pending is not None:
                    notice = "An event is already open; press ] to close it first"
                else:
                    direction = DIRECTIONS[direction_index]
                    band = BANDS[band_index]
                    pending = {
                        "video": str(video_path),
                        "start_sec": f"{current_sec:.3f}",
                        "end_sec": "",
                        "class_name": LABELS[label_index],
                        "direction": "" if direction == "bất kỳ" else direction,
                        "distance_band": "" if band == "bất kỳ" else band,
                    }
                    notice = f"Started {pending['class_name']} at {current_sec:.2f}s; press ] when it leaves the hazard area"
            elif key == ord("]"):
                paused = True
                if pending is None:
                    notice = "No open event; press [ to mark a hazard start"
                elif current_sec <= float(pending["start_sec"]):
                    notice = "End must be after start; seek ahead and press ] again"
                else:
                    pending["end_sec"] = f"{current_sec:.3f}"
                    records.append(pending)
                    pending = None
                    save_annotations(output, records)
                    notice = f"Saved event; {len(records)} total"
            elif key == ord("u"):
                if pending is not None:
                    pending = None
                    notice = "Open event cancelled"
                elif records:
                    records.pop()
                    save_annotations(output, records)
                    notice = f"Removed last event; {len(records)} remain"
    finally:
        cap.release()
        cv2.destroyWindow(window)

    print(f"Đã lưu {len(records)} sự kiện: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gắn nhãn khoảng thời gian có chướng ngại trong video.")
    parser.add_argument("video", type=Path, help="Đường dẫn video cần gắn nhãn")
    parser.add_argument("--output", type=Path, help="CSV đầu ra; mặc định <video>_annotations.csv")
    args = parser.parse_args()
    output = args.output or args.video.with_name(f"{args.video.stem}_annotations.csv")
    label_video(args.video, output)


if __name__ == "__main__":
    main()
