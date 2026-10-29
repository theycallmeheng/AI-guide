"""Sample and polygon-label a video into an Ultralytics YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training_data import (  # noqa: E402
    CLASS_NAMES,
    CLASS_NAMES_VI,
    SPLITS,
    read_manifest,
    safe_identifier,
    validate_group_splits,
    write_manifest_row,
)


WINDOW_NAME = "Gán nhãn đa giác cho video"
CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 790
IMAGE_TOP = 74
IMAGE_AREA_HEIGHT = 650


def render(state: dict[str, Any]) -> np.ndarray:
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    frame = state["frame"]
    if frame is None:
        return canvas
    height, width = frame.shape[:2]
    scale = min((CANVAS_WIDTH - 24) / width, IMAGE_AREA_HEIGHT / height, 1.0)
    display_width = max(1, round(width * scale))
    display_height = max(1, round(height * scale))
    x_offset = (CANVAS_WIDTH - display_width) // 2
    y_offset = IMAGE_TOP + (IMAGE_AREA_HEIGHT - display_height) // 2
    preview = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_AREA)
    canvas[y_offset : y_offset + display_height, x_offset : x_offset + display_width] = preview

    def display_points(points: list[tuple[int, int]]) -> np.ndarray:
        return np.array(
            [[round(x * scale) + x_offset, round(y * scale) + y_offset] for x, y in points],
            dtype=np.int32,
        )

    for polygon in state["polygons"]:
        points = display_points(polygon["points"])
        if len(points) >= 2:
            cv2.polylines(canvas, [points.reshape(-1, 1, 2)], True, (70, 220, 100), 2)
        if len(points) >= 3:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [points.reshape(-1, 1, 2)], (40, 120, 40))
            cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
        cv2.putText(
            canvas,
            CLASS_NAMES_VI[polygon["class_id"]],
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )

    active_points = display_points(state["current_points"])
    if len(active_points) >= 1:
        for point in active_points:
            cv2.circle(canvas, tuple(point), 4, (0, 220, 255), -1)
        if len(active_points) >= 2:
            cv2.polylines(canvas, [active_points.reshape(-1, 1, 2)], False, (0, 220, 255), 2)

    cv2.putText(
        canvas,
        f"{state['video_name']} | frame {state['frame_index']} | {state['split']} | {state['location_id']}",
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Nhãn {state['class_id']}: {CLASS_NAMES[state['class_id']]} / {CLASS_NAMES_VI[state['class_id']]} | đa giác: {len(state['polygons'])}",
        (14, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    if state["message"]:
        cv2.putText(
            canvas,
            state["message"],
            (14, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "Chuột trái: thêm đỉnh | chuột phải: khép đa giác | [ ]: đổi nhãn | u: lùi đỉnh | x: xóa đa giác",
        (14, 749),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "s: lưu khung hình (kể cả âm tính) | n: bỏ qua | q: kết thúc",
        (14, 773),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    state["scale"] = scale
    state["x_offset"] = x_offset
    state["y_offset"] = y_offset
    state["display_width"] = display_width
    state["display_height"] = display_height
    return canvas


def mouse_callback(event: int, x: int, y: int, _flags: int, state: dict[str, Any]) -> None:
    if state["frame"] is None:
        return
    image_x = x - state.get("x_offset", 0)
    image_y = y - state.get("y_offset", 0)
    if not (0 <= image_x < state.get("display_width", 0)):
        return
    if not (0 <= image_y < state.get("display_height", 0)):
        return
    scale = state["scale"]
    frame_height, frame_width = state["frame"].shape[:2]
    point = (
        min(frame_width - 1, max(0, round(image_x / scale))),
        min(frame_height - 1, max(0, round(image_y / scale))),
    )
    if event == cv2.EVENT_LBUTTONDOWN:
        state["current_points"].append(point)
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(state["current_points"]) >= 3:
            state["polygons"].append(
                {"class_id": state["class_id"], "points": state["current_points"][:]}
            )
            state["current_points"].clear()
        else:
            state["message"] = "Một đa giác cần ít nhất 3 đỉnh."


def label_text(polygons: list[dict[str, Any]], width: int, height: int) -> str:
    rows: list[str] = []
    for polygon in polygons:
        points = polygon["points"]
        if len(points) < 3:
            continue
        normalized = [
            (min(1.0, max(0.0, x / width)), min(1.0, max(0.0, y / height)))
            for x, y in points
        ]
        coordinates = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
        rows.append(f"{polygon['class_id']} {coordinates}")
    return "\n".join(rows) + ("\n" if rows else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gán nhãn đa giác cho các lớp chướng ngại và bề mặt trên video."
    )
    parser.add_argument("video", type=Path, help="Video đã ghi bằng góc camera dự kiến")
    parser.add_argument("--session-id", required=True, help="Mã phiên quay, ví dụ walk_001")
    parser.add_argument("--location-id", required=True, help="Mã địa điểm/tuyến, ví dụ route_A")
    parser.add_argument("--split", choices=sorted(SPLITS), required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "training")
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Khoảng cách giữa các khung hình gợi ý; nên rà soát để tránh ảnh gần trùng.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = args.video.resolve()
    if not video_path.is_file():
        raise SystemExit(f"Không tìm thấy video: {video_path}")
    if args.sample_seconds <= 0:
        raise SystemExit("--sample-seconds phải lớn hơn 0.")

    try:
        session_id = safe_identifier(args.session_id, "session_id")
        location_id = safe_identifier(args.location_id, "location_id")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    dataset_root = args.dataset.resolve()
    manifest_path = dataset_root / "session_manifest.csv"
    records = read_manifest(manifest_path)
    proposed = records + [
        {
            "session_id": session_id,
            "location_id": location_id,
            "split": args.split,
        }
    ]
    try:
        validate_group_splits(proposed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Không mở được video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frame_step = max(1, round(fps * args.sample_seconds))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Nhãn gán được:")
    for index, (name, vi_name) in enumerate(zip(CLASS_NAMES, CLASS_NAMES_VI)):
        print(f"  {index:2d}: {name} — {vi_name}")
    print("Dùng nhiều đa giác cho mỗi khung hình. Gán vùng đường/lối đi và mọi nguy hiểm nhìn thấy.")
    print("Ghi video có quyền sử dụng; tránh lưu hoặc phát tán hình ảnh nhận dạng cá nhân.")

    state: dict[str, Any] = {
        "frame": None,
        "frame_index": 0,
        "video_name": video_path.name,
        "split": args.split,
        "location_id": location_id,
        "class_id": 0,
        "polygons": [],
        "current_points": [],
        "message": "",
    }
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_WIDTH, CANVAS_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, state)
    frame_index = 0
    saved_frames = 0

    def read_sample(target: int) -> bool:
        if total_frames > 0 and target >= total_frames:
            return False
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if not ok:
            return False
        state["frame"] = frame
        state["frame_index"] = target
        state["polygons"].clear()
        state["current_points"].clear()
        state["message"] = ""
        return True

    if not read_sample(frame_index):
        capture.release()
        cv2.destroyAllWindows()
        raise SystemExit("Video không có khung hình đọc được.")

    try:
        while True:
            cv2.imshow(WINDOW_NAME, render(state))
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                break
            if key == ord("["):
                state["class_id"] = (state["class_id"] - 1) % len(CLASS_NAMES)
            elif key == ord("]"):
                state["class_id"] = (state["class_id"] + 1) % len(CLASS_NAMES)
            elif key == ord("u"):
                if state["current_points"]:
                    state["current_points"].pop()
            elif key == ord("x"):
                if state["polygons"]:
                    state["polygons"].pop()
            elif key == ord("c"):
                state["current_points"].clear()
            elif key == ord("n"):
                frame_index += frame_step
                if not read_sample(frame_index):
                    break
            elif key == ord("s"):
                if state["current_points"]:
                    state["message"] = "Khép đa giác bằng chuột phải trước khi lưu."
                    print(state["message"])
                    continue
                frame = state["frame"]
                frame_height, frame_width = frame.shape[:2]
                image_rel = Path("images") / args.split / session_id / f"frame_{frame_index:08d}.jpg"
                label_rel = Path("labels") / args.split / session_id / f"frame_{frame_index:08d}.txt"
                image_path = dataset_root / image_rel
                label_path = dataset_root / label_rel
                if image_path.exists() or label_path.exists():
                    state["message"] = "Tệp khung hình đã tồn tại; hãy dùng mã phiên quay mới."
                    print(state["message"])
                    continue
                image_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(image_path), frame):
                    state["message"] = f"Không ghi được ảnh: {image_path}"
                    print(state["message"])
                    continue
                label_path.write_text(
                    label_text(state["polygons"], frame_width, frame_height), encoding="utf-8"
                )
                write_manifest_row(
                    manifest_path,
                    {
                        "session_id": session_id,
                        "location_id": location_id,
                        "split": args.split,
                        "source_video": video_path.name,
                        "frame_index": str(frame_index),
                        "image_path": image_rel.as_posix(),
                        "object_count": str(len(state["polygons"])),
                    },
                )
                saved_frames += 1
                print(
                    f"Đã lưu frame {frame_index}: {len(state['polygons'])} vùng; "
                    f"tổng {saved_frames} khung hình."
                )
                frame_index += frame_step
                if not read_sample(frame_index):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print(f"Hoàn tất: đã lưu {saved_frames} khung hình vào {dataset_root}")
    if saved_frames == 0:
        print("Chưa có khung hình nào được lưu; dữ liệu huấn luyện chưa thay đổi.")


if __name__ == "__main__":
    main()
