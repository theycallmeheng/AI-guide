"""Camera capture and single-frame obstacle analysis for the laptop prototype."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from performance import Measurements
from guidance import AlertPolicy


HAZARD_NAMES = {
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "suitcase",
    "backpack",
    "handbag",
    "umbrella",
    "skateboard",
    "traffic light",
    "stop sign",
    "curb",
    "stairs",
    "stair",
    "step",
    "steps",
    "pothole",
    "manhole",
    "drain",
    "pole",
    "barrier",
    "bollard",
    "cone",
    "obstacle",
    "tree",
    "bus truck",
    "pole tree",
    "curb step",
    "pothole drain",
}

VIETNAMESE_NAMES = {
    "bike": "xe đạp", "electrical pole": "cột điện", "dustbin": "thùng rác",
    "guard rail": "rào chắn", "traffic cone": "cọc tiêu", "fire hydrant": "trụ nước",
    "teraffic barrel": "thùng chắn", "plant pot": "chậu cây",
    "electrical box": "tủ điện", "bicycle rack": "giá để xe",
    "person": "người",
    "bicycle": "xe đạp",
    "motorcycle": "xe máy",
    "car": "ô tô",
    "bus": "xe buýt",
    "truck": "xe tải",
    "dog": "chó",
    "chair": "ghế",
    "bench": "ghế dài",
    "suitcase": "va li",
    "backpack": "ba lô",
    "handbag": "túi xách",
    "umbrella": "ô",
    "skateboard": "ván trượt",
    "traffic light": "đèn giao thông",
    "stop sign": "biển dừng",
    "curb": "lề đường",
    "stairs": "bậc thang",
    "stair": "bậc thang",
    "step": "bậc thềm",
    "steps": "bậc thềm",
    "pothole": "ổ gà",
    "manhole": "miệng cống",
    "drain": "rãnh thoát nước",
    "pole": "cột",
    "barrier": "vật chắn",
    "bollard": "cọc chắn",
    "cone": "cọc tiêu",
    "obstacle": "vật cản",
    "tree": "cây",
    "bus truck": "xe buýt hoặc xe tải",
    "pole tree": "cột hoặc cây",
    "curb step": "lề đường hoặc bậc",
    "pothole drain": "hố hoặc miệng cống",
}


@dataclass(frozen=True)
class Obstacle:
    class_name: str
    confidence: float
    direction: str
    distance_band: str
    box: tuple[int, int, int, int]
    distance_m: float | None = None  # RGB sources never provide a measured distance.


@dataclass(frozen=True)
class FrameAnalysis:
    frame: np.ndarray
    obstacles: tuple[Obstacle, ...]
    unclear: bool
    unclear_reason: str
    blur_score: float
    clear_corridors: tuple[str, ...] = ()


@lru_cache(maxsize=8)
def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font that can render Vietnamese labels on Windows and Linux."""
    windows_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    candidates = (
        os.path.join(windows_fonts, "segoeui.ttf"),
        os.path.join(windows_fonts, "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_unicode_labels(
    frame: np.ndarray,
    labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]],
) -> np.ndarray:
    """Draw Vietnamese labels through Pillow because cv2.putText is not Unicode-safe."""
    if not labels:
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_size = max(14, min(24, round(frame.shape[0] * 0.03)))
    font = _label_font(font_size)
    padding = max(3, font_size // 5)
    width, height = image.size

    for text, (x, y), bgr_color in labels:
        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        box_width = min(width, text_width + 2 * padding)
        box_height = min(height, text_height + 2 * padding)
        x = max(0, min(x, width - box_width))
        y = max(0, min(y, height - box_height))

        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        background = tuple(int(channel * 0.65) for channel in rgb_color)
        draw.rectangle(
            (x, y, x + box_width, y + box_height),
            fill=background,
            outline=rgb_color,
            width=1,
        )
        draw.text(
            (x + padding, y + padding),
            text,
            font=font,
            fill=(255, 255, 255),
        )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


class LatestFrame:
    """A one-frame handoff so inference always works on the newest camera image."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._captured_at = 0.0
        self._sequence = 0
        self._closed = False
        self.measurements = Measurements()

    def publish(self, frame: np.ndarray) -> None:
        with self._condition:
            self._frame = frame
            self._captured_at = time.perf_counter()
            self._sequence += 1
            self.measurements.add()
            self._condition.notify_all()

    def next_after(
        self, sequence: int, timeout: float = 0.25
    ) -> tuple[int, np.ndarray | None, bool, float]:
        with self._condition:
            if self._sequence == sequence and not self._closed:
                self._condition.wait(timeout)
            if self._sequence == sequence:
                return sequence, None, self._closed, self._captured_at
            return self._sequence, self._frame, self._closed, self._captured_at

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


def is_hazard_class(class_name: str) -> bool:
    normalized = class_name.strip().lower().replace("_", " ")
    if normalized in HAZARD_NAMES:
        return True
    if normalized in {"bike", "dustbin", "guard rail", "traffic cone", "fire hydrant",
                      "teraffic barrel", "plant pot", "electrical box", "bicycle rack"}:
        return True
    # Allow reasonably named classes from a user-supplied, accessibility-focused model.
    return any(
        term in normalized
        for term in (
            "obstacle",
            "pothole",
            "manhole",
            "curb",
            "stair",
            "step",
            "barrier",
            "bollard",
            "pole",
            "tree",
            "bicycle",
            "motorcycle",
            "bus truck",
        )
    )


def _direction(center_x: float, width: int) -> str:
    ratio = center_x / max(width, 1)
    if ratio < 0.38:
        return "bên trái"
    if ratio > 0.62:
        return "bên phải"
    return "phía trước"


def _relative_band(box_height: int, box_bottom: int, frame_height: int) -> str:
    height_ratio = box_height / max(frame_height, 1)
    bottom_ratio = box_bottom / max(frame_height, 1)
    if height_ratio >= 0.40 or (height_ratio >= 0.27 and bottom_ratio >= 0.92):
        return "gần"
    if height_ratio >= 0.17 or bottom_ratio >= 0.80:
        return "vừa"
    return "xa"


def _image_quality(frame: np.ndarray) -> tuple[float, str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    reasons: list[str] = []
    if blur_score < 15.0:
        reasons.append("hình ảnh có thể bị mờ")
    if brightness < 25.0:
        reasons.append("khung hình quá tối")
    return blur_score, "; ".join(reasons)


def analyze_frame(frame: np.ndarray, model: Any, confidence: float, imgsz: int = 640,
                  device: str | None = None) -> FrameAnalysis:
    """Detect likely walkway hazards and draw them on a copy of the input frame.

    Distance bands are based only on apparent box size and image position. They are
    intentionally relative and must not be interpreted as measured distances.
    """
    # Do not run full-resolution Pillow/Laplacian/mask work on 4K phone video.
    # YOLO resizes to imgsz internally; retaining 960px is enough for the preview.
    if max(frame.shape[:2]) > 960:
        scale = 960 / max(frame.shape[:2])
        frame = cv2.resize(frame, (round(frame.shape[1]*scale), round(frame.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2]
    blur_score, quality_reason = _image_quality(frame)
    prediction_floor = max(0.08, confidence - 0.15)
    result = model.predict(
        source=frame,
        conf=prediction_floor,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )[0]
    annotated = frame.copy()
    obstacles: list[Obstacle] = []
    low_confidence_hazard = False
    names = result.names
    unicode_labels: list[tuple[str, tuple[int, int], tuple[int, int, int]]] = []
    walkway = np.zeros((height, width), dtype=np.uint8)
    blocked = np.zeros_like(walkway)

    # Segmentation models also return the detection boxes above. Blend obstacle
    # and walkway context masks first; the existing boxes still drive warnings.
    masks = getattr(result, "masks", None)
    if masks is not None and result.boxes is not None:
        mask_data = masks.data
        if hasattr(mask_data, "cpu"):
            mask_data = mask_data.cpu().numpy()
        else:
            mask_data = np.asarray(mask_data)
        if mask_data.ndim == 3 and mask_data.shape[0] == len(result.boxes):
            overlay = annotated.copy()
            has_visible_mask = False
            for index, detection in enumerate(result.boxes):
                class_id = int(detection.cls.item())
                class_name = str(names[class_id])
                normalized_name = class_name.strip().lower().replace("_", " ")
                is_walkway_context = normalized_name in {"walkway", "road"}
                if not is_hazard_class(class_name) and not is_walkway_context:
                    continue
                mask = mask_data[index]
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    )
                visible = mask > 0.5
                if not np.any(visible):
                    continue
                if normalized_name == "walkway":
                    color = (40, 180, 40)
                    if float(detection.conf.item()) >= max(confidence, .60):
                        walkway[visible] = 1
                elif normalized_name == "road":
                    color = (150, 150, 150)
                else:
                    color = (0, 190, 255)
                overlay[visible] = color
                has_visible_mask = True
            if has_visible_mask:
                annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    if result.boxes is not None:
        for detection in result.boxes:
            class_id = int(detection.cls.item())
            class_name = str(names[class_id])
            if not is_hazard_class(class_name):
                continue

            x1, y1, x2, y2 = (int(value) for value in detection.xyxy[0].tolist())
            score = float(detection.conf.item())
            # Even an uncertain hazard invalidates a proposed clear corridor.
            pad = max(5, int(width * .025))
            blocked[max(0, y1-pad):min(height, y2+pad), max(0, x1-pad):min(width, x2+pad)] = 1
            # Ignore objects high in the image or entirely at the far edges.
            if y2 < height * 0.52 or x2 < width * 0.16 or x1 > width * 0.84:
                continue

            if score < confidence:
                low_confidence_hazard = True
                color = (0, 190, 255)
                label = f"chưa rõ {score:.2f}"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                unicode_labels.append((label, (x1, y1 - 28), color))
                continue

            direction = _direction((x1 + x2) / 2, width)
            band = _relative_band(max(0, y2 - y1), y2, height)
            obstacle = Obstacle(
                class_name=class_name,
                confidence=score,
                direction=direction,
                distance_band=band,
                box=(x1, y1, x2, y2),
            )
            obstacles.append(obstacle)
            color = (255, 180, 0) if band == "xa" else (0, 190, 255)
            if band == "gần":
                color = (0, 70, 255)
            normalized_name = class_name.lower().replace("_", " ")
            vi_name = VIETNAMESE_NAMES.get(normalized_name, class_name)
            label = f"{vi_name} • {band} • {score:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            unicode_labels.append((label, (x1, y1 - 28), color))

    annotated = _draw_unicode_labels(annotated, unicode_labels)

    unclear_reason = quality_reason
    if low_confidence_hazard:
        unclear_reason = "; ".join(
            part for part in (unclear_reason, "có vật thể nhưng độ tin cậy thấp") if part
        )
    clear_corridors: list[str] = []
    for label, left, right in (("bên trái", .15, .38), ("phía trước", .38, .62), ("bên phải", .62, .85)):
        region = walkway[int(height*.60):int(height*.96), int(width*left):int(width*right)].copy()
        occupied = blocked[int(height*.60):int(height*.96), int(width*left):int(width*right)]
        if np.any(occupied):
            continue  # A small pole/hole still invalidates the whole proposed corridor.
        region[occupied > 0] = 0
        if region.size and region.mean() >= .85:
            count, components, stats, _ = cv2.connectedComponentsWithStats(region)
            if count > 1:
                largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                connected = components == largest
                if connected[:max(1, len(region)//8)].mean() > .8 and connected[-max(1, len(region)//8):].mean() > .8:
                    clear_corridors.append(label)
    return FrameAnalysis(
        frame=annotated,
        obstacles=tuple(obstacles),
        unclear=bool(unclear_reason),
        unclear_reason=unclear_reason,
        blur_score=blur_score,
        clear_corridors=tuple(clear_corridors),
    )


def open_camera(index: int, width: int = 640, height: int = 480, fps: int = 60) -> cv2.VideoCapture | None:
    """Open a webcam using the available Windows capture backends."""
    backends = (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY) if os.name == "nt" else (cv2.CAP_ANY,)
    for backend in backends:
        camera = cv2.VideoCapture(index, backend)
        if not camera.isOpened():
            camera.release()
            continue
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(5):
            ok, _frame = camera.read()
            if ok:
                return camera
            time.sleep(0.1)
        camera.release()
    return None


def open_ip_stream(url: str) -> cv2.VideoCapture | None:
    """Open a network camera stream using FFmpeg when available."""
    ffmpeg = getattr(cv2, "CAP_FFMPEG", None)
    if ffmpeg is not None:
        timeout_params: list[int] = []
        open_timeout = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        read_timeout = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
        if open_timeout is not None and read_timeout is not None:
            timeout_params = [open_timeout, 5000, read_timeout, 2000]
        camera: cv2.VideoCapture | None = None
        try:
            camera = cv2.VideoCapture(url, ffmpeg, timeout_params) if timeout_params else cv2.VideoCapture(url, ffmpeg)
        except (cv2.error, TypeError):
            pass
        if camera is not None and camera.isOpened():
            return camera
        if camera is not None:
            camera.release()

    try:
        camera = cv2.VideoCapture(url)
    except cv2.error:
        return None
    if camera.isOpened():
        return camera
    camera.release()
    return None


class CameraCapture(threading.Thread):
    def __init__(
        self,
        source: int | str,
        latest_frame: LatestFrame,
        stop_event: threading.Event,
        events: Any,
        live_stream: bool = False,
    ) -> None:
        super().__init__(name="camera-capture", daemon=True)
        self.source = source
        self.latest_frame = latest_frame
        self.stop_event = stop_event
        self.events = events
        self.live_stream = live_stream

    def run(self) -> None:
        camera = None
        try:
            if isinstance(self.source, int):
                camera = open_camera(self.source)
            elif self.live_stream:
                camera = open_ip_stream(self.source)
            else:
                camera = cv2.VideoCapture(self.source)
            if camera is None or not camera.isOpened():
                if isinstance(self.source, int):
                    text = "Không mở được camera đã chọn."
                elif self.live_stream:
                    text = "Không mở được luồng IP. Kiểm tra URL, mạng Wi-Fi và ứng dụng camera trên iPhone."
                else:
                    text = "Không mở được video đã chọn."
                self.events.put({"type": "status", "text": text})
                self.stop_event.set()
                return

            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            is_video = isinstance(self.source, str) and not self.live_stream
            fps = float(camera.get(cv2.CAP_PROP_FPS)) if is_video else 0.0
            frame_period = 1.0 / min(max(fps, 1.0), 60.0) if fps > 0 else 1.0 / 25.0
            if self.live_stream:
                initial_status = "Camera IP đã kết nối; đang khởi tạo nhận diện…"
            elif is_video:
                initial_status = "Video đã mở; đang khởi tạo nhận diện…"
            else:
                initial_status = "Camera đã mở; đang khởi tạo nhận diện…"
            self.events.put({
                "type": "status",
                "text": initial_status,
            })
            self.events.put({"type": "camera_info", "width": int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             "height": int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), "reported_fps": float(np.nan_to_num(camera.get(cv2.CAP_PROP_FPS)))})
            next_frame_at = time.perf_counter()
            while not self.stop_event.is_set():
                ok, frame = camera.read()
                if not ok:
                    if is_video:
                        text = "Video đã phát hết."
                    elif self.live_stream:
                        text = "Mất tín hiệu camera IP. Kiểm tra Wi-Fi và trạng thái phát trực tiếp trên iPhone."
                    else:
                        text = "Mất tín hiệu camera."
                    self.stop_event.set()
                    self.events.put({"type": "source_finished", "text": text})
                    break
                self.latest_frame.publish(frame)
                if is_video:
                    next_frame_at += frame_period
                    self.stop_event.wait(max(0, next_frame_at - time.perf_counter()))
        except Exception as exc:
            self.stop_event.set()
            self.events.put({"type": "error", "text": f"Không đọc được nguồn hình ảnh ({type(exc).__name__}). Kiểm tra camera và kết nối."})
        finally:
            if camera is not None:
                camera.release()
            self.latest_frame.close()

class InferenceWorker(threading.Thread):
    def __init__(
        self,
        model_path: str,
        confidence: float,
        latest_frame: LatestFrame,
        stop_event: threading.Event,
        events: Any,
        speech: Any,
        imgsz: int = 480,
        device: str | None = None,
        ready_event: threading.Event | None = None,
    ) -> None:
        super().__init__(name="obstacle-inference", daemon=True)
        self.model_path = model_path
        self.confidence = confidence
        self.latest_frame = latest_frame
        self.stop_event = stop_event
        self.events = events
        self.speech = speech
        self.imgsz, self.device, self.ready_event = imgsz, device, ready_event
        self.policy = AlertPolicy()
        self.measurements = Measurements()
        self.processing = Measurements()
        self._processed_frames = 0
        self._skipped_frames = 0
        self._spoken_alerts = 0


    def run(self) -> None:
        try:
            import torch
            torch.set_num_threads(max(1, int(os.environ.get("GUIDE_TORCH_THREADS", "8"))))
            from ultralytics import YOLO

            self.events.put({"type": "status", "text": "Đang nạp mô hình nhận diện…"})
            model = YOLO(self.model_path, task="detect" if self.model_path.endswith(".onnx") else None)
            # Load kernels before opening a video, otherwise its first seconds are lost.
            model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=self.imgsz,
                          device=self.device, verbose=False)
            torch.set_num_threads(max(1, int(os.environ.get("GUIDE_TORCH_THREADS", "8"))))
            if self.ready_event is not None:
                self.ready_event.set()
            self.events.put({"type": "status", "text": "Đang theo dõi hình ảnh. Không có cảnh báo không đồng nghĩa đường an toàn."})
            sequence = 0
            while not self.stop_event.is_set():
                new_sequence, frame, closed, captured_at = self.latest_frame.next_after(sequence)
                if frame is None:
                    if closed or self.stop_event.is_set():
                        break
                    continue
                self._skipped_frames += max(0, new_sequence - sequence - 1)
                sequence = new_sequence
                started = time.perf_counter()
                analysis = analyze_frame(frame, model, self.confidence, self.imgsz, self.device)
                elapsed_ms = (time.perf_counter() - captured_at) * 1000.0
                self.processing.add((time.perf_counter() - started) * 1000)
                self.measurements.add(elapsed_ms)
                if elapsed_ms > 1500 or self.stop_event.is_set():
                    # A slow first inference must never speak instructions for an old frame.
                    if not self.stop_event.is_set():
                        self.events.put({"type": "analysis_stale"})
                        self._emit_metrics()
                    continue
                self._processed_frames += 1
                guidance, should_speak = self.policy.update(analysis, VIETNAMESE_NAMES, time.perf_counter())
                self.events.put(
                    {
                        "type": "analysis",
                        "analysis": analysis,
                        "latency_ms": elapsed_ms,
                        "skipped_frames": self._skipped_frames,
                        "captured_at": captured_at,
                        "guidance": guidance,
                    }
                )
                if should_speak and not self.stop_event.is_set():
                    self._spoken_alerts += 1
                    self.speech.speak(guidance.message, captured_at, priority=guidance.priority,
                                      clip=guidance.clip, key=guidance.key)
                    self.events.put({"type": "alert_count", "count": self._spoken_alerts})
                if self._processed_frames == 1 or time.perf_counter() - getattr(self, "_last_metrics", 0) >= .5:
                    self._emit_metrics()
                    self._last_metrics = time.perf_counter()
        except Exception as exc:  # surface model, camera-frame, and runtime errors in the UI
            self.events.put({"type": "error", "text": f"Lỗi nhận diện: {exc}"})
            self.stop_event.set()
            self.latest_frame.close()
        finally:
            self.stop_event.set()
            self.latest_frame.close()
            self._emit_metrics()
            self.events.put({"type": "finished"})

    def _emit_metrics(self):
        stats, camera, processing = self.measurements.snapshot(), self.latest_frame.measurements.snapshot(), self.processing.snapshot()
        self.events.put({"type": "metrics", "processed_frames": stats["count"],
                         "capture_frames": camera["count"], "camera_fps": camera["fps"],
                         "inference_fps": stats["fps"], "skipped_frames": self._skipped_frames,
                         "p50_frame_to_inference_ms": stats["p50_ms"],
                         "p95_frame_to_inference_ms": stats["p95_ms"],
                         "p95_processing_ms": processing["p95_ms"], "alert_count": self._spoken_alerts})
