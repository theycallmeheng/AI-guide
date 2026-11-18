"""Shared runtime for desktop and local web. The UI never owns inference threads."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
import time
from collections import deque
from urllib.parse import urlsplit

from feedback import SpeechWorker
from obstacle_detector import CameraCapture, InferenceWorker, LatestFrame
from performance import Measurements

APP_DIR = Path(__file__).resolve().parent

def default_model():
    return next((p for p in (APP_DIR/"models/results/yolo26n_best.pt", APP_DIR/"models/results/best.pt", APP_DIR/"models/walkway_obstacles.pt",
                            APP_DIR/"models/yolo11n.pt", APP_DIR/"yolo11n.pt") if p.is_file()),
                APP_DIR/"models/yolo11n.pt")

def default_segformer_model():
    path = APP_DIR/"models/segformer_sidewalk"
    if (path/"config.json").is_file() and any(
        (path/name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return str(path)
    return ""

def available_core_profiles() -> dict[str, dict[str, str | None]]:
    profiles: dict[str, dict[str, str | None]] = {}
    ov_igpu = APP_DIR/"configs/guide_openvino_igpu.json"
    ov_int8 = APP_DIR/"configs/guide_openvino_int8.json"
    ov_ready = (APP_DIR/"models/openvino/export_manifest.json").is_file()
    if ov_ready and ov_igpu.is_file():
        profiles["openvino_igpu"] = {
            "label": "Guide v3 OpenVINO (P-core + Iris Xe iGPU)",
            "path": str(ov_igpu),
        }
    if ov_ready and ov_int8.is_file():
        profiles["openvino_int8"] = {
            "label": "Guide v3 OpenVINO Full INT8 (Siêu nhanh)",
            "path": str(ov_int8),
        }
    opt_pt = APP_DIR/"configs/guide_optimized.json"
    if opt_pt.is_file():
        profiles["pytorch_v3"] = {
            "label": "Guide v3 PyTorch (Đa mô hình 3D)",
            "path": str(opt_pt),
        }
    profiles["legacy_v1"] = {
        "label": "Cơ bản V1 (Đối chiếu cũ)",
        "path": None,
    }
    return profiles

@dataclass(frozen=True)
class PipelineConfig:
    source: int | str = 0
    model: str = str(default_model())
    confidence: float = .30
    imgsz: int = 480
    live_stream: bool = False
    device: str | None = None
    segformer_model: str | None = default_segformer_model()
    allow_routes: bool = False
    core_config: str | None = None
    environment: str = "alley"
    mirror_input: bool = False

    def validate(self):
        if self.imgsz not in (320, 480, 640):
            raise ValueError("Chọn kích thước suy luận 320, 480 hoặc 640.")
        if not .1 <= self.confidence <= .9:
            raise ValueError("Độ tin cậy phải từ 0.1 đến 0.9.")
        if self.core_config:
            core_path = Path(self.core_config).resolve()
            if not core_path.is_relative_to(APP_DIR.resolve()) or not core_path.is_file():
                raise ValueError("Không tìm thấy cấu hình lõi Guide v3 hợp lệ trong dự án.")
            if self.environment not in {"sidewalk", "alley", "indoor"}:
                raise ValueError("Bối cảnh phải là sidewalk, alley hoặc indoor.")
        else:
            if not Path(self.model).exists():
                raise ValueError("Không tìm thấy mô hình cục bộ. Chọn best.pt hoặc mô hình ONNX đã xuất.")
            if self.segformer_model:
                segmenter = Path(self.segformer_model)
                if not (segmenter/"config.json").is_file() or not any(
                    (segmenter/name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
                ):
                    raise ValueError("Thư mục SegFormer cần có config.json và model.safetensors hoặc pytorch_model.bin.")
        if isinstance(self.source, bool) or not isinstance(self.source, (str, int)):
            raise ValueError("Nguồn camera không hợp lệ.")
        if isinstance(self.source, int):
            if not 0 <= self.source <= 30:
                raise ValueError("Chỉ số camera phải từ 0 đến 30.")
        elif self.live_stream:
            url = urlsplit(self.source)
            if url.scheme.lower() not in {"rtsp", "rtsps", "http", "https"} or not url.hostname:
                raise ValueError("Nhập URL luồng RTSP/MJPEG đầy đủ.")
        elif not Path(self.source).is_file():
            raise ValueError("Không tìm thấy video đã chọn.")

class SilentSpeech:
    def observe(self, *args, **kwargs): pass
    def speak(self, *args, **kwargs): pass
    def cancel(self): pass
    def clear_pending(self): pass
    def close(self): pass
    def set_enabled(self, enabled): pass

class GuidePipeline:
    def __init__(self, speech=True, engine=None):
        self._lock = threading.RLock()
        self._lifecycle = threading.Lock()
        self._state = {"status": "Sẵn sàng", "speech": "Âm thanh đang khởi tạo" if speech else "Đã tắt âm thanh",
                       "guidance": "Chọn nguồn hình ảnh để bắt đầu.", "action": "observe",
                       "running": False, "metrics": {}, "report": None}
        self._frame = None
        self._frame_id = 0
        self.audio_events = deque(maxlen=10000)
        self._session_clock = time.perf_counter()
        self._supervisor = None
        self.stop_event = threading.Event()
        self.latest = None
        self.inference = None
        self.capture = None
        self.audio_stats = Measurements()
        self.dispatch_stats = Measurements()
        self.idle_audio_stats = Measurements()
        self.queue_stats = Measurements()
        self._sound_enabled = bool(speech)
        self.speech = SpeechWorker(self, engine) if speech else SilentSpeech()

    def set_sound_enabled(self, enabled: bool):
        self._sound_enabled = bool(enabled)
        self.speech.set_enabled(enabled)
        if self.inference is not None and hasattr(self.inference, "set_sound_enabled"):
            self.inference.set_sound_enabled(enabled)

    def test_alert(self):
        if self.inference is not None and self.inference.is_alive() and hasattr(self.inference, "test_speak_stop"):
            self.inference.test_speak_stop()
        else:
            self.speech.speak("Dừng lại.", priority=3, clip="stop", max_age=10)

    def put(self, event):
        kind = event["type"]
        with self._lock:
            if kind in {"speech", "speech_error"}:
                self._state["speech"] = event["text"]
            elif kind == "analysis":
                # Drop frames completed after Stop or source failure.
                if self.stop_event.is_set():
                    return
                if hasattr(self.speech, "observe"):
                    self.speech.observe(event["guidance"].key, event["captured_at"], event["guidance"].priority)
                self._frame = event
                self._frame_id += 1
                self._state.update(guidance=event["guidance"].message, action=event["guidance"].action,
                                   status="Đang nhận diện • chưa rõ" if event["analysis"].unclear else "Đang nhận diện")
            elif kind == "feedback_latency":
                self.audio_events.append(dict(event, time_s=time.perf_counter()-self._session_clock))
                self.queue_stats.add(event.get('queue_wait_ms', 0))
                if not event.get('queued_behind_active', False):
                    self.idle_audio_stats.add(event['latency_ms'])
                self._state['metrics'].update(p95_idle_audio_submit_ms=self.idle_audio_stats.snapshot()['p95_ms'],
                                             p95_audio_queue_wait_ms=self.queue_stats.snapshot()['p95_ms'])
                self.audio_stats.add(event["latency_ms"])
                stats = self.audio_stats.snapshot()
                self.dispatch_stats.add(event.get("dispatch_ms",0))
                dispatch = self.dispatch_stats.snapshot()
                self._state["metrics"].update(audio_samples=stats["count"], p50_frame_to_audio_submit_ms=stats["p50_ms"],
                                               p95_frame_to_audio_submit_ms=stats["p95_ms"],
                                               p95_speech_dispatch_ms=dispatch["p95_ms"],
                                               audio_backend=event.get("backend","unknown"))
            elif kind == "metrics":
                self._state["metrics"].update({k: v for k, v in event.items() if k != "type"})
            elif kind == "analysis_stale":
                self._frame = None
                self._state.update(status="AI đang xử lý quá chậm. Khung xem là video gốc; chưa đủ kết quả để chỉ dẫn. Tạm tắt SegFormer hoặc dùng GPU.",
                                   guidance="Chưa có kết quả đủ mới để chỉ dẫn.", action="unavailable")
                self.speech.clear_pending()
            elif kind == "camera_info":
                self._state["camera"] = {k: v for k, v in event.items() if k != "type"}
            elif kind in {"status", "source_finished", "error"}:
                self._state["status"] = event["text"]
                if kind in {"error", "source_finished"}:
                    self._state.update(guidance=event["text"], action="unavailable")
                    self._frame = None
                    if hasattr(self, "speech"):
                        self.speech.clear_pending()
                    if kind == "source_finished":
                        self._source_failure = isinstance(self.config.source, int) or self.config.live_stream
            elif kind == "alert_count":
                self._state["metrics"]["alert_count"] = event["count"]

    def snapshot(self):
        with self._lock:
            state = json.loads(json.dumps(self._state, ensure_ascii=False, allow_nan=False))
            frame = self._frame
            state["frame_id"] = self._frame_id
            state["frame_age_ms"] = round((time.perf_counter()-frame["captured_at"])*1000) if frame else None
            return state

    def latest_analysis(self):
        with self._lock:
            return self._frame_id, self._frame

    def latest_preview(self):
        latest = self.latest
        if latest is None or self.stop_event.is_set():
            return None
        return latest.preview_snapshot()

    def is_running(self):
        return any(worker is not None and worker.is_alive() for worker in (self._supervisor, self.capture, self.inference))

    def speak(self, *args, **kwargs):
        # Serialize warning submission with Stop: no inference may enqueue after cancellation.
        with self._lock:
            if not self.stop_event.is_set():
                self.speech.speak(*args, **kwargs)

    def start(self, config):
        config.validate()
        with self._lifecycle:
            if self.is_running():
                raise RuntimeError("Phiên nhận diện đang chạy hoặc chưa dừng xong.")
            self.config = config
            self._source_failure = False
            self.speech.cancel()
            self.stop_event = threading.Event()
            self.latest = LatestFrame()
            self.audio_stats = Measurements()
            self.idle_audio_stats = Measurements()
            self.queue_stats = Measurements()
            self.audio_events.clear()
            self._session_clock = time.perf_counter()
            self.dispatch_stats = Measurements()
            with self._lock:
                self._frame = None
                self._state.update(status="Đang nạp và làm nóng mô hình…", guidance="Đang chờ hình ảnh mới.",
                                   action="observe", running=True, metrics={}, report=None)
            ready = threading.Event()
            if config.core_config:
                from guide_core.live import CoreInferenceWorker
                self.inference = CoreInferenceWorker(
                    config.core_config, self.latest, self.stop_event, self, APP_DIR,
                    confidence=config.confidence, imgsz=config.imgsz,
                    environment=config.environment, mirror_input=config.mirror_input,
                    sound_enabled=self._sound_enabled, ready_event=ready,
                )
            else:
                self.inference = InferenceWorker(config.model, config.confidence, self.latest, self.stop_event,
                                                 self, self, config.imgsz, config.device, ready,
                                                 config.segformer_model, config.allow_routes)
            self.capture = CameraCapture(config.source, self.latest, self.stop_event, self, config.live_stream)
            self._supervisor = threading.Thread(target=self._run, args=(ready,), name="guide-session", daemon=True)
            self._supervisor.start()

    def _run(self, ready):
        self.inference.start()
        started = time.perf_counter()
        started_at = datetime.now().isoformat()
        captured_started = None
        stale_alert = False
        try:
            while not ready.wait(.1):
                if self.stop_event.is_set() or not self.inference.is_alive():
                    return
            if self.stop_event.is_set():
                return
            self.capture.start()
            captured_started = time.perf_counter()
            while self.inference.is_alive() or self.capture.is_alive():
                if self.stop_event.wait(.1):
                    break
                _, _, _, captured_at = self.latest.next_after(-1, timeout=0)
                age = time.perf_counter() - (captured_at or captured_started)
                if age > .7 and not stale_alert:
                    stale_alert = True
                    with self._lock:
                        self._frame = None
                        self._state.update(guidance="Mất khung hình mới. Hãy dừng để kiểm tra.", action="unavailable")
                    self.speech.speak("Mất tín hiệu camera. Hãy dừng để kiểm tra.", priority=3, clip="disconnected")
                elif age < .5:
                    stale_alert = False
                if age > 5:
                    self.put({"type": "error", "text": "Camera không gửi khung hình mới trong 5 giây. Kết nối lại rồi thử lại."})
                    self.stop_event.set()
                    break
        finally:
            self.stop_event.set()
            self.latest.close()
            if self.capture.ident is not None:
                self.capture.join(timeout=3)
            self.inference.join(timeout=10)
            report = {"started_at": started_at, "duration_s": round(time.perf_counter()-started, 2),
                      "imgsz": self.config.imgsz, "model": Path(self.config.model).name,
                      "core_config": Path(self.config.core_config).name if self.config.core_config else None,
                      "environment": self.config.environment if self.config.core_config else None,
                      "segformer_model": Path(self.config.segformer_model).name if self.config.segformer_model else None,
                      "route_guidance_enabled": self.config.allow_routes,
                      "source_type": "camera" if isinstance(self.config.source, int) else "ip" if self.config.live_stream else "video",
                      "metrics_window": "last 2048 latency samples; FPS from last 2 seconds",
                      "audio_latency": "software playback submission; hardware onset not measured",
                      "measurement_origin": "frame decoded on laptop; transport/decoder delay is not measured",
                      "metrics": self.snapshot()["metrics"],
                      "audio_events": list(self.audio_events),
                      "audio_event_clock": "seconds since session start, not video presentation timestamps; align before event scoring",
                      "audio_event_capacity": 10000}
            destination = APP_DIR/"reports"/("session_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json")
            try:
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                saved = str(destination)
            except OSError:
                saved = None
            with self._lock:
                self._frame = None
                self._state.update(running=False, action="unavailable", guidance="Phiên đã kết thúc.", report=saved)
                if not self._state["status"].startswith(("Lỗi", "Mất", "Không", "Video", "Camera không")):
                    self._state["status"] = "Đã dừng."
            if self._source_failure:
                self.speech.speak("Mất tín hiệu camera. Hãy dừng để kiểm tra.", priority=3, clip="disconnected")

    def stop(self):
        with self._lock:
            self.stop_event.set()
            self.speech.cancel()
        if self.latest:
            self.latest.close()
        with self._lock:
            self._frame = None
            self._state.update(status="Đang dừng…" if self.is_running() else "Đã dừng.", guidance="Đã dừng.", action="unavailable")

    def close(self):
        self.stop()
        if self._supervisor:
            self._supervisor.join(timeout=4)
        self.speech.close()
