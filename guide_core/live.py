"""Live camera/stream multi-rate worker bridging GuideEngine (OpenVINO/PyTorch) with Desktop & Web UIs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
import threading
import time
import uuid
import cv2
import numpy as np

from .audio import AudioTimeline, ClipBank
from .config import Config
from .engine import GuideEngine
from .fusion import fuse
from .planning import _vi_label, route_blocked, urgent
from .types import Action, Decision, DIRECTIONAL, Frame, Scene


class LiveSpeakerSink:
    """Hardware PCM audio sink implementing submit(samples, rate) / stop() for AudioTimeline."""
    def __init__(self, events, stereo=True, output_factory=None):
        from audio_output import Audio, create_output
        self._Audio = Audio
        self.events = events
        self.channels = 2 if stereo else 1
        self._lock = threading.Lock()
        self.output = None
        self.backend_name = "unavailable"
        self._last_capture_at = None
        self._was_busy = False
        factory = output_factory or (lambda: create_output(channels=self.channels))
        try:
            self.output = factory()
            self.backend_name = getattr(self.output, "name", "PCM") + (" Stereo" if stereo else " Mono")
            # Warm hardware buffer once outside the alert path (20ms silence at 24kHz).
            warmup_bytes = bytes(960 * self.channels)
            self.output.play(Audio(warmup_bytes, channels=self.channels))
            deadline = time.perf_counter() + 0.5
            while self.output.busy() and time.perf_counter() < deadline:
                time.sleep(0.005)
            self.output.stop()
        except Exception as exc:
            self.output = None
            self.events.put({"type": "speech_error", "text": f"Thiết bị loa chưa sẵn sàng: {exc}"})

    def bind_capture_time(self, captured_at: float, was_busy: bool) -> None:
        self._last_capture_at = captured_at
        self._was_busy = was_busy

    def submit(self, samples: np.ndarray, rate: int) -> None:
        with self._lock:
            if self.output is None:
                return
            pcm = np.asarray(samples, dtype="<i2").tobytes()
            audio = self._Audio(pcm, channels=self.channels)
            t_req = time.perf_counter()
            self.output.play(audio)
            submitted_at = time.perf_counter()
        cap_t = self._last_capture_at or t_req
        self.events.put({
            "type": "feedback_latency",
            "latency_ms": max(0.0, (submitted_at - cap_t) * 1000.0),
            "queue_wait_ms": 0.0,
            "queued_behind_active": self._was_busy,
            "key": "guide_core_live",
            "priority": 3,
            "dispatch_ms": (submitted_at - t_req) * 1000.0,
            "audio_duration_ms": audio.duration * 1000.0,
            "backend": self.backend_name,
            "measurement": "audio_submit",
        })

    def stop(self) -> None:
        with self._lock:
            if self.output is not None:
                try:
                    self.output.stop()
                except Exception:
                    pass

    def close(self) -> None:
        with self._lock:
            if self.output is not None:
                try:
                    self.output.close()
                except Exception:
                    pass
                self.output = None


def render_live_hud(scene: Scene, decision: Decision, stage: str = "live") -> np.ndarray:
    """Fast presentation-thread HUD overlay (<2.5ms) with 3D depth radar PIP and corridor ribbon."""
    from obstacle_detector import _draw_unicode_labels

    frame = scene.frame.bgr.copy()
    h, w = frame.shape[:2]

    # 1. Highlight RANSAC 3D Depth Anomalies (unlabelled positive obstacles & drop-offs).
    anomalies = getattr(scene, "depth_anomalies", None)
    if anomalies is not None:
        overlay = frame.copy()
        changed = False
        pos = anomalies.get("positive")
        drop = anomalies.get("dropoff")
        if pos is not None and pos.shape == (h, w) and np.any(pos):
            overlay[pos] = (30, 40, 235)
            changed = True
        if drop is not None and drop.shape == (h, w) and np.any(drop):
            overlay[drop] = (230, 210, 0)
            changed = True
        if changed:
            frame = cv2.addWeighted(overlay, 0.30, frame, 0.70, 0)

    # 2. Draw confirmed or candidate walkable corridor ribbon.
    route = decision.route
    if route is not None and route.mask is not None and route.mask.shape == (h, w) and np.any(route.mask):
        corridor_overlay = frame.copy()
        ribbon_color = (50, 200, 75) if decision.action in DIRECTIONAL else (40, 175, 220)
        corridor_overlay[route.mask] = ribbon_color
        frame = cv2.addWeighted(corridor_overlay, 0.26, frame, 0.74, 0)
    if route is not None and len(route.path) >= 2:
        pts = np.array(route.path, dtype=np.int32)
        line_color = (60, 245, 110) if decision.action in DIRECTIONAL else (60, 210, 255)
        cv2.polylines(frame, [pts], False, line_color, 3, cv2.LINE_AA)

    # 3. Draw tracked hazard boxes with Vietnamese labels, head-zone tag, and Looming TTC.
    labels = []
    for obj in scene.detections:
        x1, y1, x2, y2 = map(int, obj.box)
        ttc = getattr(obj, "ttc_s", None)
        is_head = getattr(obj, "zone", "ground") == "head"
        urgent_box = (ttc is not None and ttc <= 1.6) or is_head or (y2 > h * 0.80 and x1 < w * 0.62 and x2 > w * 0.38)
        color = (0, 55, 255) if urgent_box else (0, 185, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        vi = _vi_label(obj.label).replace(" phía trước", "")
        tags = [vi]
        if is_head:
            tags.append("TRÊN ĐẦU")
        if ttc is not None:
            tags.append(f"TTC {ttc:.1f}s")
        tags.append(f"{obj.confidence:.2f}")
        labels.append((" • ".join(tags), (x1, max(4, y1 - 28)), color))

    # 4. Compact top-right Depth Radar PIP thumbnail when depth is active.
    if scene.depth is not None and scene.depth.get("relative_nearness") is not None:
        pip_w = max(96, min(160, w // 4))
        pip_h = max(64, round(pip_w * h / max(w, 1)))
        near_u8 = (np.clip(scene.depth["relative_nearness"], 0, 1) * 255).astype(np.uint8)
        pip = cv2.applyColorMap(cv2.resize(near_u8, (pip_w, pip_h)), cv2.COLORMAP_INFERNO)
        if w > pip_w + 12 and h > pip_h + 12:
            frame[8:8 + pip_h, w - pip_w - 8:w - 8] = pip
            cv2.rectangle(frame, (w - pip_w - 8, 8), (w - 8, 8 + pip_h), (220, 230, 240), 1)

    if labels:
        frame = _draw_unicode_labels(frame, labels)
    return frame


@dataclass(frozen=True)
class LiveSceneView:
    scene: Scene
    decision: Decision
    stage: str
    unclear: bool

    def render_fn(self) -> np.ndarray:
        return render_live_hud(self.scene, self.decision, self.stage)


@dataclass(frozen=True)
class LiveGuidanceView:
    message: str
    key: str
    priority: int
    action: str


class CoreInferenceWorker(threading.Thread):
    """Multi-rate live inference thread running fast detector (P-cores) + async context (Iris Xe iGPU)."""
    def __init__(
        self,
        core_config_path: str | Path,
        latest_frame,
        stop_event: threading.Event,
        events,
        app_dir: Path,
        *,
        confidence: float | None = None,
        imgsz: int | None = None,
        environment: str | None = None,
        mirror_input: bool = False,
        sound_enabled: bool = True,
        ready_event: threading.Event | None = None,
        output_factory=None,
    ) -> None:
        super().__init__(name="guide-core-live", daemon=True)
        self.core_config_path = Path(core_config_path)
        self.latest_frame = latest_frame
        self.stop_event = stop_event
        self.events = events
        self.app_dir = Path(app_dir)
        self.confidence = confidence
        self.imgsz = imgsz
        self.environment = environment
        self.mirror_input = bool(mirror_input)
        self._sound_enabled = bool(sound_enabled)
        self.ready_event = ready_event
        self._output_factory = output_factory
        self._audio_lock = threading.Lock()
        self.audio: AudioTimeline | None = None
        self.sink: LiveSpeakerSink | None = None
        self.bank: ClipBank | None = None
        self._spoken_alerts = 0
        self._skipped_frames = 0

    def set_sound_enabled(self, enabled: bool) -> None:
        with self._audio_lock:
            self._sound_enabled = bool(enabled)
            if self.audio is not None:
                if not enabled:
                    self.audio.stop(time.perf_counter())
                else:
                    self.audio.enabled = True

    def test_speak_stop(self) -> None:
        with self._audio_lock:
            if self.audio is not None and self.sink is not None and self._sound_enabled:
                now = time.perf_counter()
                dec = Decision(Action.STOP, now, now + 2.0, ("manual-test", 0), "manual_test", priority=3)
                self.sink.bind_capture_time(now, self.audio.active is not None)
                self.audio.enabled = True
                self.audio.update(dec, now)

    def run(self) -> None:
        from performance import Measurements

        engine = None
        measurements = Measurements()
        processing = Measurements()
        try:
            cfg = Config.read(self.core_config_path)
            cfg.mode = "realtime_replay"
            cfg.mirror_input = self.mirror_input
            if self.environment in {"auto", "sidewalk", "alley", "indoor"}:
                cfg.environment = self.environment
            for m in cfg.models:
                p = Path(m.path)
                if not p.is_absolute():
                    m.path = str(self.app_dir / p)
                if m.role == "detector":
                    if self.confidence is not None:
                        m.confidence = float(self.confidence)
                    if self.imgsz in (320, 480, 640) and m.adapter == "yolo":
                        m.imgsz = int(self.imgsz)

            audio_dir = Path(cfg.audio_dir)
            if not audio_dir.is_absolute():
                audio_dir = self.app_dir / audio_dir

            self.events.put({"type": "status", "text": "Đang nạp lõi Guide v3 (OpenVINO / Đa mô hình)…"})
            engine = GuideEngine(cfg).load()
            self.bank = ClipBank(audio_dir)
            self.sink = LiveSpeakerSink(self.events, stereo=cfg.stereo_audio, output_factory=self._output_factory)
            with self._audio_lock:
                self.audio = AudioTimeline(
                    self.bank,
                    sink=self.sink,
                    preempt_emergency=cfg.preempt_emergency_audio,
                    smart_arbitration=cfg.smart_arbitration,
                    post_stop_hold_s=cfg.post_stop_hold_s,
                    post_stop_turn_hold_s=cfg.post_stop_turn_hold_s,
                    tail_grace_s=cfg.tail_grace_s,
                    revocation_debounce_frames=cfg.revocation_debounce_frames,
                    earcon=cfg.earcon_enabled,
                    stereo=cfg.stereo_audio,
                )
                self.audio.enabled = self._sound_enabled
                if cfg.smart_arbitration:
                    engine.policy.spoken_direction = self.audio.direction_in_effect

            dev_summary = ", ".join(f"{a.spec.role}:{a.device}" for a in engine.adapters)
            self.events.put({
                "type": "speech",
                "text": f"Khánh Vy Stereo • Earcon 55ms • {self.sink.backend_name}",
            })
            if self.ready_event is not None:
                self.ready_event.set()
            self.events.put({
                "type": "status",
                "text": f"Đang chạy Guide v3 ({dev_summary}) • Bối cảnh: {cfg.environment}",
            })

            session = "live-" + uuid.uuid4().hex[:10]
            origin = time.perf_counter()
            sequence = 0
            detector_job = None
            context_job = None
            last_context_seq = -1
            later_scenes: list[Scene] = []
            context_roles = {m.role for m in cfg.models if m.enabled and m.role != "detector"}
            last_metrics_at = time.perf_counter()

            def publish_decision(scene: Scene, decision: Decision, stage: str) -> None:
                now = time.perf_counter()
                elapsed_ms = (now - scene.frame.received_at) * 1000.0
                measurements.add(elapsed_ms)
                with self._audio_lock:
                    if self.audio is not None and self._sound_enabled:
                        before = len(self.audio.segments)
                        was_busy = self.audio.active is not None
                        if self.sink is not None:
                            self.sink.bind_capture_time(scene.frame.received_at, was_busy)
                        self.audio.update(decision, now)
                        if len(self.audio.segments) > before:
                            self._spoken_alerts += 1
                            self.events.put({"type": "alert_count", "count": self._spoken_alerts})
                view = LiveSceneView(scene=scene, decision=decision, stage=stage, unclear=decision.action == Action.UNCLEAR)
                guidance = LiveGuidanceView(
                    message=decision.text,
                    key=f"{decision.action.value}|{decision.reason}",
                    priority=decision.priority,
                    action=decision.action.value.lower(),
                )
                self.events.put({
                    "type": "analysis",
                    "analysis": view,
                    "latency_ms": elapsed_ms,
                    "skipped_frames": self._skipped_frames,
                    "captured_at": scene.frame.received_at,
                    "frame_sequence": scene.frame.sequence,
                    "guidance": guidance,
                })
                if measurements.snapshot()["count"] <= 5:
                    self._emit_metrics(measurements, processing)

            with ThreadPoolExecutor(max_workers=2) as pool:
                while not self.stop_event.is_set():
                    now = time.perf_counter()
                    if detector_job is not None and detector_job[2].done():
                        frame, det_started, future = detector_job
                        detector_job = None
                        results = future.result()
                        processing.add((time.perf_counter() - det_started) * 1000.0)
                        if not any(r.skipped for r in results):
                            scene = fuse(frame, results)
                            if scene.detector_valid:
                                scene = engine.tracker.update(scene)
                            now = time.perf_counter()
                            if context_job is not None and now - context_job[0].frame.received_at <= cfg.max_age_s:
                                later_scenes.append(scene)
                            if now - frame.received_at <= cfg.max_age_s:
                                dec = engine.policy.update(scene, None, now, False)
                                dec = replace(dec, valid_until=min(dec.valid_until, frame.received_at + cfg.max_age_s))
                                engine.policy.active = dec
                                publish_decision(scene, dec, "detector")
                                if context_job is None and context_roles and scene.observation_valid:
                                    ctx_fut = pool.submit(engine.predict, frame, context_roles)
                                    context_job = (scene, results, ctx_fut)
                                    later_scenes = []
                            else:
                                self.events.put({"type": "analysis_stale"})

                    if context_job is not None and context_job[2].done():
                        anchor, det_results, ctx_fut = context_job
                        context_job = None
                        ctx_results = ctx_fut.result()
                        frame = anchor.frame
                        now = time.perf_counter()
                        usable = (
                            now - frame.received_at <= cfg.max_age_s
                            and frame.sequence > last_context_seq
                            and not any(r.skipped for r in ctx_results)
                        )
                        if usable:
                            scene = fuse(frame, det_results + ctx_results)
                            scene.detections = anchor.detections
                            scene.motion_invalid = anchor.motion_invalid
                            scene.warp = anchor.warp
                            route = engine.planner.plan(scene)
                            now = time.perf_counter()
                            veto = any(
                                not newer.observation_valid or newer.errors
                                or route_blocked(route, newer) or urgent(newer)
                                for newer in later_scenes
                            )
                            if veto:
                                engine.policy.reset_confirmation()
                            elif now - frame.received_at <= cfg.max_age_s:
                                dec = engine.policy.update(scene, route, now, True)
                                dec = replace(dec, valid_until=frame.received_at + cfg.max_age_s)
                                engine.policy.active = dec
                                last_context_seq = frame.sequence
                                publish_decision(scene, dec, "fused")
                        later_scenes.clear()

                    with self._audio_lock:
                        if self.audio is not None and self._sound_enabled:
                            self.audio.tick(time.perf_counter())

                    if detector_job is None:
                        new_seq, raw_bgr, closed, captured_at = self.latest_frame.next_after(sequence, timeout=0.015)
                        if raw_bgr is None:
                            if closed and context_job is None:
                                break
                        else:
                            self._skipped_frames += max(0, new_seq - sequence - 1)
                            sequence = new_seq
                            now = time.perf_counter()
                            if (now - captured_at) > cfg.max_input_age_s:
                                self.events.put({"type": "analysis_stale"})
                            else:
                                pts = max(0.0, captured_at - origin)
                                f_obj = Frame.prepare(raw_bgr, session, new_seq, pts, cfg.max_side,
                                                      cfg.mirror_input, received_at=captured_at)
                                detector_job = (f_obj, now, pool.submit(engine.predict, f_obj, {"detector"}))
                    else:
                        time.sleep(0.004)

                    if time.perf_counter() - last_metrics_at >= 0.4:
                        self._emit_metrics(measurements, processing)
                        last_metrics_at = time.perf_counter()
        except Exception as exc:
            self.events.put({"type": "error", "text": f"Lỗi lõi Guide v3: {exc}"})
            self.stop_event.set()
            self.latest_frame.close()
        finally:
            self.stop_event.set()
            self.latest_frame.close()
            with self._audio_lock:
                if self.audio is not None:
                    self.audio.stop(time.perf_counter())
                if self.sink is not None:
                    self.sink.close()
            if engine is not None:
                engine.close()
            self._emit_metrics(measurements, processing)
            self.events.put({"type": "finished"})

    def _emit_metrics(self, measurements, processing) -> None:
        stats = measurements.snapshot()
        camera = self.latest_frame.measurements.snapshot()
        proc = processing.snapshot()
        try:
            import psutil
            rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            rss_mb = None
        self.events.put({
            "type": "metrics",
            "processed_frames": stats["count"],
            "capture_frames": camera["count"],
            "camera_fps": camera["fps"],
            "process_rss_mb": rss_mb,
            "inference_fps": stats["fps"],
            "skipped_frames": self._skipped_frames,
            "p50_frame_to_inference_ms": stats["p50_ms"],
            "p95_frame_to_inference_ms": stats["p95_ms"],
            "p95_processing_ms": proc["p95_ms"],
            "alert_count": self._spoken_alerts,
        })
