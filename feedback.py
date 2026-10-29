"""Preloaded Vietnamese alerts with serialized, latest-warning playback."""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
import time
import wave
from audio_output import Audio, create_output, read_audio
from fast_alerts import AlertBank, FIXED
from tts_engines import create_engine
from voice_profiles import selected_profile

APP_DIR = Path(__file__).resolve().parent
CLIPS = FIXED  # Kept for the standalone synthesis benchmark.

@dataclass(frozen=True)
class Request:
    generation: int
    sequence: int
    text: str
    key: str | None
    captured_at: float
    priority: int
    expires_at: float
    requested_at: float

def wav_seconds(path):
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()

class SpeechWorker:
    def __init__(self, events, engine=None, cache_dir=None, *, fast_dir=None, output_factory=None):
        from obstacle_detector import VIETNAMESE_NAMES
        self.events = events
        self.engine_name = engine or os.environ.get("GUIDE_TTS_ENGINE", "windows")
        self.cache_dir = Path(cache_dir or APP_DIR/"assets/audio"/self.engine_name)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        profile = selected_profile()
        self._custom_profile = not (cache_dir or fast_dir) and profile["id"] != "an"
        self.voice_label = profile["name"] if not (cache_dir or fast_dir) else "giọng tùy chỉnh"
        self.bank = AlertBank(fast_dir or (Path(cache_dir)/"fast" if cache_dir else APP_DIR/profile["bank"]), VIETNAMESE_NAMES)
        self._output_factory = output_factory or create_output
        self._memory = OrderedDict()
        self._condition = threading.Condition(threading.RLock())
        self._generation = 0
        self._request_sequence = 0
        self._pending = self._play = None
        self._closed = False
        self._enabled = True
        self._active_key = None
        self._active_sequence = None
        self._active_priority = 0
        self.audio_ready = threading.Event()
        self._output_error = None
        self._synth_thread = threading.Thread(target=self._synthesize, name="speech-synthesis", daemon=True)
        self._play_thread = threading.Thread(target=self._playback, name="speech-playback", daemon=True)
        self._synth_thread.start()
        self._play_thread.start()

    def _load(self, path):
        path = Path(path)
        if path not in self._memory:
            self._memory[path] = read_audio(path)
            if len(self._memory) > 64: self._memory.popitem(last=False)
        return self._memory[path]

    def speak(self, message, frame_captured_at=None, *, priority=1, clip=None, key=None, max_age=2.5):
        now = time.perf_counter()
        captured_at = frame_captured_at or now
        with self._condition:
            if self._closed or not self._enabled or captured_at + max_age < now: return
            if key and key == self._active_key and priority <= self._active_priority:
                # The active alert is still the latest state. Drop any different
                # warning that was queued while this same alert kept appearing.
                self._request_sequence += 1
                self._pending = self._play = None
                self._condition.notify_all()
                return
            queued = self._play[0] if self._play is not None else self._pending
            if key and queued is not None and queued.key == key and priority <= queued.priority: return
            self._request_sequence += 1
            request = Request(self._generation, self._request_sequence, message, key,
                              captured_at, priority, captured_at+max_age, now)
            self._pending = self._play = None
            # Command first, followed by the object, direction and relative band.
            audio = self.bank.compose(key,priority)
            if audio is None and clip in self.bank.audio and (not key or priority >= 3):
                audio = self.bank.audio[clip]
            if audio is not None:
                self._play = (request,audio)
            else:
                digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:32]
                path = self.cache_dir/(digest+".wav")
                if path.is_file():
                    self._play = (request,self._load(path))
                else:
                    # No generic intro before a dynamic sentence.
                    self._pending = request
            self._condition.notify_all()

    def cancel(self):
        with self._condition:
            self._generation += 1
            self._pending = self._play = None
            self._active_key = None
            self._active_sequence = None
            self._active_priority = 0
            self._condition.notify_all()

    def clear_pending(self):
        """Discard queued/synthesizing warnings without interrupting current audio."""
        with self._condition:
            self._request_sequence += 1
            self._pending = self._play = None
            self._condition.notify_all()

    def set_enabled(self, enabled):
        with self._condition:
            self.cancel()
            self._enabled = bool(enabled)

    def _synthesize(self):
        engine = fast_engine = None
        try:
            missing = len(self.bank.audio) < len(set(self.bank.labels.values())) + 6 + len(FIXED)
            if missing:
                if self._custom_profile:
                    raise RuntimeError(f"Bộ giọng {self.voice_label} thiếu đoạn. Chạy tools/setup_voice.py với --prepare để bổ sung.")
                # Fast templates stay independent of the optional neural TTS model.
                fast_engine = create_engine("windows" if os.name == "nt" else self.engine_name)
                self.events.put({"type":"speech","text":"Đang chuẩn bị ngân hàng cảnh báo nhanh…"})
                self.bank.prepare(fast_engine, lambda:self._closed)
            if self._output_error is None:
                self.events.put({"type":"speech","text":f"Sẵn sàng: {self.voice_label} • cảnh báo Việt trong RAM"})
            while True:
                with self._condition:
                    self._condition.wait_for(lambda:self._closed or self._pending is not None)
                    if self._closed: return
                    request,self._pending = self._pending,None
                if request.expires_at < time.perf_counter(): continue
                try:
                    if engine is None:
                        engine = fast_engine if self.engine_name == "windows" and fast_engine is not None else create_engine(self.engine_name)
                    path = self.cache_dir/(hashlib.sha256(request.text.encode("utf-8")).hexdigest()[:32]+".wav")
                    temporary = path.with_suffix(".tmp.wav")
                    try:
                        if not path.is_file():
                            engine.synthesize(request.text,temporary)
                            read_audio(temporary)
                            temporary.replace(path)
                    finally: temporary.unlink(missing_ok=True)
                    audio = read_audio(path)
                    with self._condition:
                        if (not self._closed and request.generation == self._generation
                                and request.sequence == self._request_sequence
                                and request.expires_at >= time.perf_counter()):
                            self._play = (request,audio)
                            self._condition.notify_all()
                except Exception as exc:
                    self.events.put({"type":"speech_error","text":f"Không tổng hợp được câu động: {exc}"})
        except Exception as exc:
            self.events.put({"type":"speech_error","text":f"TTS: {exc}. Cảnh báo đã chuẩn bị vẫn dùng được."})
        finally:
            for item in (engine,fast_engine):
                if item is not None and hasattr(item,"close"):
                    item.close()
                if engine is fast_engine: break

    def _playback(self):
        output = None
        try:
            output = self._output_factory()
            # Open and warm the device once, outside the warning path.
            output.play(Audio(bytes(960)))  # 20ms silence at 24kHz mono PCM16.
            deadline = time.perf_counter()+1
            while output.busy() and time.perf_counter()<deadline and not self._closed:
                time.sleep(.005)
            output.stop()
            self.audio_ready.set()
            while True:
                with self._condition:
                    self._condition.wait_for(lambda:self._closed or self._play is not None)
                    if self._closed: return
                    request,audio = self._play
                    self._play = None
                    if request.generation != self._generation: continue
                    # Once queued, a warning waits for the active clip to finish.
                    # Newer requests replace this slot; they never interrupt playback.
                    self._active_key = request.key
                    self._active_sequence = request.sequence
                    self._active_priority = request.priority
                    output.play(audio)
                    submitted_at = time.perf_counter()
                # Never call the event sink while holding the audio lock.
                self.events.put({"type":"feedback_latency","latency_ms":(submitted_at-request.captured_at)*1000,
                                 "dispatch_ms":(submitted_at-request.requested_at)*1000,
                                 "audio_duration_ms":audio.duration*1000,"backend":output.name,
                                 "measurement":"audio_submit"})
                while output.busy():
                    with self._condition:
                        if self._closed or request.generation != self._generation: break
                        self._condition.wait(timeout=.005)
                output.stop()
                with self._condition:
                    if self._active_sequence == request.sequence:
                        self._active_key = None
                        self._active_sequence = None
                        self._active_priority = 0
        except Exception as exc:
            self._output_error = str(exc)
            self.events.put({"type":"speech_error","text":f"Không phát được âm thanh: {exc}. Kiểm tra thiết bị âm thanh rồi mở lại app."})
        finally:
            if output is not None: output.close()

    def close(self):
        with self._condition:
            self.cancel()
            self._closed = True
            self._condition.notify_all()
        self._play_thread.join(timeout=1)
        self._synth_thread.join(timeout=.5)
