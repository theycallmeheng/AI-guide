"""Offline speech adapters, each owned by one synthesis thread."""
import os
from pathlib import Path
import hashlib
import threading

_FETCH_LOCK = threading.Lock()


def _local_vieneu_fetch(repo, files, subfolder):
    """Materialize ONNX and external data together, preserving ORT path validation.

    HF snapshot symlinks can resolve graph and .data into different blob folders.
    A regular local download keeps external tensors inside the graph directory.
    """
    from huggingface_hub import hf_hub_download
    directory = Path(__file__).resolve().parent / "models/tts" / hashlib.sha256(repo.encode()).hexdigest()[:12]
    last = None
    for name in files:
        try:
            relative = f"{subfolder}/{name}" if subfolder else name
            target = directory / relative
            if not target.is_file():
                hf_hub_download(repo, relative, local_dir=str(directory))
            last = target
        except Exception:
            if not name.endswith(".json"):
                raise
    if last is None:
        raise RuntimeError("Không tải được tệp mô hình VieNeu.")
    return last.parent

class WindowsVoice:
    def __init__(self):
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        self.synth = SpeechSynthesizer()
        voice = next((v for v in SpeechSynthesizer.all_voices if v.language.lower().startswith("vi")), None)
        if voice is None:
            raise RuntimeError("Chưa có giọng Việt Windows. Cài Microsoft An hoặc chọn VieNeu/Korva theo README.")
        self.synth.voice = voice
        self.name = voice.display_name

    def synthesize(self, text, path):
        from winrt.windows.storage.streams import DataReader
        stream = self.synth.synthesize_text_to_stream_async(text).get()
        reader = DataReader(stream.get_input_stream_at(0))
        try:
            size = int(stream.size)
            reader.load_async(size).get()
            data = bytearray(size)
            reader.read_bytes(data)
            Path(path).write_bytes(data)
        finally:
            reader.close()
            stream.close()

    def close(self):
        self.synth.close()

class KorvaVoice:
    def __init__(self):
        from korvatts import TTS
        assets = os.environ.get("KORVATTS_ASSETS_DIR")
        self.tts = TTS(assets_dir=assets, device="cpu", num_threads=4) if assets else TTS(device="cpu", num_threads=4)
        self.name = "KorvaTTS ONNX"

    def synthesize(self, text, path):
        audio, _ = self.tts.synthesize(text, voice=os.environ.get("GUIDE_KORVA_VOICE", "khanh_vy"),
                                      lang="vi", total_steps=16, speed=1.05)
        self.tts.save_audio(audio, str(path))

class VieNeuVoice:
    def __init__(self):
        from vieneu import Vieneu
        from vieneu._v3_turbo_engine.onnx_runtime_lite import OnnxV3LiteEngine
        precision = os.environ.get("GUIDE_VIENEU_PRECISION", "int8")
        with _FETCH_LOCK:
            original_fetch = OnnxV3LiteEngine._fetch
            OnnxV3LiteEngine._fetch = staticmethod(_local_vieneu_fetch)
            try:
                self.tts = Vieneu(mode="v3turbo", device="cpu", backend="onnx", precision=precision, threads=4)
            finally:
                OnnxV3LiteEngine._fetch = staticmethod(original_fetch)
        self.name = f"VieNeu v3 Turbo ONNX {precision}"

    def synthesize(self, text, path):
        audio = self.tts.infer(text, voice=os.environ.get("GUIDE_VIENEU_VOICE", "Mai Anh"))
        self.tts.save(audio, str(path))

def create_engine(name):
    constructors = {"windows": WindowsVoice, "korva": KorvaVoice, "vieneu": VieNeuVoice}
    if name not in constructors:
        raise ValueError(f"TTS engine không hợp lệ: {name}")
    return constructors[name]()
