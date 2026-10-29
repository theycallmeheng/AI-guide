"""Small Vietnamese word bank: describe known hazards without runtime synthesis."""
import hashlib
from pathlib import Path
import wave
from audio_output import read_audio, join_audio, save_audio

FIXED = {"stop": "Dừng lại.", "unclear": "Chưa nhìn rõ.",
         "caution_left": "Cẩn thận bên trái.", "caution_right": "Cẩn thận bên phải.",
         "caution_front": "Cẩn thận phía trước.", "disconnected": "Mất camera. Dừng lại.",
         "slow_down": "Đi chậm lại.", "go_forward": "Đi thẳng, chú ý quan sát.",
         "turn_left": "Rẽ trái sau khi kiểm tra lối đi.",
         "turn_right": "Rẽ phải sau khi kiểm tra lối đi."}
DIRECTIONS = ("bên trái", "bên phải", "phía trước")
BANDS = ("gần", "vừa", "xa")

def word_id(text):
    return "word_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]

class AlertBank:
    def __init__(self, directory, labels):
        self.directory = Path(directory)
        self.labels = {k.lower().replace("_"," "): v for k,v in labels.items()}
        self.audio = {}
        self.load_existing()

    def load_existing(self):
        texts = set(self.labels.values()) | set(DIRECTIONS) | set(BANDS)
        for key in list(FIXED) + [word_id(t) for t in texts]:
            path = self.directory/(key+".wav")
            if path.is_file() and key not in self.audio:
                try: self.audio[key] = read_audio(path)
                except (OSError, ValueError, wave.Error): pass

    def prepare(self, engine, stopped=lambda: False):
        self.directory.mkdir(parents=True, exist_ok=True)
        entries = list(FIXED.items())
        entries += [(word_id(t), t+".") for t in sorted(set(self.labels.values()) | set(DIRECTIONS) | set(BANDS))]
        for key,text in entries:
            if stopped(): return
            path = self.directory/(key+".wav")
            if key not in self.audio:
                temporary = path.with_suffix(".tmp.wav")
                try:
                    engine.synthesize(text, temporary)
                    audio = read_audio(temporary)
                    save_audio(audio, temporary)  # Store trimmed PCM in the bank itself.
                    temporary.replace(path)
                    self.audio[key] = audio
                finally: temporary.unlink(missing_ok=True)

    def compose(self, key, priority):
        if not key: return None
        if key == "route|front":
            return self.audio.get("stop" if priority >= 3 else "go_forward")
        parts = key.split("|")
        command = None
        if len(parts) == 4 and parts[0] in ("go_forward","turn_left","turn_right"):
            command,parts = parts[0],parts[1:]
        if len(parts) != 3: return None
        category,band,direction = parts
        label = self.labels.get(category.lower().replace("_"," "))
        if label is None or direction not in DIRECTIONS or band not in BANDS: return None
        details = [word_id(label), word_id(direction), word_id(band)]
        keys = ["slow_down"] + details
        if priority >= 3:
            keys = ["stop",word_id(label),word_id(direction)]
        elif command in ("turn_left","turn_right"):
            keys = ["stop",command] + details
        elif command == "go_forward":
            keys = [command] + details
        if not all(k in self.audio for k in keys): return None
        return join_audio([self.audio[k] for k in keys])
