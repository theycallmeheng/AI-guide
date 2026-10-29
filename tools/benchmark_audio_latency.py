"""Measure software audio submission silently; physical speaker onset is not measured."""
import argparse
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import wave
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from audio_output import Audio, RATE, WaveOutput, read_audio, save_audio
from feedback import SpeechWorker, APP_DIR

class QuietOutput(WaveOutput):
    def play(self,audio):
        super().play(Audio(bytes(len(audio.pcm))))

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("--samples",type=int,default=10)
    args=p.parse_args()
    if args.samples<1:p.error("--samples must be positive")
    events=queue.Queue()
    worker=SpeechWorker(events,output_factory=QuietOutput)
    current=[]
    try:
        if not worker.audio_ready.wait(5):raise RuntimeError("Audio output not ready")
        if worker.bank.compose("Person|gần|phía trước",3) is None:raise RuntimeError("Run tools/prepare_fast_alerts.py first")
        for _ in range(args.samples):
            worker.cancel()
            worker.speak("Dừng lại. Có người gần ở phía trước.",key="Person|gần|phía trước",priority=3)
            deadline=time.perf_counter()+3
            while time.perf_counter()<deadline:
                event=events.get(timeout=3)
                if event["type"]=="feedback_latency":
                    current.append(event["dispatch_ms"]);break
                if event["type"]=="speech_error":raise RuntimeError(event["text"])
            else:raise TimeoutError("No playback measurement")
            worker.cancel()
            time.sleep(.05)
    finally:worker.close()
    import winsound
    legacy=[]
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"quiet.wav"
        save_audio(Audio(bytes(RATE//50*2)),path)
        for _ in range(args.samples):
            t=time.perf_counter()
            winsound.PlaySound(str(path),winsound.SND_FILENAME|winsound.SND_ASYNC|winsound.SND_NODEFAULT)
            legacy.append((time.perf_counter()-t)*1000)
            winsound.PlaySound(None,0)
            time.sleep(.05)
    original=APP_DIR/"assets/audio/windows/stop.wav"
    with wave.open(str(original),"rb") as wav:
        original_seconds=wav.getnframes()/wav.getframerate()
    summary=lambda data:{"samples":len(data),"p50_ms":round(float(np.percentile(data,50)),3),
                         "p95_ms":round(float(np.percentile(data,95)),3),"values_ms":[round(v,3) for v in data]}
    report={"measurement":"software submission, no microphone; test PCM is silent",
            "persistent_worker":summary(current),"previous_PlaySound":summary(legacy),
            "stop_original_ms":round(original_seconds*1000,2),
            "stop_trimmed_ms":round(read_audio(original).duration*1000,2)}
    target=APP_DIR/"reports/audio_latency_improved.json"
    target.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print(target)

if __name__=="__main__":main()
