"""Run with .venv-neural Python; uses existing Korva Khánh Vy assets."""
from pathlib import Path
import sys
import json
import wave
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guide_core.types import MESSAGES
from tts_engines import KorvaVoice

def main():
    out=Path("assets/audio/guide_core/khanh_vy")
    out.mkdir(parents=True,exist_ok=True)
    phrases={a.value.lower():text for a,text in MESSAGES.items()}
    phrases["narrow"]="Lối đi hẹp. Đi chậm lại."
    phrases["forward_corridor_blocked"] = "Dừng lại. Lối phía trước bị chắn."
    phrases["transverse_step_ahead"] = "Dừng lại. Có bậc hoặc mép bục phía trước."
    phrases["scan_for_bypass"] = "Đứng yên. Hướng camera sang hai bên để tìm lối vòng."
    phrases["bypass_left"] = "Có lối vòng bên trái. Lệch nhẹ sang trái."
    phrases["bypass_right"] = "Có lối vòng bên phải. Lệch nhẹ sang phải."
    engine=None
    for name,text in phrases.items():
        path=out/(name+".wav")
        if path.exists(): continue
        if engine is None: engine=KorvaVoice()
        engine.synthesize(text,path)
        # Uniform portable PCM16 mono format.
        import soundfile as sf
        samples,rate=sf.read(path,dtype="float32")
        if samples.ndim==2: samples=samples.mean(1)
        if rate!=24000:
            samples=np.interp(np.arange(round(len(samples)*24000/rate))*rate/24000,np.arange(len(samples)),samples)
        pcm=(np.clip(samples,-1,1)*32767).astype("<i2")
        with wave.open(str(path),"wb") as f:
            f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(pcm.tobytes())
        print(name,flush=True)
    (out/"manifest.json").write_text(json.dumps({"voice":"khanh_vy","engine":"KorvaTTS","phrases":phrases},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__": main()

