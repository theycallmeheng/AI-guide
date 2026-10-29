"""Generate comparable Vietnamese voice previews; does not change the active voice."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audio_output import read_audio,save_audio

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--voices",nargs="+",default=["khanh_vy","hoang_nam"])
    args=parser.parse_args()
    from korvatts import TTS
    tts=TTS(device="cpu",num_threads=4)
    directory=Path(__file__).resolve().parents[1]/"assets/audio/previews"
    directory.mkdir(parents=True,exist_ok=True)
    for voice in args.voices:
        if voice not in tts.list_voices():raise ValueError("Voice unavailable: "+voice)
        path=directory/(voice+".wav")
        audio,_=tts.synthesize("Dừng lại. Có người phía trước. Cẩn thận xe máy bên trái.",voice=voice,lang="vi",total_steps=16,speed=1.05)
        temporary=path.with_suffix(".tmp.wav")
        try:
            tts.save_audio(audio,str(temporary))
            save_audio(read_audio(temporary),path)
        finally:temporary.unlink(missing_ok=True)
        print(str(path),flush=True)

if __name__=="__main__":main()
