"""Prepare offline alert WAVs, or measure complete synthesis (not streaming first audio)."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feedback import APP_DIR, CLIPS, wav_seconds
from tts_engines import create_engine

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("--engine", choices=("windows","vieneu","korva"), required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--repeat", type=int, default=3)
    args=p.parse_args()
    started=time.perf_counter()
    engine=create_engine(args.engine)
    load_ms=(time.perf_counter()-started)*1000
    directory=APP_DIR/"assets/audio"/args.engine
    directory.mkdir(parents=True,exist_ok=True)
    for key,text in CLIPS.items():
        target=directory/(key+".wav")
        engine.synthesize(text,target)
        wav_seconds(target)
        print("Prepared:",key,flush=True)
    if args.prepare_only: return
    samples=[]
    phrases=["Có người gần ở phía trước.","Có xe máy bên trái.","Chưa nhìn rõ. Hãy dừng để kiểm tra."]
    for _ in range(args.repeat):
        for text in phrases:
            started=time.perf_counter()
            path=directory/"benchmark.wav"
            engine.synthesize(text,path)
            elapsed=(time.perf_counter()-started)*1000
            duration=wav_seconds(path)
            samples.append({"text":text,"full_synthesis_ms":round(elapsed,2),"duration_s":round(duration,2),
                            "rtf":round(elapsed/1000/max(duration,.01),3)})
    result={"engine":args.engine,"load_ms":round(load_ms,2),"measurement":"full WAV synthesis; not first-audio latency",
            "samples":samples}
    destination=APP_DIR/"reports"/("tts_"+args.engine+".json")
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(destination)

if __name__=="__main__": main()
