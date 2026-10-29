"""Prepare and activate a voice profile; restart Guide afterwards."""
import argparse
from pathlib import Path
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fast_alerts import AlertBank
from obstacle_detector import VIETNAMESE_NAMES
from tts_engines import create_engine
from voice_profiles import APP_DIR,PRESETS,selected_id,validate_bank,activate

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("voice",nargs="?",choices=tuple(PRESETS))
    parser.add_argument("--prepare",action="store_true")
    parser.add_argument("--activate",action="store_true")
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    if args.list:
        for identifier,preset in PRESETS.items():
            try:validate_bank(identifier,VIETNAMESE_NAMES);ready=True
            except ValueError:ready=False
            print(json.dumps({"id":identifier,"name":preset["name"],"ready":ready,"active":identifier==selected_id()},ensure_ascii=False))
        return
    if not args.voice or not (args.prepare or args.activate):parser.error("Choose a voice and --prepare or --activate")
    preset=PRESETS[args.voice]
    if args.prepare:
        directory=APP_DIR/preset["bank"]
        bank=AlertBank(directory,VIETNAMESE_NAMES)
        try:validate_bank(args.voice,VIETNAMESE_NAMES)
        except ValueError:
            if preset["voice"]:os.environ["GUIDE_KORVA_VOICE"]=preset["voice"]
            engine=create_engine(preset["engine"])
            try:
                print("Đang tạo bộ cảnh báo "+preset["name"]+"…",flush=True)
                bank.prepare(engine)
            finally:
                if hasattr(engine,"close"):engine.close()
        validate_bank(args.voice,VIETNAMESE_NAMES)
        manifest={"voice":args.voice,"name":preset["name"],"engine":preset["engine"],
                  "clips":len(bank.audio),"complete":True}
        (directory/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
        print("Đã chuẩn bị "+preset["name"],flush=True)
    if args.activate:
        activate(args.voice,VIETNAMESE_NAMES)
        print("Đã chọn "+preset["name"]+". Đóng app cũ rồi chạy python main.py.",flush=True)

if __name__=="__main__":main()
