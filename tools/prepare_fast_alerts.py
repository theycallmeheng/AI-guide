"""Prepare short offline Vietnamese alert components using the installed Windows voice."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fast_alerts import AlertBank
from obstacle_detector import VIETNAMESE_NAMES
from tts_engines import create_engine

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser()
    parser.add_argument("--engine",choices=("windows","korva","vieneu"),default="windows")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"assets/audio/fast")
    parser.add_argument("--rebuild",action="store_true",help="Regenerate all bank clips with the selected voice")
    args=parser.parse_args()
    bank=AlertBank(args.output,VIETNAMESE_NAMES)
    if args.rebuild:bank.audio.clear()
    engine=create_engine(args.engine)
    try: bank.prepare(engine)
    finally:
        if hasattr(engine,"close"):engine.close()
    print(json.dumps({"clips":len(bank.audio),"directory":str(args.output)},ensure_ascii=False))

if __name__=="__main__":main()
