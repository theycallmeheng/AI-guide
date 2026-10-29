"""Export a local YOLO detector to dynamic ONNX for an explicit benchmark."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import default_model

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("--model",default=str(default_model()))
    args=p.parse_args()
    from ultralytics import YOLO
    result=YOLO(args.model).export(format="onnx",imgsz=640,dynamic=True,simplify=False,opset=17,device="cpu")
    print(result)

if __name__=="__main__": main()
