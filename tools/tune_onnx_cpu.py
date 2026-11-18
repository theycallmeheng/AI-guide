"""Replay identical frames to measure CPU thread settings without changing model weights."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from cpu_runtime import configure_onnx_cpu
from obstacle_detector import analyze_frame

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--model',default='models/results/best.onnx')
    p.add_argument('--samples',type=int,default=60)
    p.add_argument('--threads',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--sizes',type=int,nargs='+',default=[320,480])
    p.add_argument('--output',default='reports/onnx_cpu_tuning.json')
    args=p.parse_args()
    assert args.samples > 0
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    model=YOLO(args.model,task='detect')
    model.predict(np.zeros((480,640,3),np.uint8),imgsz=480,verbose=False,device='cpu')
    rows=[]
    for threads in args.threads:
        assert configure_onnx_cpu(model,args.model,threads)
        torch.set_num_threads(1)
        for size in args.sizes:
            cap=cv2.VideoCapture(args.source)
            timing=[]
            try:
                ok,frame=cap.read()
                assert ok
                for _ in range(5): analyze_frame(frame,model,.3,size,'cpu',render=False)
                for _ in range(args.samples):
                    ok,frame=cap.read()
                    if not ok: break
                    start=time.perf_counter()
                    analyze_frame(frame,model,.3,size,'cpu',render=False)
                    timing.append((time.perf_counter()-start)*1000)
            finally:cap.release()
            assert timing
            row=dict(threads=threads,imgsz=size,samples=len(timing),fps=round(1000/np.mean(timing),2),p95_ms=round(float(np.percentile(timing,95)),2))
            rows.append(row);print(json.dumps(row),flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(model=args.model,source=args.source,torch_threads=1,opencv_threads=1,
        method='Same sequential frames, five warmups. Detection only; no transport/UI/audio. Accuracy not evaluated.',results=rows),indent=2),encoding='utf-8')

if __name__=='__main__':main()
