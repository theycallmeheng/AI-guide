"""Compare warmed, sequential end-to-end analysis throughput on identical video frames."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from obstacle_detector import analyze_frame
from pipeline import APP_DIR, default_model

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Recorded video path")
    p.add_argument("--model", action="append", help="Repeat to compare PT and exported ONNX")
    p.add_argument("--sizes", type=int, nargs="+", default=[320,480,640])
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--output", type=Path, default=APP_DIR/"reports/runtime_benchmark.json")
    args = p.parse_args()
    if args.samples < 1: p.error("--samples must be positive")
    import torch
    torch.set_num_threads(args.threads)
    from ultralytics import YOLO
    rows = []
    for model_path in args.model or [str(default_model())]:
        model = YOLO(model_path)
        for size in args.sizes:
            cap = cv2.VideoCapture(args.source)
            timings, decode = [], []
            try:
                ok, frame = cap.read()
                if not ok: raise ValueError("Cannot read source video")
                for _ in range(3): analyze_frame(frame, model, .3, size, args.device)
                torch.set_num_threads(args.threads)  # Ultralytics setup may override the initial value.
                for _ in range(args.samples):
                    t = time.perf_counter()
                    ok, frame = cap.read()
                    if not ok: break
                    decode.append((time.perf_counter()-t)*1000)
                    t = time.perf_counter()
                    analyze_frame(frame, model, .3, size, args.device)
                    timings.append((time.perf_counter()-t)*1000)
            finally: cap.release()
            if not timings: raise ValueError("Video has no frames after warmup")
            row = {"model": str(model_path), "imgsz": size, "samples": len(timings),
                   "analysis_fps": round(1000/np.mean(timings),2),
                   "p50_analysis_ms": round(float(np.percentile(timings,50)),2),
                   "p95_analysis_ms": round(float(np.percentile(timings,95)),2),
                   "p95_decode_ms": round(float(np.percentile(decode,95)),2)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {"hardware": {"cpu": platform.processor(), "logical_cores": os.cpu_count(),
                          "torch": torch.__version__, "cuda": torch.cuda.is_available(), "device": args.device, "threads": args.threads},
              "method": "3 warmups; same contiguous frames; annotations included; camera transport, display and speech excluded",
              "results": rows}
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Report:", args.output)

if __name__ == "__main__": main()
