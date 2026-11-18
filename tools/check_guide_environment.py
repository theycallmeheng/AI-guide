"""Load all selected adapters and predict one explicit video frame before a long job."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guide_core.config import Config
from guide_core.engine import GuideEngine
from guide_core.types import Frame
import cv2
import torch

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/guide_kaggle.json");p.add_argument("--video",required=True)
    p.add_argument("--report",default="compatibility.json")
    a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    cfg=Config.read(a.config)
    cfg.mode = "offline"  # The compatibility probe must execute every enabled adapter.
    cap=cv2.VideoCapture(a.video);ok,img=cap.read();cap.release()
    if not ok: raise RuntimeError("Explicit video cannot be decoded")
    engine=GuideEngine(cfg)
    try:
        engine.load()
        scene,decision,results=engine.process(Frame.prepare(img,"compatibility",0,0,cfg.max_side,cfg.mirror_input))
        failures=engine.failures|{r.model_id:r.error for r in results if r.error}
        report={"cuda":torch.cuda.is_available(),"torch":torch.__version__,
                "models":[{"id":r.model_id,"ms":(r.finished_at-r.started_at)*1000,"error":r.error} for r in results],
                "versions":{n:importlib.metadata.version(n) for n in ["ultralytics","transformers","opencv-python","lap","imageio-ffmpeg"]},
                "failures":failures,"action":decision.action.value}
        Path(a.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps(report,indent=2))
        if failures: raise RuntimeError("Compatibility check failed; inspect failures before continuing")
    finally: engine.close()

