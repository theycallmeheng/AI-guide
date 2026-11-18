"""Paired-frame detector comparison and model ablations; no default model replacement."""
from pathlib import Path
import argparse
import copy
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guide_core.config import Config
from guide_core.demo import run
from guide_core.evaluate import read_run

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/guide_kaggle.json")
    p.add_argument("--video",required=True);p.add_argument("--output",required=True)
    p.add_argument("--seconds",type=float,help="Limit the preview. With labels, the default is the full video.")
    p.add_argument("--annotations")
    p.add_argument("--sizes",type=int,nargs="+",default=[480])
    args=p.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        p.error("--seconds must be positive")
    seconds = args.seconds if args.seconds is not None or args.annotations else 10
    base=Config.read(args.config)
    base.mode="offline"
    detectors=[m for m in base.models if m.role=="detector"]
    experiments=[]
    for detector in detectors:
        for size in args.sizes:
            cfg=copy.deepcopy(base)
            for m in cfg.models:
                m.enabled=m.id==detector.id
                if m.role=="detector": m.imgsz=size
            experiments.append((detector.id+"_only_"+str(size),cfg))
    primary=next(model.id for model in detectors if model.enabled)
    for name,roles,second in (("plus_semantic",{"detector","semantic"},False),
                             ("plus_depth",{"detector","semantic","depth"},False),
                             ("plus_second_detector",{"detector","semantic","depth"},True)):
        cfg=copy.deepcopy(base)
        for m in cfg.models:
            m.enabled=m.role in roles and (m.role!="detector" or m.id==primary or second)
        experiments.append((name,cfg))
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    summaries={}
    reference=None
    for name,cfg in experiments:
        report=run(cfg,args.video,output/name,seconds)
        if reference is None: reference=report["frame_sequences"]
        if report["frame_sequences"]!=reference: raise RuntimeError("Paired-frame comparison failed")
        summary={"fps":report["unique_inference_frames_per_second"],"pipeline_ms":report["pipeline_ms"],
                 "model_ms":report["model_ms"],"load_failures":report["load_failures"],
                 "actions":report["action_counts"],"quality":"unscored_without_reviewed_annotations"}
        if args.annotations:
            _, summary["evaluation"], _, _ = read_run(args.annotations, output/name)
            summary["quality"] = "scored_with_reviewed_annotations"
        summaries[name]=summary
        (output/"comparison.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    print("Saved paired-frame comparison; default model unchanged.")
if __name__=="__main__": main()

