from dataclasses import replace
import cv2
import numpy as np
from .types import Scene
from .models import canonical

CONTEXT = {"building", "road", "sidewalk", "floor", "traffic sign", "pedestrian crosswalk", "sky"}
def is_hazard(detection):
    return detection.label not in CONTEXT

def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda b:max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(area(a)+area(b)-inter,1e-9)

def merge_detections(detections):
    merged=[]
    for obj in sorted(detections,key=lambda d:-d.confidence):
        obj=replace(obj,label=canonical(obj.label))
        duplicate=next((i for i,d in enumerate(merged) if d.label==obj.label and iou(d.box,obj.box)>.55),None)
        if duplicate is None:
            merged.append(obj)
        else:
            old=merged[duplicate]
            merged[duplicate]=replace(old,sources=tuple(sorted(set(old.sources+obj.sources))))
    return merged

def fuse(frame, results):
    detections, semantic, depth, errors=[],None,None,[]
    detector_valid=False
    for result in results:
        if result.frame_key != frame.key or result.geometry != frame.geometry:
            raise ValueError("Cross-frame or incompatible-geometry fusion rejected.")
        if result.skipped:
            continue
        if result.error:
            errors.append(result.model_id+": "+result.error)
            continue
        if result.role=="detector":
            detector_valid=True
            for d in result.payload:
                h,w=frame.bgr.shape[:2]
                coords=np.asarray(d.box)
                if (np.isfinite(coords).all() and 0<=d.confidence<=1 and
                    0<=coords[0]<coords[2]<=w+1 and 0<=coords[1]<coords[3]<=h+1):
                    detections.append(d)
                else:
                    errors.append(result.model_id+": invalid detection")
        elif result.role=="semantic":
            if semantic is not None: raise ValueError("Ambiguous semantic providers")
            if result.payload["classes"].shape!=frame.bgr.shape[:2]: raise ValueError("Semantic geometry mismatch")
            semantic=result.payload
        elif result.role=="depth":
            if depth is not None: raise ValueError("Ambiguous depth providers")
            if result.payload["relative_nearness"].shape!=frame.bgr.shape[:2]: raise ValueError("Depth geometry mismatch")
            if result.payload.get("unit")!="relative": raise ValueError("Unsupported depth unit")
            depth=result.payload
    gray=cv2.cvtColor(frame.bgr,cv2.COLOR_BGR2GRAY)
    return Scene(frame,merge_detections(detections),semantic,depth,errors,
                 blur_score=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                 brightness=float(gray.mean()),detector_valid=detector_valid)

