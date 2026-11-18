"""Adapters return original analysis-frame coordinates; all weights are local."""
from __future__ import annotations
import time
from pathlib import Path
from typing import Protocol
import cv2
import numpy as np
from .types import Detection, ModelResult

ALIASES = {"bike": "bicycle", "electrical pole": "pole", "traffic cone": "cone",
           "guard rail": "barrier", "teraffic barrel": "barrier", "traffic barrel": "barrier",
           "stair": "stairs", "steps": "stairs", "plant pot": "plant pot"}
def canonical(name, mapping=None):
    name = " ".join(str(name).lower().replace("_", " ").split())
    name = (mapping or {}).get(name, name)
    return ALIASES.get(name, name)

class Adapter(Protocol):
    def load(self): ...
    def predict(self, frame): ...
    def close(self): ...

REGISTRY = {}
def register(name):
    def decorate(cls):
        REGISTRY[name] = cls
        return cls
    return decorate

def resolve_device(value):
    import torch
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if str(value).isdigit():
        value = "cuda:" + str(value)
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; enable Kaggle GPU or explicitly select CPU.")
    return str(value)

class BaseAdapter:
    def __init__(self, spec):
        self.spec = spec
        self.device = resolve_device(spec.device)
        self.model = None

    def close(self):
        self.model = None

SIDEWALK_LABELS = {"sidewalk", "walkway", "footpath", "pavement", "path", "pedestrian walkway"}
ALLEY_LABELS = SIDEWALK_LABELS | {"dirt track", "road"}
INDOOR_LABELS = {"floor", "flooring", "rug"}
STRUCTURE_LABELS = {
    "wall", "fence", "door", "stairs", "stairway", "escalator", "step", "building", "railing",
    "bannister", "person", "car", "bus", "truck", "van", "minibike", "bicycle", "tree",
    "column", "pole", "streetlight", "signboard", "cabinet", "chair", "table", "bench",
    "box", "barrel", "ashcan", "water", "river", "swimming pool",
}
STAIR_LABELS = {"stairs", "stairway", "escalator", "step"}
GROUND_BOUND_LABELS = {
    "stairs", "step", "pothole", "manhole", "chair", "bench", "table",
    "dog", "bicycle", "motorcycle", "car", "bus", "truck", "van", "minibike",
    "cone", "dustbin", "plant pot", "person",
}

def _classify_zone(label, box, hw):
    if label in GROUND_BOUND_LABELS:
        return "ground"
    h, w = hw
    y2_limit = h * (.45 if h > w else .78)
    return "head" if box[1] < h * .48 and box[3] < y2_limit else "ground"

@register("yolo")
class YoloAdapter(BaseAdapter):
    def precision(self):
        from ultralytics.cfg import DEFAULT_CFG_DICT
        return {"quantize":16 if self.device.startswith("cuda") else 32} if "quantize" in DEFAULT_CFG_DICT else {"half":self.device.startswith("cuda")}

    def load(self):
        import torch
        from ultralytics import YOLO
        p = Path(self.spec.path)
        if not (p.is_file() or p.is_dir()):
            raise FileNotFoundError(self.spec.path)
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        task = "detect" if str(p).lower().endswith((".onnx", "_openvino_model")) else None
        self.model = YOLO(str(p), task=task)
        self.model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=self.spec.imgsz,
                           device=self.device, verbose=False, **self.precision())
        if str(p).lower().endswith(".onnx") and self.device == "cpu":
            try:
                from cpu_runtime import configure_onnx_cpu
                configure_onnx_cpu(self.model, str(p), 4)
            except Exception:
                pass

    def predict(self, frame):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.set_device(self.device)
        h_img = frame.bgr.shape[0]
        result = self.model.predict(frame.bgr, imgsz=self.spec.imgsz, conf=self.spec.confidence,
                                    device=self.device, verbose=False, **self.precision())[0]
        detections = []
        if result.boxes is not None:
            for row in result.boxes.data.cpu().numpy():
                label = canonical(result.names[int(row[5])], self.spec.label_map)
                box = tuple(map(float, row[:4]))
                zone = _classify_zone(label, box, frame.bgr.shape[:2])
                detections.append(Detection(label, float(row[4]), box, (self.spec.id,), zone=zone))
        return detections

class HFAdapter(BaseAdapter):
    def load(self):
        import torch
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, AutoModelForDepthEstimation
        path = Path(self.spec.path)
        if not (path / "config.json").is_file():
            raise FileNotFoundError("Complete local HF folder required: " + str(path))
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        self.processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
        model_cls = AutoModelForSemanticSegmentation if self.spec.role == "semantic" else AutoModelForDepthEstimation
        self.model = model_cls.from_pretrained(str(path), local_files_only=True).to(self.device).eval()
        self.labels = {int(k): canonical(v, self.spec.label_map) for k,v in self.model.config.id2label.items()} if self.spec.role == "semantic" else {}
        if self.spec.role == "semantic":
            self._group_ids = {
                name: torch.tensor([k for k, v in self.labels.items() if v in vocab],
                                   dtype=torch.long, device=self.device)
                for name, vocab in (
                    ("sidewalk", SIDEWALK_LABELS),
                    ("alley", ALLEY_LABELS),
                    ("indoor", INDOOR_LABELS),
                    ("structure", STRUCTURE_LABELS),
                    ("stairs", STAIR_LABELS),
                )
            }

    def _target_hw(self, hw):
        h, w = hw
        scale = min(1.0, self.spec.imgsz / max(h, w, 1))
        stride = 32 if self.spec.role == "semantic" else 14
        th = max(stride, int(round(h * scale / stride)) * stride)
        tw = max(stride, int(round(w * scale / stride)) * stride)
        return th, tw

    def forward(self, frame):
        import torch
        if self.device.startswith("cuda"):
            torch.cuda.set_device(self.device)
        rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
        th, tw = self._target_hw(rgb.shape[:2])
        # Preserve camera aspect ratio so flat ground is never distorted into vertical walls.
        kwargs = {"images": rgb, "return_tensors": "pt", "size": {"height": th, "width": tw}}
        if self.spec.role == "depth":
            kwargs["keep_aspect_ratio"] = False
        inputs = self.processor(**kwargs)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
            return self.model(**inputs)

@register("segformer")
class SemanticAdapter(HFAdapter):
    def predict(self, frame):
        import torch
        import torch.nn.functional as F
        target_hw = frame.bgr.shape[:2]
        with torch.inference_mode():
            logits = self.forward(frame).logits.float()
            # Compute softmax at native 1/4 feature resolution instead of upsampling all 150 channels.
            probs = logits.softmax(1)
            group_order = ("sidewalk", "alley", "indoor", "structure", "stairs")
            grouped = []
            for key in group_order:
                idx = self._group_ids[key]
                if idx.numel():
                    grouped.append(probs.index_select(1, idx).sum(1, keepdim=True))
                else:
                    grouped.append(torch.zeros((1, 1, *probs.shape[2:]), dtype=probs.dtype, device=probs.device))
            group_tensor = F.interpolate(torch.cat(grouped, dim=1), size=target_hw,
                                         mode="bilinear", align_corners=False)[0].cpu().numpy()
            group_probs = {key: group_tensor[i] for i, key in enumerate(group_order)}

            peak = probs[0].amax(dim=(1, 2))
            active = torch.nonzero(peak > 0.04, as_tuple=False).squeeze(1)
            if active.numel() == 0:
                active = peak.topk(min(8, probs.shape[1])).indices
            up_active = F.interpolate(probs.index_select(1, active), size=target_hw,
                                      mode="bilinear", align_corners=False)
            confidence, local_idx = up_active.max(1)
            classes = active[local_idx[0]].cpu().numpy().astype(np.int16)
            conf_np = confidence[0].cpu().numpy()
        return {"classes": classes, "confidence": conf_np, "labels": self.labels, "group_probs": group_probs}

@register("depth_anything")
class DepthAdapter(HFAdapter):
    def predict(self, frame):
        import torch
        depth = self.forward(frame).predicted_depth.float().unsqueeze(1)
        depth = torch.nn.functional.interpolate(depth, size=frame.bgr.shape[:2], mode="bilinear", align_corners=False)[0,0]
        raw = depth.detach().cpu().numpy()
        valid = np.isfinite(raw)
        if valid.sum() < raw.size*.95:
            return {"relative_nearness": np.zeros_like(raw), "valid": np.zeros_like(valid), "unit": "relative"}
        lo, hi = np.percentile(raw[valid], [2,98])
        if hi-lo < 1e-6:
            valid[:] = False
        near = np.clip((raw-lo)/max(hi-lo,1e-6),0,1)
        return {"relative_nearness": np.nan_to_num(near), "valid": valid, "unit": "relative"}

_OV_CORE = None
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _get_ov_core():
    global _OV_CORE
    if _OV_CORE is None:
        import os
        cache_dir = Path("models/openvino/.ov_cache").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("cl_cache_dir", str(cache_dir))
        import openvino as ov
        _OV_CORE = ov.Core()
        try:
            _OV_CORE.set_property({"CACHE_DIR": str(cache_dir)})
        except Exception:
            pass
    return _OV_CORE

def resolve_openvino_device(value, default_target="GPU"):
    raw = str(value or "auto").strip()
    for prefix in ("openvino:", "intel:"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):]
    if raw.lower() in {"auto", ""}:
        target = default_target
    elif raw.lower() in {"igpu", "gpu"}:
        target = "GPU"
    elif raw.lower() == "cpu":
        target = "CPU"
    else:
        target = raw.upper()
    try:
        available = _get_ov_core().available_devices
    except Exception:
        available = ["CPU"]
    if target.startswith("GPU"):
        if target in available:
            chosen = target
        else:
            gpu_devs = [d for d in available if d.startswith("GPU")]
            chosen = gpu_devs[0] if gpu_devs else "CPU"
    else:
        chosen = "CPU" if target not in available else target
    return f"openvino:{chosen}", chosen

def _letterbox_norm(bgr, size):
    h, w = bgr.shape[:2]
    scale = size / max(h, w, 1)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:nh, :nw] = norm
    return np.transpose(canvas, (2, 0, 1))[None, ...], (nh, nw)

class OpenVinoBaseAdapter:
    default_ov_target = "GPU"
    def __init__(self, spec):
        import threading
        self.spec = spec
        self.device, self.ov_device = resolve_openvino_device(spec.device, self.default_ov_target)
        self.compiled = None
        self.infer_request = None
        self._lock = threading.Lock()

    def _compile_xml(self, xml_path):
        core = _get_ov_core()
        props = {"PERFORMANCE_HINT": "LATENCY"}
        if self.ov_device == "CPU":
            props["INFERENCE_NUM_THREADS"] = "4"
        try:
            self.compiled = core.compile_model(str(xml_path), self.ov_device, props)
        except Exception:
            self.ov_device = "CPU"
            self.device = "openvino:CPU"
            props["INFERENCE_NUM_THREADS"] = "4"
            self.compiled = core.compile_model(str(xml_path), "CPU", props)
        self.infer_request = self.compiled.create_infer_request()
        self.output_port = self.compiled.output(0)

    def close(self):
        self.infer_request = None
        self.compiled = None

@register("openvino_yolo")
class OpenVinoYoloAdapter(OpenVinoBaseAdapter):
    default_ov_target = "CPU"

    def load(self):
        p = Path(self.spec.path)
        xml_files = sorted(p.glob("*.xml")) if p.is_dir() else []
        if not xml_files:
            raise FileNotFoundError("OpenVINO YOLO directory with .xml required: " + str(p))
        meta_files = sorted(p.glob("*.yaml"))
        self.names = {}
        self.ov_imgsz = self.spec.imgsz
        self.end2end = False
        if meta_files:
            import yaml
            meta = yaml.safe_load(meta_files[0].read_text(encoding="utf-8")) or {}
            raw_names = meta.get("names", {})
            if isinstance(raw_names, dict):
                self.names = {int(k): str(v) for k, v in raw_names.items()}
            elif isinstance(raw_names, list):
                self.names = {i: str(v) for i, v in enumerate(raw_names)}
            sz = meta.get("imgsz")
            if isinstance(sz, (list, tuple)) and sz:
                self.ov_imgsz = int(sz[0])
            elif isinstance(sz, int):
                self.ov_imgsz = sz
            self.end2end = bool(meta.get("end2end", False))
        self._compile_xml(xml_files[0])
        dummy = np.zeros((1, 3, self.ov_imgsz, self.ov_imgsz), dtype=np.float32)
        self.infer_request.infer({0: dummy})

    def predict(self, frame):
        h, w = frame.bgr.shape[:2]
        sz = self.ov_imgsz
        scale = min(sz / max(h, 1), sz / max(w, 1))
        nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
        dw, dh = (sz - nw) // 2, (sz - nh) // 2
        resized = cv2.resize(frame.bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        canvas[dh:dh + nh, dw:dw + nw] = resized
        blob = np.transpose(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0), (2, 0, 1))[None, ...]
        with self._lock:
            out = self.infer_request.infer({0: blob})[self.output_port]
        pred = out[0] if out.ndim == 3 else out

        detections = []
        conf_th = float(self.spec.confidence)
        if self.end2end or (pred.ndim == 2 and pred.shape[-1] == 6 and pred.shape[0] <= 300):
            keep = pred[pred[:, 4] >= conf_th]
            for row in keep:
                x1 = float(np.clip((row[0] - dw) / scale, 0, w))
                y1 = float(np.clip((row[1] - dh) / scale, 0, h))
                x2 = float(np.clip((row[2] - dw) / scale, 0, w))
                y2 = float(np.clip((row[3] - dh) / scale, 0, h))
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_cls = self.names.get(int(row[5]), str(int(row[5])))
                label = canonical(raw_cls, self.spec.label_map)
                zone = _classify_zone(label, (x1, y1, x2, y2), (h, w))
                detections.append(Detection(label, float(row[4]), (x1, y1, x2, y2), (self.spec.id,), zone=zone))
            return detections

        # Standard Ultralytics Detect head: (4 + nc, num_anchors)
        if pred.shape[0] > pred.shape[1] and pred.shape[1] <= 128:
            pred = pred.T
        scores = pred[4:]
        cls_ids = scores.argmax(axis=0)
        confs = scores[cls_ids, np.arange(scores.shape[1])]
        mask = confs >= conf_th
        if not np.any(mask):
            return detections
        boxes_cxcywh = pred[:4, mask].T
        confs = confs[mask]
        cls_ids = cls_ids[mask]
        xywh = np.column_stack([
            boxes_cxcywh[:, 0] - 0.5 * boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 1] - 0.5 * boxes_cxcywh[:, 3],
            boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 3],
        ])
        indices = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), conf_th, 0.45)
        if len(indices) == 0:
            return detections
        for idx in np.asarray(indices).reshape(-1):
            bx, by, bw, bh = xywh[idx]
            x1 = float(np.clip((bx - dw) / scale, 0, w))
            y1 = float(np.clip((by - dh) / scale, 0, h))
            x2 = float(np.clip((bx + bw - dw) / scale, 0, w))
            y2 = float(np.clip((by + bh - dh) / scale, 0, h))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_cls = self.names.get(int(cls_ids[idx]), str(int(cls_ids[idx])))
            label = canonical(raw_cls, self.spec.label_map)
            zone = _classify_zone(label, (x1, y1, x2, y2), (h, w))
            detections.append(Detection(label, float(confs[idx]), (x1, y1, x2, y2), (self.spec.id,), zone=zone))
        return detections

@register("openvino_segformer")
class OpenVinoSemanticAdapter(OpenVinoBaseAdapter):
    default_ov_target = "GPU"

    def load(self):
        import json
        path = Path(self.spec.path)
        xml_path = path / "model.xml"
        cfg_path = path / "config.json"
        if not (xml_path.is_file() and cfg_path.is_file()):
            raise FileNotFoundError("OpenVINO SegFormer folder requires model.xml and config.json: " + str(path))
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        meta_path = path / "ov_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        self.ov_imgsz = int(meta.get("imgsz", self.spec.imgsz))
        self.labels = {int(k): canonical(v, self.spec.label_map) for k, v in cfg.get("id2label", {}).items()}
        self._group_ids = {
            name: np.array([k for k, v in self.labels.items() if v in vocab], dtype=np.int64)
            for name, vocab in (
                ("sidewalk", SIDEWALK_LABELS),
                ("alley", ALLEY_LABELS),
                ("indoor", INDOOR_LABELS),
                ("structure", STRUCTURE_LABELS),
                ("stairs", STAIR_LABELS),
            )
        }
        self._compile_xml(xml_path)
        dummy = np.zeros((1, 3, self.ov_imgsz, self.ov_imgsz), dtype=np.float32)
        self.infer_request.infer({0: dummy})

    def predict(self, frame):
        h, w = frame.bgr.shape[:2]
        tensor, (nh, nw) = _letterbox_norm(frame.bgr, self.ov_imgsz)
        with self._lock:
            probs = self.infer_request.infer({0: tensor})[self.output_port]
        fh = max(1, int(np.ceil(nh * probs.shape[2] / self.ov_imgsz)))
        fw = max(1, int(np.ceil(nw * probs.shape[3] / self.ov_imgsz)))
        valid_probs = probs[0, :, :fh, :fw]

        group_probs = {}
        for key, idx in self._group_ids.items():
            if idx.size:
                g_map = valid_probs[idx].sum(axis=0).astype(np.float32)
                group_probs[key] = cv2.resize(g_map, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                group_probs[key] = np.zeros((h, w), dtype=np.float32)

        peak = valid_probs.max(axis=(1, 2))
        active = np.flatnonzero(peak > 0.04)
        if active.size == 0:
            active = np.argsort(peak)[-min(8, valid_probs.shape[0]):]
        sub_hwc = np.transpose(valid_probs[active], (1, 2, 0)).astype(np.float32)
        up = cv2.resize(sub_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[..., None]
        local_idx = up.argmax(axis=2)
        conf_np = np.take_along_axis(up, local_idx[..., None], axis=2)[..., 0]
        classes = active[local_idx].astype(np.int16)
        return {"classes": classes, "confidence": conf_np, "labels": self.labels, "group_probs": group_probs}

@register("openvino_depth")
class OpenVinoDepthAdapter(OpenVinoBaseAdapter):
    default_ov_target = "GPU"

    def load(self):
        import json
        path = Path(self.spec.path)
        xml_path = path / "model.xml"
        if not xml_path.is_file():
            raise FileNotFoundError("OpenVINO Depth folder requires model.xml: " + str(path))
        meta_path = path / "ov_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        self.ov_imgsz = int(meta.get("imgsz", self.spec.imgsz))
        self._compile_xml(xml_path)
        dummy = np.zeros((1, 3, self.ov_imgsz, self.ov_imgsz), dtype=np.float32)
        self.infer_request.infer({0: dummy})

    def predict(self, frame):
        h, w = frame.bgr.shape[:2]
        tensor, (nh, nw) = _letterbox_norm(frame.bgr, self.ov_imgsz)
        with self._lock:
            out = self.infer_request.infer({0: tensor})[self.output_port]
        pred = out[0] if out.ndim == 3 else out
        ph = max(1, round(nh * pred.shape[0] / self.ov_imgsz))
        pw = max(1, round(nw * pred.shape[1] / self.ov_imgsz))
        raw = cv2.resize(pred[:ph, :pw].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        valid = np.isfinite(raw)
        if valid.sum() < raw.size * .95:
            return {"relative_nearness": np.zeros_like(raw), "valid": np.zeros_like(valid), "unit": "relative"}
        lo, hi = np.percentile(raw[valid], [2, 98])
        if hi - lo < 1e-6:
            valid[:] = False
        near = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1)
        return {"relative_nearness": np.nan_to_num(near), "valid": valid, "unit": "relative"}

def predict_result(adapter, frame):
    start = time.perf_counter()
    try:
        payload = adapter.predict(frame)
        error = None
    except Exception as exc:
        payload, error = None, str(exc)
    return ModelResult(adapter.spec.id, adapter.spec.role, frame.key, frame.geometry,
                       start, time.perf_counter(), payload, error)

