"""Export YOLO26n/v8n, SegFormer, and Depth Anything V2 to OpenVINO FP16 & INT8 for Intel CPU + Iris Xe iGPU."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox_normalized(bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Aspect-preserving resize + zero-mean letterbox pad to static (1, 3, size, size) NCHW float32."""
    h, w = bgr.shape[:2]
    scale = size / max(h, w, 1)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    canvas[:nh, :nw] = norm
    return np.transpose(canvas, (2, 0, 1))[None, ...], (nh, nw)


def collect_calibration_bgr(limit: int = 48) -> list[np.ndarray]:
    """Gather diverse training-only frames (never validation/test) for NNCF INT8 calibration."""
    images: list[np.ndarray] = []
    train_img_dir = ROOT / "data/training/images/train"
    if train_img_dir.is_dir():
        for p in sorted(train_img_dir.rglob("*.jpg"))[:limit]:
            raw_bytes = np.fromfile(str(p), dtype=np.uint8)
            img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR) if raw_bytes.size else None
            if img is not None:
                images.append(img)
    if len(images) < limit:
        for vid_name in ("train1.mp4", "train2.mp4", "train3.mp4", "train4.mp4", "train5.mp4", "train6.mp4"):
            vp = ROOT / "data/videos" / vid_name
            if not vp.is_file():
                continue
            cap = cv2.VideoCapture(str(vp))
            idx = 0
            while len(images) < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % 45 == 0:
                    images.append(frame)
                idx += 1
            cap.release()
            if len(images) >= limit:
                break
    if not images:
        rng = np.random.default_rng(42)
        images = [rng.integers(30, 220, (480, 640, 3), dtype=np.uint8) for _ in range(16)]
    return images[:limit]


class SegFormerExportWrapper(nn.Module):
    """Wraps SegFormer + native-resolution softmax so OpenVINO iGPU/CPU executes softmax in graph."""
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=pixel_values).logits
        return logits.float().softmax(dim=1)


class DepthExportWrapper(nn.Module):
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=pixel_values).predicted_depth.float()


def export_yolo_openvino(pt_path: Path, out_root: Path, imgsz: int, calib_bgr: list[np.ndarray], do_int8: bool) -> dict:
    import openvino as ov
    from ultralytics import YOLO

    core = ov.Core()
    print(f"[1/3] Exporting YOLO ({pt_path.name}) to OpenVINO FP16 (imgsz={imgsz})...", flush=True)
    exported_dir = Path(YOLO(str(pt_path)).export(format="openvino", imgsz=imgsz, half=True, dynamic=False, device="cpu"))
    fp16_dir = out_root / f"{pt_path.stem}_fp16_openvino_model"
    if fp16_dir.exists():
        shutil.rmtree(fp16_dir)
    shutil.move(str(exported_dir), str(fp16_dir))
    info = {"fp16": str(fp16_dir.relative_to(ROOT).as_posix())}

    if do_int8:
        import nncf
        print(f"      Quantizing YOLO ({pt_path.name}) to OpenVINO INT8 with {len(calib_bgr)} training frames...", flush=True)
        xml_files = list(fp16_dir.glob("*.xml"))
        ov_model = core.read_model(str(xml_files[0]))
        input_name = ov_model.inputs[0].get_any_name()

        def transform_fn(bgr: np.ndarray):
            h, w = bgr.shape[:2]
            scale = imgsz / max(h, w, 1)
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
            canvas[:nh, :nw] = resized
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return {input_name: np.transpose(rgb, (2, 0, 1))[None, ...]}

        q_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calib_bgr, transform_fn),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(calib_bgr),
        )
        int8_dir = out_root / f"{pt_path.stem}_int8_openvino_model"
        int8_dir.mkdir(parents=True, exist_ok=True)
        for meta in fp16_dir.glob("*.yaml"):
            shutil.copy2(meta, int8_dir / meta.name)
        ov.save_model(q_model, str(int8_dir / xml_files[0].name), compress_to_fp16=True)
        info["int8"] = str(int8_dir.relative_to(ROOT).as_posix())
    return info


def export_segformer_openvino(src_dir: Path, out_root: Path, imgsz: int, calib_bgr: list[np.ndarray], do_int8: bool) -> dict:
    import openvino as ov
    from transformers import AutoModelForSemanticSegmentation

    print(f"[2/3] Exporting SegFormer ({src_dir.name}) to static {imgsz}x{imgsz} OpenVINO FP16...", flush=True)
    hf_model = AutoModelForSemanticSegmentation.from_pretrained(str(src_dir), local_files_only=True).eval()
    wrapper = SegFormerExportWrapper(hf_model).eval()
    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
    with torch.inference_mode():
        ov_model = ov.convert_model(wrapper, example_input=dummy, input=[("pixel_values", [1, 3, imgsz, imgsz])])

    fp16_dir = out_root / "segformer_sidewalk_fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(fp16_dir / "model.xml"), compress_to_fp16=True)
    shutil.copy2(src_dir / "config.json", fp16_dir / "config.json")
    (fp16_dir / "ov_meta.json").write_text(json.dumps({"imgsz": imgsz, "precision": "fp16"}, indent=2), encoding="utf-8")
    info = {"fp16": str(fp16_dir.relative_to(ROOT).as_posix())}

    if do_int8:
        import nncf
        print(f"      Quantizing SegFormer to OpenVINO INT8...", flush=True)
        q_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calib_bgr, lambda bgr: {"pixel_values": letterbox_normalized(bgr, imgsz)[0]}),
            model_type=nncf.ModelType.TRANSFORMER,
            subset_size=min(32, len(calib_bgr)),
        )
        int8_dir = out_root / "segformer_sidewalk_int8"
        int8_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(q_model, str(int8_dir / "model.xml"), compress_to_fp16=True)
        shutil.copy2(src_dir / "config.json", int8_dir / "config.json")
        (int8_dir / "ov_meta.json").write_text(json.dumps({"imgsz": imgsz, "precision": "int8"}, indent=2), encoding="utf-8")
        info["int8"] = str(int8_dir.relative_to(ROOT).as_posix())
    return info


def export_depth_openvino(src_dir: Path, out_root: Path, imgsz: int, calib_bgr: list[np.ndarray], do_int8: bool) -> dict:
    import openvino as ov
    from transformers import AutoModelForDepthEstimation

    print(f"[3/3] Exporting Depth Anything V2 ({src_dir.name}) to static {imgsz}x{imgsz} OpenVINO FP16...", flush=True)
    hf_model = AutoModelForDepthEstimation.from_pretrained(str(src_dir), local_files_only=True).eval()
    wrapper = DepthExportWrapper(hf_model).eval()
    dummy = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
    with torch.inference_mode():
        ov_model = ov.convert_model(wrapper, example_input=dummy, input=[("pixel_values", [1, 3, imgsz, imgsz])])

    fp16_dir = out_root / "depth_v2_small_fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(fp16_dir / "model.xml"), compress_to_fp16=True)
    shutil.copy2(src_dir / "config.json", fp16_dir / "config.json")
    (fp16_dir / "ov_meta.json").write_text(json.dumps({"imgsz": imgsz, "precision": "fp16"}, indent=2), encoding="utf-8")
    info = {"fp16": str(fp16_dir.relative_to(ROOT).as_posix())}

    if do_int8:
        import nncf
        print(f"      Quantizing Depth Anything V2 to OpenVINO INT8...", flush=True)
        q_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calib_bgr, lambda bgr: {"pixel_values": letterbox_normalized(bgr, imgsz)[0]}),
            model_type=nncf.ModelType.TRANSFORMER,
            subset_size=min(32, len(calib_bgr)),
        )
        int8_dir = out_root / "depth_v2_small_int8"
        int8_dir.mkdir(parents=True, exist_ok=True)
        ov.save_model(q_model, str(int8_dir / "model.xml"), compress_to_fp16=True)
        shutil.copy2(src_dir / "config.json", int8_dir / "config.json")
        (int8_dir / "ov_meta.json").write_text(json.dumps({"imgsz": imgsz, "precision": "int8"}, indent=2), encoding="utf-8")
        info["int8"] = str(int8_dir.relative_to(ROOT).as_posix())
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", default="models/results/yolo26n_best.pt")
    parser.add_argument("--semantic", default="models/segformer_sidewalk")
    parser.add_argument("--depth", default="models/depth_anything_v2_small")
    parser.add_argument("--out-dir", default="models/openvino")
    parser.add_argument("--yolo-size", type=int, default=480)
    parser.add_argument("--seg-size", type=int, default=384)
    parser.add_argument("--depth-size", type=int, default=350)
    parser.add_argument("--no-int8", action="store_true")
    args = parser.parse_args()

    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    device_names = {d: core.get_property(d, "FULL_DEVICE_NAME") for d in devices}
    print("OpenVINO devices detected:", json.dumps(device_names, ensure_ascii=False))

    out_root = ROOT / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    calib = collect_calibration_bgr(40)
    do_int8 = not args.no_int8

    t0 = time.perf_counter()
    manifest = {
        "devices_at_export": device_names,
        "detector": export_yolo_openvino(ROOT / args.detector, out_root, args.yolo_size, calib, do_int8),
        "semantic": export_segformer_openvino(ROOT / args.semantic, out_root, args.seg_size, calib, do_int8),
        "depth": export_depth_openvino(ROOT / args.depth, out_root, args.depth_size, calib, do_int8),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }
    manifest_path = out_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Export complete:", json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
