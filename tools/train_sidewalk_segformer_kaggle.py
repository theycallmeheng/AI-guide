"""Train Sidewalk SegFormer with bounded evaluation memory on one Kaggle GPU."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import zipfile

# Must precede the first torch/CUDA import in a fresh notebook kernel.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

DATASET_ID = "segments/sidewalk-semantic"
MODEL_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512"
IGNORE_INDEX = 0
TRANSFORMERS_VERSION = "4.57.1"


def memory_snapshot():
    """Process RAM, actual container limit when exposed, and CUDA allocator usage."""
    import psutil

    gib = 1024 ** 3
    result = {"ram_process_gib": round(psutil.Process().memory_info().rss / gib, 3)}
    for used_file, limit_file in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            limit = int(Path(limit_file).read_text().strip())
            used = int(Path(used_file).read_text().strip())
            if 0 < limit < 2 ** 60:
                result.update(ram_container_gib=round(used / gib, 3),
                              ram_container_limit_gib=round(limit / gib, 3))
                break
        except (OSError, ValueError):
            continue
    if torch.cuda.is_available():
        result.update(vram_allocated_gib=round(torch.cuda.memory_allocated() / gib, 3),
                      vram_reserved_gib=round(torch.cuda.memory_reserved() / gib, 3),
                      vram_peak_gib=round(torch.cuda.max_memory_allocated() / gib, 3))
    return result


@contextmanager
def open_source_image(value):
    """Close decoded files/byte streams immediately after one source sample."""
    stream = None
    if isinstance(value, Image.Image):
        image = value.copy()
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            stream = io.BytesIO(value["bytes"])
            image = Image.open(stream)
        elif value.get("path"):
            image = Image.open(value["path"])
        else:
            raise ValueError("Ảnh Hugging Face thiếu cả bytes và path.")
    else:
        image = Image.fromarray(np.asarray(value))
    try:
        yield image
    finally:
        image.close()
        if stream is not None:
            stream.close()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def prepare_resized_cache(dataset, root, indices, image_size, num_labels, fingerprint):
    """Resize on disk, one source image/mask at a time; never cache decoded tensors."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {"format": 1, "fingerprint": fingerprint, "image_size": image_size,
                "num_labels": num_labels, "ignore_index": IGNORE_INDEX}
    marker = root / "cache_metadata.json"
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Cache khác dữ liệu/cấu hình. Chọn thư mục cache mới.")
    atomic_json(marker, metadata)
    (root / "images").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    indices = sorted(set(int(index) for index in indices))
    for number, index in enumerate(indices, 1):
        image_path = root / "images" / f"{index:06d}.jpg"
        mask_path = root / "labels" / f"{index:06d}.png"
        if not (image_path.is_file() and mask_path.is_file()):
            row = dataset[index]
            with open_source_image(row["pixel_values"]) as source_image:
                with source_image.convert("RGB") as rgb:
                    with rgb.resize((image_size, image_size), Image.Resampling.BILINEAR) as resized:
                        temporary = image_path.with_suffix(".tmp.jpg")
                        resized.save(temporary, format="JPEG", quality=95)
                        temporary.replace(image_path)
            with open_source_image(row["label"]) as source_mask:
                with source_mask.resize((image_size, image_size), Image.Resampling.NEAREST) as resized:
                    mask = np.array(resized, copy=True)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            if mask.ndim != 2 or np.any((mask < 0) | ((mask >= num_labels) & (mask != 255))):
                raise ValueError(f"Mask nguồn {index} có shape/ID không hợp lệ.")
            mask = np.where(mask == 255, IGNORE_INDEX, mask).astype(np.uint8)
            temporary = mask_path.with_suffix(".tmp.png")
            with Image.fromarray(mask) as image:
                image.save(temporary, format="PNG")
            temporary.replace(mask_path)
            del row, mask
        if number % 100 == 0 or number == len(indices):
            gc.collect()
            print(f"Cache {number}/{len(indices)}", memory_snapshot(), flush=True)
    return root


class CachedSidewalkDataset(Dataset):
    def __init__(self, root, indices, processor):
        self.root = Path(root)
        self.indices = list(indices)
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source_index = self.indices[index]
        with Image.open(self.root / "images" / f"{source_index:06d}.jpg") as image:
            image_array = np.array(image, dtype=np.uint8, copy=True)
        with Image.open(self.root / "labels" / f"{source_index:06d}.png") as mask:
            mask_array = np.array(mask, dtype=np.uint8, copy=True)
        encoded = self.processor(images=image_array, segmentation_maps=mask_array, return_tensors="pt")
        return {"pixel_values": encoded["pixel_values"][0].float(),
                "labels": encoded["labels"][0].long()}


def data_collator(batch):
    return {name: torch.stack([sample[name] for sample in batch]) for name in ("pixel_values", "labels")}


@torch.no_grad()
def logits_to_confusion(logits, labels, *, num_labels):
    """Replace each eval batch's logits with one tiny C x C count matrix."""
    if isinstance(logits, tuple):
        logits = logits[0]
    resized = F.interpolate(logits.float(), size=labels.shape[-2:], mode="bilinear", align_corners=False)
    prediction = resized.argmax(dim=1)
    valid = (labels != IGNORE_INDEX) & (labels >= 0) & (labels < num_labels)
    encoded = labels[valid].long() * num_labels + prediction[valid].long()
    return torch.bincount(encoded, minlength=num_labels ** 2).reshape(1, num_labels, num_labels)


class StreamingSegmentationMetrics:
    """Accumulate counts, not per-image predictions or batch-averaged IoUs."""
    def __init__(self, num_labels):
        self.num_labels = num_labels
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_labels, self.num_labels), dtype=np.int64)

    def __call__(self, eval_pred, compute_result=False):
        counts = eval_pred.predictions
        if isinstance(counts, torch.Tensor):
            counts = counts.detach().cpu().numpy()
        counts = np.asarray(counts)
        if counts.shape[-2:] != self.confusion.shape:
            raise ValueError("Metric phải nhận ma trận đếm từ logits_to_confusion.")
        self.confusion += counts.reshape(-1, self.num_labels, self.num_labels).sum(axis=0)
        if not compute_result:
            return {}
        true_positive = np.diag(self.confusion).astype(np.float64)
        target_total = self.confusion.sum(axis=1)
        union = target_total + self.confusion.sum(axis=0) - true_positive
        present = union > 0
        present[IGNORE_INDEX] = False
        present_accuracy = target_total > 0
        present_accuracy[IGNORE_INDEX] = False
        iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=union > 0)
        accuracy = np.divide(true_positive, target_total, out=np.zeros_like(true_positive), where=target_total > 0)
        result = {"mean_iou": float(iou[present].mean()) if present.any() else 0.0,
                  "mean_accuracy": float(accuracy[present_accuracy].mean()) if present_accuracy.any() else 0.0,
                  "pixel_accuracy": float(true_positive.sum() / self.confusion.sum()) if self.confusion.sum() else 0.0}
        self.reset()
        return result


def memory_callback(log_path):
    from transformers import TrainerCallback

    class MemoryLog(TrainerCallback):
        def record(self, state, event):
            sample = {"step": state.global_step, "epoch": state.epoch, "event": event,
                      "time": time.time(), **memory_snapshot()}
            with Path(log_path).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, allow_nan=False) + "\n")
            print("MEMORY", json.dumps(sample), flush=True)

        def on_log(self, args, state, control, **kwargs):
            self.record(state, "log")

        def on_save(self, args, state, control, **kwargs):
            self.record(state, "checkpoint_saved")

    return MemoryLog()


def make_portable_zip(root):
    root = Path(root)
    destination = root.with_suffix(".zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(root.rglob("*")):
            if source.is_file():
                archive.write(source, arcname=source.relative_to(root).as_posix())
    return destination


def run_training(mode="smoke", *, output_root="/kaggle/working", image_size=384, epochs=10,
                 hf_token=None, resume_checkpoint=None):
    """smoke checks train/eval/save; full runs the original 800/100/100 split."""
    from functools import partial
    import transformers
    from datasets import Dataset as HFDataset, Image as HFImage, load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import (SegformerConfig, SegformerForSemanticSegmentation,
                              SegformerImageProcessor, Trainer, TrainingArguments, set_seed)
    from transformers.trainer_utils import get_last_checkpoint

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Cần transformers=={TRANSFORMERS_VERSION}. Chạy ô cài thư viện rồi restart kernel.")
    if mode not in {"smoke", "full"} or image_size not in {320, 384, 512} or epochs < 1:
        raise ValueError("mode phải là smoke/full; ảnh 320/384/512; epochs >= 1.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Cần đúng một GPU hiển thị. Restart kernel và chạy ô CUDA_VISIBLE_DEVICES=0 đầu tiên.")
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            from kaggle_secrets import UserSecretsClient
            hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Thiếu Kaggle Secret HF_TOKEN có quyền đọc dataset.")

    set_seed(42)
    output_root = Path(output_root)
    checkpoint_dir = output_root / f"segformer_sidewalk_{mode}_checkpoints"
    final_dir = output_root / ("segformer_sidewalk_best" if mode == "full" else "segformer_sidewalk_smoke_best")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing = get_last_checkpoint(str(checkpoint_dir))
    if existing and not resume_checkpoint:
        raise RuntimeError(f"Đã có checkpoint: {existing}. Đặt RESUME_CHECKPOINT=True để tiếp tục, hoặc dùng output_root mới.")
    if resume_checkpoint is True:
        resume_checkpoint = existing
        if not resume_checkpoint:
            raise RuntimeError("Không tìm thấy checkpoint để tiếp tục.")
    if resume_checkpoint:
        resume_checkpoint = str(resume_checkpoint)
        needed = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "config.json", "model.safetensors")
        missing = [name for name in needed if not (Path(resume_checkpoint) / name).is_file()]
        if missing:
            raise RuntimeError(f"Checkpoint chưa đầy đủ, thiếu: {missing}. Không thể resume chính xác.")

    print("MODE:", mode, "| GPU:", torch.cuda.get_device_name(0), "| ảnh:", image_size, flush=True)
    print("Khởi tạo:", memory_snapshot(), flush=True)
    api = HfApi(token=hf_token)
    prior = None
    if resume_checkpoint:
        prior_config = Path(resume_checkpoint).parent / "run_config.json"
        if not prior_config.is_file():
            raise RuntimeError("Thiếu run_config.json cạnh checkpoint. Tải cả thư mục output của bản notebook mới.")
        prior = json.loads(prior_config.read_text(encoding="utf-8"))
    dataset_revision = prior["dataset_revision"] if prior else api.dataset_info(DATASET_ID).sha
    model_revision = prior["model_revision"] if prior else api.model_info(MODEL_CHECKPOINT).sha
    raw = load_dataset(DATASET_ID, split="train", token=hf_token, revision=dataset_revision, keep_in_memory=False)
    label_path = hf_hub_download(DATASET_ID, "id2label.json", repo_type="dataset",
                                 revision=dataset_revision, token=hf_token)
    id2label = {int(key): value for key, value in json.loads(Path(label_path).read_text()).items()}
    num_labels = len(id2label)
    if sorted(id2label) != list(range(num_labels)) or num_labels > 255 or id2label.get(0) != "unlabeled":
        raise ValueError("Taxonomy nguồn đã thay đổi; cần kiểm tra trước khi train.")

    # Match the user's original two-stage HF split, retaining original row IDs.
    indexed = HFDataset.from_dict({"source_index": list(range(len(raw)))})
    first = indexed.shuffle(seed=42).train_test_split(test_size=.20, seed=42)
    second = first["test"].train_test_split(test_size=.50, seed=42)
    split_indices = {"train": list(first["train"]["source_index"]),
                     "validation": list(second["train"]["source_index"]),
                     "test": list(second["test"]["source_index"])}
    if mode == "smoke":
        split_indices = {name: indices[:64 if name == "train" else 16] for name, indices in split_indices.items()}
    split_digest = hashlib.sha256(json.dumps(split_indices, sort_keys=True).encode()).hexdigest()
    settings = {"mode": mode, "image_size": image_size, "epochs": epochs,
                "train_batch_size": 1, "eval_batch_size": 1, "gradient_accumulation_steps": 4,
                "dataset": DATASET_ID, "dataset_revision": dataset_revision,
                "model": MODEL_CHECKPOINT, "model_revision": model_revision,
                "split_digest": split_digest, "seed": 42, "ignore_index": IGNORE_INDEX,
                "split_method": "random_image_baseline_not_independent_routes",
                "transformers": transformers.__version__, "torch": torch.__version__}
    config_path = checkpoint_dir / "run_config.json"
    if prior:
        for field in ("mode", "image_size", "epochs", "dataset_revision", "model_revision", "split_digest",
                      "train_batch_size", "gradient_accumulation_steps", "transformers"):
            if prior.get(field) != settings[field]:
                raise RuntimeError(f"Cấu hình resume khác ở {field}; giữ cấu hình của lần chạy trước.")
    atomic_json(config_path, settings)
    atomic_json(checkpoint_dir / "split_indices.json", split_indices)

    fingerprint = f"{dataset_revision}_{raw._fingerprint}"
    raw = raw.cast_column("pixel_values", HFImage(decode=False)).cast_column("label", HFImage(decode=False))
    cache_dir = output_root / f"sidewalk_cache_{image_size}_{dataset_revision[:8]}"
    used_indices = [index for values in split_indices.values() for index in values]
    prepare_resized_cache(raw, cache_dir, used_indices, image_size, num_labels, fingerprint)
    del raw, indexed, first, second
    gc.collect()

    processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT, revision=model_revision,
                    token=hf_token, do_reduce_labels=False, size={"height": image_size, "width": image_size})
    datasets = {name: CachedSidewalkDataset(cache_dir, indices, processor) for name, indices in split_indices.items()}
    config = SegformerConfig.from_pretrained(MODEL_CHECKPOINT, revision=model_revision, token=hf_token)
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = {value: key for key, value in id2label.items()}
    config.semantic_loss_ignore_index = IGNORE_INDEX
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_CHECKPOINT, revision=model_revision,
                          token=hf_token, config=config, ignore_mismatched_sizes=True)
    processor.save_pretrained(checkpoint_dir)
    metrics = StreamingSegmentationMetrics(num_labels)
    max_steps = 20 if mode == "smoke" else -1
    total_steps = max_steps if max_steps > 0 else math.ceil(len(datasets["train"]) / 4) * epochs
    interval = 10 if mode == "smoke" else 100
    args = TrainingArguments(
        output_dir=str(checkpoint_dir), learning_rate=6e-5, num_train_epochs=epochs, max_steps=max_steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=4,
        weight_decay=.01, warmup_steps=max(1, math.ceil(total_steps * .05)),
        eval_strategy="steps", eval_steps=interval, save_strategy="steps", save_steps=interval,
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="mean_iou", greater_is_better=True,
        logging_steps=5 if mode == "smoke" else 25, logging_first_step=True,
        batch_eval_metrics=True, eval_do_concat_batches=False, eval_accumulation_steps=1,
        dataloader_num_workers=0, dataloader_pin_memory=False, remove_unused_columns=False,
        fp16=True, report_to="none", seed=42, optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
                      data_collator=data_collator, processing_class=processor, compute_metrics=metrics,
                      preprocess_logits_for_metrics=partial(logits_to_confusion, num_labels=num_labels),
                      callbacks=[memory_callback(checkpoint_dir / "memory_log.jsonl")])
    print("Số ảnh:", {name: len(value) for name, value in datasets.items()}, flush=True)
    print("Bắt đầu train; validation/checkpoint ở mỗi", interval, "optimizer steps.", flush=True)
    try:
        result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_state()
        trainer.save_model(str(final_dir))
        processor.save_pretrained(final_dir)
        atomic_json(final_dir / "id2label.json", id2label)
        atomic_json(final_dir / "run_config.json", settings)
        test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
        summary = {"status": "SMOKE_COMPLETED" if mode == "smoke" else "TRAINING_COMPLETED",
                   "best_checkpoint": trainer.state.best_model_checkpoint,
                   "best_validation_mean_iou": trainer.state.best_metric,
                   "train": result.metrics, "test": test_metrics, "memory": memory_snapshot()}
        atomic_json(final_dir / "training_summary.json", summary)
        destination = make_portable_zip(final_dir)
        atomic_json(checkpoint_dir / "COMPLETED.json", summary)
        (checkpoint_dir / "FAILED.json").unlink(missing_ok=True)
        print(summary["status"], flush=True)
        print("Mô hình:", final_dir, "| ZIP:", destination, flush=True)
        return summary
    except Exception as exc:
        atomic_json(checkpoint_dir / "FAILED.json", {"type": type(exc).__name__, "memory": memory_snapshot()})
        print("Phiên lỗi; giữ checkpoint đã lưu và memory_log.jsonl để kiểm tra.", flush=True)
        raise
    finally:
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-root", default="/kaggle/working")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", nargs="?", const=True)
    options = parser.parse_args()
    run_training(options.mode, output_root=options.output_root, image_size=options.image_size,
                 epochs=options.epochs, resume_checkpoint=options.resume)
