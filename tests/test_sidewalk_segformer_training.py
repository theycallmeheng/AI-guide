"""Focused offline checks for bounded segmentation metrics and cached labels."""
import importlib.util
import importlib.metadata
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import zipfile

import numpy as np
from PIL import Image
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "train_sidewalk_segformer_kaggle.py"
spec = importlib.util.spec_from_file_location("sidewalk_training", SCRIPT)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def reference_confusion(prediction, target, classes):
    result = np.zeros((classes, classes), dtype=np.int64)
    for expected, predicted in zip(target.reshape(-1), prediction.reshape(-1)):
        if 0 < expected < classes:
            result[expected, predicted] += 1
    return result


class SegmentationMetricTests(unittest.TestCase):
    def test_gpu_preprocessor_matches_pixel_reference_and_ignores_void(self):
        prediction = torch.tensor([[[0, 2, 1], [2, 0, 1]]])
        target = torch.tensor([[[0, 1, 1], [2, 2, 255]]])
        logits = torch.full((1, 3, 2, 3), -10.0)
        logits.scatter_(1, prediction.unsqueeze(1), 10.0)
        counts = training.logits_to_confusion((logits,), target, num_labels=3)
        expected = reference_confusion(prediction.numpy(), target.numpy(), 3)
        np.testing.assert_array_equal(counts.numpy()[0], expected)
        self.assertEqual(counts.shape, (1, 3, 3))
        self.assertEqual(counts.sum().item(), 4)

    def test_streaming_uses_global_counts_and_resets_between_evaluations(self):
        metric = training.StreamingSegmentationMetrics(3)
        first = np.array([[[0, 0, 0], [0, 90, 10], [0, 0, 0]]])
        second = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 1, 9]]])
        self.assertEqual(metric(SimpleNamespace(predictions=first)), {})
        result = metric(SimpleNamespace(predictions=second), compute_result=True)
        self.assertAlmostEqual(result["mean_iou"], (90 / 101 + 9 / 20) / 2)
        self.assertAlmostEqual(result["mean_accuracy"], .9)
        self.assertAlmostEqual(result["pixel_accuracy"], 99 / 110)
        self.assertFalse(metric.confusion.any())
        perfect = torch.tensor([[[0, 0, 0], [0, 5, 0], [0, 0, 0]]])
        self.assertEqual(metric(SimpleNamespace(predictions=perfect), compute_result=True)["mean_iou"], 1.0)

    def test_upsampling_happens_before_argmax(self):
        logits = torch.tensor([[[[0.0]], [[2.0]], [[1.0]]]])
        labels = torch.ones((1, 4, 6), dtype=torch.long)
        counts = training.logits_to_confusion(logits, labels, num_labels=3)
        self.assertEqual(counts[0, 1, 1].item(), 24)

    def test_accumulator_storage_is_constant_across_many_batches(self):
        metric = training.StreamingSegmentationMetrics(35)
        counts = np.zeros((1, 35, 35), dtype=np.int64)
        counts[0, 1, 1] = 100
        for _ in range(200):
            metric(SimpleNamespace(predictions=counts))
        self.assertEqual(metric.confusion.nbytes, 35 * 35 * 8)
        self.assertEqual(set(vars(metric)), {"num_labels", "confusion"})
        result = metric(SimpleNamespace(predictions=np.zeros_like(counts)), compute_result=True)
        self.assertEqual(result["pixel_accuracy"], 1.0)


class CacheAndExportTests(unittest.TestCase):
    def test_cache_keeps_class_ids_and_maps_255_to_ignore(self):
        mask = np.array([[0, 1], [2, 255]], dtype=np.uint8)
        stream = io.BytesIO()
        Image.fromarray(mask).save(stream, format="PNG")
        rows = [{"pixel_values": np.full((2, 2, 3), 127, dtype=np.uint8),
                 "label": {"bytes": stream.getvalue(), "path": None}}]
        with TemporaryDirectory() as temporary:
            root = training.prepare_resized_cache(rows, temporary, [0], 4, 3, "test-fingerprint")
            with Image.open(root / "labels" / "000000.png") as image:
                resized = np.array(image)
            np.testing.assert_array_equal(resized, np.array([[0, 0, 1, 1], [0, 0, 1, 1],
                                                            [2, 2, 0, 0], [2, 2, 0, 0]]))
            with self.assertRaisesRegex(ValueError, "Cache"):
                training.prepare_resized_cache(rows, temporary, [0], 8, 3, "test-fingerprint")

    def test_zip_uses_forward_slashes_for_kaggle(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            training.atomic_json(root / "nested" / "config.json", {"id": "test"})
            archive = training.make_portable_zip(root)
            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.namelist(), ["nested/config.json"])
                self.assertEqual(json.loads(handle.read("nested/config.json")), {"id": "test"})


def stable_trainer_available():
    try:
        return importlib.metadata.version("transformers") == training.TRANSFORMERS_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


@unittest.skipUnless(stable_trainer_available(), "Optional integration check requires the notebook's pinned Transformers")
class TrainerIntegrationTests(unittest.TestCase):
    def test_train_evaluate_save_reload_and_export_without_remote_downloads(self):
        from functools import partial
        from transformers import (SegformerConfig, SegformerForSemanticSegmentation,
                                  SegformerImageProcessor, Trainer, TrainingArguments)

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with TemporaryDirectory() as temporary:
                root = Path(temporary)
                mask = np.ones((32, 32), dtype=np.uint8)
                mask[0] = 0
                mask[16:] = 2
                rows = [{"pixel_values": np.full((32, 32, 3), 50 + 20 * i, dtype=np.uint8),
                         "label": mask} for i in range(4)]
                cache = training.prepare_resized_cache(rows, root / "cache", range(4), 32, 3, "synthetic")
                processor = SegformerImageProcessor(size={"height": 32, "width": 32}, do_reduce_labels=False)
                dataset = training.CachedSidewalkDataset(cache, range(4), processor)
                config = SegformerConfig(num_labels=3, depths=[1, 1, 1, 1],
                                         hidden_sizes=[8, 16, 32, 64], decoder_hidden_size=16,
                                         num_attention_heads=[1, 2, 4, 8], sr_ratios=[8, 4, 2, 1],
                                         semantic_loss_ignore_index=0, drop_path_rate=0)
                model = SegformerForSemanticSegmentation(config)
                metric = training.StreamingSegmentationMetrics(3)
                args = TrainingArguments(output_dir=str(root / "checkpoints"), use_cpu=True,
                                         max_steps=2, per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                         eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
                                         save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="mean_iou",
                                         batch_eval_metrics=True, eval_do_concat_batches=False, eval_accumulation_steps=1,
                                         remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False,
                                         report_to="none", disable_tqdm=True, logging_strategy="no")
                trainer = Trainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset,
                                  data_collator=training.data_collator, compute_metrics=metric, processing_class=processor,
                                  preprocess_logits_for_metrics=partial(training.logits_to_confusion, num_labels=3),
                                  callbacks=[training.memory_callback(root / "memory.jsonl")])
                trainer.train()
                self.assertEqual(trainer.state.global_step, 2)
                checkpoint = Path(trainer.state.best_model_checkpoint)
                for name in ("model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
                    self.assertTrue((checkpoint / name).is_file(), name)
                result = trainer.evaluate()
                self.assertTrue(0 <= result["eval_mean_iou"] <= 1)
                self.assertFalse(metric.confusion.any())
                trainer.save_model(str(root / "best"))
                reloaded = SegformerForSemanticSegmentation.from_pretrained(root / "best")
                self.assertEqual(reloaded.config.num_labels, 3)
                archive = training.make_portable_zip(root / "best")
                with zipfile.ZipFile(archive) as handle:
                    self.assertIn("model.safetensors", handle.namelist())
                    self.assertIn("preprocessor_config.json", handle.namelist())
                records = [json.loads(line) for line in (root / "memory.jsonl").read_text().splitlines()]
                self.assertTrue(any(record["event"] == "checkpoint_saved" for record in records))
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
