import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from guide_core.evaluate import check_split_leakage, read_run, score_manifest


class EvaluationManifestTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def write_json(self, path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def example(self, number, clear, false_alert, audio=True):
        digest = str(number) * 64
        filename = f"route{number}.mp4"
        annotations = self.root / f"route{number}.json"
        run = self.root / f"run{number}"
        run.mkdir()
        self.write_json(annotations, {
            "schema_version": 1, "reviewed": True, "split": "test",
            "session_id": f"session{number}", "location_id": f"location{number}",
            "video": filename, "video_sha256": digest, "duration_s": clear + 2,
            "intervals": [
                {"start": 0, "end": 2, "kind": "hazard", "allowed_actions": ["STOP"]},
                {"start": 2, "end": clear + 2, "kind": "clear", "allowed_actions": ["KEEP_HEADING"]},
            ],
        })
        self.write_json(run / "report.json", {
            "video_sha256": {filename: digest}, "source_video_duration_s": clear + 2,
            "video_duration_s": clear + 10, "mode": "realtime_replay",
            "audio": {"mode": "scheduled" if audio else "disabled"},
        })
        (run / "decisions.jsonl").write_text('{"time_s":1,"action":"STOP"}\n', encoding="utf-8")
        if audio:
            alerts = [{"time_s": 1, "event": "scheduled", "action": "STOP"}]
            if false_alert:
                alerts.append({"time_s": 3, "event": "scheduled", "action": "STOP"})
            (run / "audio_events.jsonl").write_text(
                "\n".join(map(json.dumps, alerts)), encoding="utf-8")
            self.write_json(run / "speech_segments.json", [])
        return {"annotations": annotations.name, "run": run.name}

    def manifest(self, items):
        path = self.root / "test.json"
        self.write_json(path, {"schema_version": 1, "split": "test", "items": items})
        return path

    def test_rates_use_combined_event_counts_and_clear_duration(self):
        first = self.example(1, 60, True)
        second = self.example(2, 120, False)
        result = score_manifest(self.manifest([first, second]))
        self.assertEqual(result["videos"], 2)
        self.assertEqual(result["overall"]["clear_minutes"], 3)
        self.assertAlmostEqual(result["overall"]["false_alerts_per_clear_minute"], 1 / 3)

    def test_disabled_audio_does_not_claim_zero_false_alerts(self):
        result = score_manifest(self.manifest([self.example(1, 60, False, audio=False)]))
        self.assertIsNone(result["overall"]["false_alerts_per_clear_minute"])
        self.assertFalse(result["overall"]["audio_evaluated"])

    def test_replay_drain_time_does_not_count_as_source_coverage(self):
        item = self.example(1, 60, False)
        annotation = self.root / item["annotations"]
        data = json.loads(annotation.read_text())
        data["duration_s"] += 5
        self.write_json(annotation, data)
        with self.assertRaisesRegex(ValueError, "covers only"):
            read_run(annotation, self.root / item["run"])

    def test_same_name_different_sha_is_rejected(self):
        item = self.example(1, 60, False)
        annotation = self.root / item["annotations"]
        data = json.loads(annotation.read_text())
        data["video_sha256"] = "f" * 64
        self.write_json(annotation, data)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            read_run(annotation, self.root / item["run"])

    def test_different_analysis_settings_cannot_be_pooled(self):
        first = self.example(1, 60, False)
        second = self.example(2, 60, False)
        self.write_json(self.root / first["run"] / "config.json", {"sample_fps": 10})
        self.write_json(self.root / second["run"] / "config.json", {"sample_fps": 5})
        with self.assertRaisesRegex(ValueError, "different weights, analysis modes or settings"):
            score_manifest(self.manifest([first, second]))

    def test_duplicate_video_and_case_variant_location(self):
        item = self.example(1, 60, False)
        with self.assertRaisesRegex(ValueError, "same video"):
            score_manifest(self.manifest([item, item]))
        with self.assertRaisesRegex(ValueError, "location_id"):
            check_split_leakage([
                {"location_id": " alley-01 ", "split": "validation"},
                {"location_id": "Alley-01", "split": "test"},
            ])


if __name__ == "__main__":
    unittest.main()

