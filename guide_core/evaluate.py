"""Score event annotations against a single run or a collection of recorded runs."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from .types import Action, DIRECTIONAL

ACTIONS = {action.value for action in Action}
DIRECTIONS = {action.value for action in DIRECTIONAL}
WARNINGS = ACTIONS - DIRECTIONS


def read_json(path):
    return Path(path).read_text(encoding="utf-8-sig")


def load_jsonl(path):
    return [json.loads(line) for line in read_json(path).splitlines() if line.strip()]


def validate_annotations(data):
    if data.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    if data.get("reviewed") is not True:
        raise ValueError("Annotations must be reviewed before scoring")
    for field in ("session_id", "location_id"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            raise ValueError("Missing " + field)
    if data.get("split") not in {"validation", "test"}:
        raise ValueError("Use validation/test, not training footage")
    duration = data["duration_s"]
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Expected positive finite video duration")
    digest = data.get("video_sha256")
    if digest is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise ValueError("Invalid video_sha256")
    intervals = data["intervals"]
    if not intervals:
        raise ValueError("No labelled intervals")
    for event in intervals:
        if not 0 <= event["start"] < event["end"] <= duration:
            raise ValueError("Invalid interval")
        kind = event["kind"]
        if kind not in {"hazard", "clear", "guidance", "unclear"}:
            raise ValueError("Unknown event kind")
        allowed = set(event["allowed_actions"])
        forbidden = set(event.get("forbidden_actions", []))
        if not allowed or not (allowed | forbidden) <= ACTIONS or allowed & forbidden:
            raise ValueError("Explicit disjoint allowed/forbidden actions required")
        if kind == "hazard" and not allowed & {"STOP", "SLOW"}:
            raise ValueError("Hazard event must define acceptable warning STOP/SLOW")
        if kind == "guidance" and not allowed & DIRECTIONS:
            raise ValueError("Guidance opportunity must permit at least one direction")
        if kind == "unclear" and "UNCLEAR" not in allowed:
            raise ValueError("Unclear event must allow UNCLEAR")
    for i, first in enumerate(intervals):
        for second in intervals[i + 1:]:
            if min(first["end"], second["end"]) <= max(first["start"], second["start"]):
                continue
            kinds = {first["kind"], second["kind"]}
            conflict = (
                set(first["allowed_actions"]) & set(second.get("forbidden_actions", []))
                or set(second["allowed_actions"]) & set(first.get("forbidden_actions", []))
            )
            if len(kinds) == 1 or kinds == {"clear", "hazard"} or conflict:
                raise ValueError("Duplicate/contradictory event intervals")
    return intervals


def check_split_leakage(annotations):
    seen = {}
    for data in annotations:
        for field in ("location_id", "session_id", "video_sha256"):
            value = data.get(field)
            if not value:
                continue
            key = (field, value.strip().casefold())
            if key in seen and seen[key] != data["split"]:
                raise ValueError(f"Validation/test leakage: {field}={value}")
            seen[key] = data["split"]


def summarize(counts, audio_verified, speech_verified, run_errors=None):
    hazards = counts["hazard_events"]
    minutes = counts["clear_seconds"] / 60
    opportunities = counts["guidance_opportunities"]
    unclear = counts["unclear_events"]
    enough = hazards >= 100 and minutes >= 30
    recall = counts["alerted_hazard_events"] / hazards if hazards and audio_verified else None
    false_rate = counts["false_alerts"] / minutes if minutes and audio_verified else None
    coverage = counts["correct_guidance_events"] / opportunities if opportunities else None
    return {
        **counts,
        "clear_minutes": minutes,
        "decision_event_recall": counts["detected_hazard_events"] / hazards if hazards else None,
        "scheduled_or_submitted_alert_event_recall": recall,
        "false_alerts_per_clear_minute": false_rate,
        "correct_direction_opportunity_coverage": coverage,
        "spoken_direction_opportunity_coverage": (
            counts["spoken_guidance_events"] / opportunities if opportunities and audio_verified else None
        ),
        "unclear_event_recall": counts["correct_unclear_events"] / unclear if unclear else None,
        "audio_evaluated": audio_verified,
        "speech_segments_evaluated": speech_verified,
        "run_errors": run_errors or [],
        "preliminary": not enough,
        "accuracy_targets_met": bool(
            enough and not run_errors and speech_verified
            and recall is not None and recall >= .9 and false_rate is not None and false_rate <= .2
            and not counts["unsafe_direction_decision_count"] and not counts["unsafe_speech_segment_count"]
            and coverage and counts["spoken_guidance_events"]
        ),
        "realtime_latency_target_verified": False,
        "note": "Exported speech is a schedule. Hardware submission latency and physical safety are unverified.",
    }


def score(data, decisions, audio_events=None, speech_segments=None):
    intervals = validate_annotations(data)
    if not decisions:
        raise ValueError("No decisions to evaluate")
    hazards = [event for event in intervals if event["kind"] == "hazard"]
    clear = [event for event in intervals if event["kind"] == "clear"]
    guidance = [event for event in intervals if event["kind"] == "guidance"]
    unclear = [event for event in intervals if event["kind"] == "unclear"]

    def inside(record, event):
        timestamp = record["emitted_at_s"] if "emitted_at_s" in record else record["time_s"]
        return event["start"] <= timestamp < event["end"]

    def count_matches(events, records, eligible=ACTIONS):
        return sum(any(inside(record, event) and record["action"] in eligible
                       and record["action"] in event["allowed_actions"] for record in records)
                   for event in events)

    issued = [record for record in (audio_events or [])
              if record["event"] in {"scheduled", "submitted_to_device"}]
    hazards_alerted = [
        sum(inside(record, event) and record["action"] in event["allowed_actions"]
            and record["action"] in {"STOP", "SLOW"} for record in issued)
        for event in hazards
    ]
    counts = {
        "hazard_events": len(hazards),
        "clear_seconds": sum(event["end"] - event["start"] for event in clear),
        "detected_hazard_events": count_matches(hazards, decisions, {"STOP", "SLOW"}),
        "alerted_hazard_events": sum(count > 0 for count in hazards_alerted),
        "false_alerts": sum(
            record["action"] in WARNINGS and any(inside(record, event) and
            record["action"] not in event["allowed_actions"] for event in clear) for record in issued
        ),
        "unsafe_direction_decision_count": sum(
            record["action"] in DIRECTIONS and any(inside(record, event) and
            record["action"] in event.get("forbidden_actions", []) for event in intervals)
            for record in decisions
        ),
        "unsafe_speech_segment_count": sum(
            segment["action"] in DIRECTIONS and any(
                min(segment["end"], event["end"]) > max(segment["start"], event["start"])
                and segment["action"] in event.get("forbidden_actions", []) for event in intervals
            ) for segment in (speech_segments or [])
        ),
        "guidance_opportunities": len(guidance),
        "correct_guidance_events": count_matches(guidance, decisions, DIRECTIONS),
        "spoken_guidance_events": count_matches(guidance, issued, DIRECTIONS),
        "unclear_events": len(unclear),
        "correct_unclear_events": count_matches(unclear, decisions, {"UNCLEAR"}),
        "duplicate_alerts_within_hazards": sum(max(0, count - 1) for count in hazards_alerted),
    }
    return summarize(counts, audio_events is not None, speech_segments is not None)


def read_run(annotations, root):
    data = json.loads(read_json(annotations))
    validate_annotations(data)
    root = Path(root)
    report = json.loads(read_json(root / "report.json"))
    digests = report.get("video_sha256", {})
    if len(digests) != 1:
        raise ValueError("Expected exactly one source video fingerprint in report.json")
    name, digest = next(iter(digests.items()))
    expected = data.get("video_sha256")
    if expected:
        if expected.casefold() != digest.casefold():
            raise ValueError(f"Annotation/video SHA-256 mismatch: {annotations}")
    elif PurePosixPath(str(data.get("video", "")).replace("\\", "/")).name != name:
        raise ValueError(f"Annotation names a different video: {annotations}")

    duration = report.get("source_video_duration_s")
    if duration is None and report.get("mode") == "offline":
        duration = report.get("video_duration_s")
    if duration is None:
        raise ValueError("Source video duration is missing. Replay the video with the updated demo.")
    if data["duration_s"] > duration + .05 + 1e-9:
        raise ValueError(f"Run covers only {duration:.2f}s. Process the full labelled video: {root}")

    decisions = load_jsonl(root / "decisions.jsonl")
    audio = segments = None
    if report.get("audio", {}).get("mode") != "disabled":
        audio = load_jsonl(root / "audio_events.jsonl")
        segments = json.loads(read_json(root / "speech_segments.json"))
    metrics = score(data, decisions, audio, segments)
    errors = sorted({
        *(f"{model}: {failure}" for model, failure in report.get("load_failures", {}).items()),
        *(f"{model['id']}: {model['error']}" for record in decisions
          for model in record.get("models", []) if model.get("error") and not model.get("skipped")),
    })
    if errors:
        metrics.update(run_errors=errors, accuracy_targets_met=False)
    return data, metrics, digest.casefold(), report


def score_manifest(path):
    path = Path(path)
    manifest = json.loads(read_json(path))
    split = manifest.get("split")
    if manifest.get("schema_version") != 1 or split not in {"validation", "test"} or not manifest.get("items"):
        raise ValueError("Expected a version 1 manifest with a split and one or more items")
    entries = []
    seen_videos = set()
    reference = None
    counts = Counter()
    for item in manifest["items"]:
        root = path.parent / item["run"]
        data, result, digest, report = read_run(path.parent / item["annotations"], root)
        if data["split"] != split:
            raise ValueError("All entries must belong to the manifest split")
        if digest in seen_videos:
            raise ValueError("The same video appears more than once in the manifest")
        config_path = root / "config.json"
        config = json.loads(read_json(config_path)) if config_path.is_file() else {}
        per_video_keys = {"environment", "mirror_input", "audio_dir", "models"}
        shared_config = {key: value for key, value in config.items() if key not in per_video_keys}
        models = sorted(({key: value for key, value in model.items() if key not in {"path", "device"}}
                         for model in config.get("models", []) if model.get("enabled", True)),
                        key=lambda model: model["id"])
        signature = (report.get("mode"), report.get("model_sha256"), shared_config, models)
        if reference is not None and signature != reference:
            raise ValueError(f"Runs use different weights, analysis modes or settings: {root}")
        reference = signature
        seen_videos.add(digest)
        counts.update({key: value for key, value in result.items() if key in COUNT_KEYS})
        entries.append({**item, "video": data.get("video"), "video_sha256": digest,
                        "session_id": data["session_id"], "location_id": data["location_id"],
                        "metrics": result})
    errors = [f"{entry['run']}: {error}" for entry in entries for error in entry["metrics"]["run_errors"]]
    return {
        "split": split,
        "videos": len(entries),
        "sessions": len({entry["session_id"].strip().casefold() for entry in entries}),
        "locations": len({entry["location_id"].strip().casefold() for entry in entries}),
        "overall": summarize(
            dict(counts), all(entry["metrics"]["audio_evaluated"] for entry in entries),
            all(entry["metrics"]["speech_segments_evaluated"] for entry in entries), errors,
        ),
        "items": entries,
    }


COUNT_KEYS = {
    "hazard_events", "clear_seconds", "detected_hazard_events", "alerted_hazard_events",
    "false_alerts", "unsafe_direction_decision_count", "unsafe_speech_segment_count",
    "guidance_opportunities", "correct_guidance_events", "spoken_guidance_events",
    "unclear_events", "correct_unclear_events", "duplicate_alerts_within_hazards",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations")
    source.add_argument("--manifest")
    parser.add_argument("--run")
    parser.add_argument("--output")
    args = parser.parse_args()
    if bool(args.annotations) != bool(args.run):
        parser.error("--annotations and --run must be used together")
    if args.manifest:
        result = score_manifest(args.manifest)
        default = Path(args.manifest).with_suffix(".evaluation.json")
    else:
        _, result, _, _ = read_run(args.annotations, args.run)
        default = Path(args.run) / "evaluation.json"
    output = Path(args.output) if args.output else default
    rendered = json.dumps(result, indent=2)
    output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

