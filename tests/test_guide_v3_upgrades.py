from dataclasses import replace
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from test_guide_core import config, scene, frame, decision, Bank
from guide_core.types import Action, Detection, Scene
from guide_core.planning import CorridorPlanner, analyze_depth_geometry, urgent_reason
from guide_core.policy import DecisionPolicy
from guide_core.tracking import MotionTracker
from guide_core.audio import AudioTimeline, pan_stereo, synthesize_earcon
from obstacle_detector import analyze_frame


class GuideV3UpgradeTests(unittest.TestCase):
    def test_ransac_catches_unlabelled_positive_depth_obstacle_and_stops(self):
        s = scene()
        # Smooth unlabelled box/wall directly ahead: YOLO misses it, SegFormer still says sidewalk.
        s.depth["relative_nearness"][155:232, 118:202] = 0.92
        planner = CorridorPlanner(config(depth=True))
        route = planner.plan(s)
        self.assertIsNone(route.action)
        d = DecisionPolicy(config(depth=True)).update(s, route, 0.0)
        self.assertEqual(d.action, Action.STOP)
        self.assertEqual(d.reason, "near_depth_obstacle")

    def test_ransac_catches_negative_dropoff_pit_ahead(self):
        s = scene()
        # Sudden downward pit / missing drain / downward steps right in front of feet.
        s.depth["relative_nearness"][172:232, 115:205] = 0.05
        planner = CorridorPlanner(config(depth=True))
        route = planner.plan(s)
        self.assertIsNone(route.action)
        d = DecisionPolicy(config(depth=True)).update(s, route, 0.0)
        self.assertEqual(d.action, Action.STOP)
        self.assertEqual(d.reason, "near_dropoff_hazard")

    def test_overhanging_head_hazard_fused_with_depth_ignores_distant_trees(self):
        c = config(depth=True)
        # Distant tree canopy in upper center (low nearness = far away) -> must NOT false-alarm STOP.
        far_scene = scene()
        far_scene.detections = [Detection("tree", 0.88, (115, 20, 205, 130), ("y",), zone="head")]
        self.assertFalse(urgent_reason(far_scene, c.ttc_stop_s)[0])

        # Close overhanging branch/barrier at chest/head height (high nearness = close) -> MUST STOP!
        near_scene = scene()
        near_scene.detections = [Detection("tree", 0.88, (115, 20, 205, 130), ("y",), zone="head")]
        near_scene.depth["relative_nearness"][20:130, 115:205] = 0.86
        is_urg, why, hint = urgent_reason(near_scene, c.ttc_stop_s)
        self.assertTrue(is_urg)
        self.assertEqual(why, "overhead_center_obstacle")
        self.assertIn("ngang tầm đầu", hint)

    def test_v1_detector_retains_central_overhanging_hazard(self):
        branch = SimpleNamespace(cls=np.array(0), conf=np.array(0.85), xyxy=np.array([[40, 8, 60, 44]]))

        class FakeModel:
            def predict(self, **kwargs):
                return [SimpleNamespace(names={0: "tree"}, boxes=[branch], masks=None)]

        img = np.full((100, 100, 3), 120, dtype=np.uint8)
        out = analyze_frame(img, FakeModel(), confidence=0.3, render=False)
        self.assertEqual(len(out.obstacles), 1)
        self.assertEqual(out.obstacles[0].distance_band, "gần")

    def test_looming_ttc_stops_for_dead_center_approaching_object(self):
        c = config(depth=False)
        tracker = MotionTracker(c)
        policy = DecisionPolicy(c)
        # Object stays dead-center (cx=160, cy=115) so center shift ~0, but box expands rapidly!
        boxes = [(142, 90, 178, 140), (134, 80, 186, 150), (120, 62, 200, 168)]
        base_img = frame(0, 0).bgr
        last_decision = None
        for idx, box in enumerate(boxes):
            pts = idx * 0.1
            f = frame(idx, pts)
            # Keep background identical so ego-motion warp is identity.
            f = type(f)(f.session, idx, pts, 0.0, base_img.copy(), f.original_hw, False)
            s = Scene(f, [Detection("motorcycle", 0.9, box, ("y",))])
            s = tracker.update(s)
            last_decision = policy.update(s, None, pts, full=False)
        self.assertIsNotNone(s.detections[0].ttc_s)
        self.assertLess(s.detections[0].ttc_s, 1.6)
        self.assertEqual(last_decision.action, Action.STOP)
        self.assertEqual(last_decision.reason, "approaching_obstacle_ttc")

    def test_multi_horizon_allows_mid_range_corridor_and_group_prob_fusion(self):
        c = config(depth=False)
        s = scene()
        s.depth = None
        # Distant horizon row band (top 20% of ROI: y=108..132) is blocked by a distant building/turn,
        # while 80% of the walkway ahead (y=133..230) is completely clear and wide.
        s.semantic["classes"][108:132, :] = 0
        route = CorridorPlanner(c).plan(s)
        self.assertEqual(route.action, Action.KEEP_HEADING)
        self.assertGreaterEqual(route.horizon_ratio, 0.70)
        self.assertEqual(route.clock_hour, 12)

        # Synonym dilution: top-1 confidence is 0.36 (<0.60), but sidewalk+path group_prob is 0.74 (>=0.60).
        s2 = scene()
        s2.depth = None
        s2.semantic["confidence"][:] = 0.36
        s2.semantic["group_probs"] = {
            "sidewalk": np.full(s2.frame.bgr.shape[:2], 0.74, dtype=np.float32),
            "structure": np.zeros(s2.frame.bgr.shape[:2], dtype=np.float32),
        }
        route2 = CorridorPlanner(c).plan(s2)
        self.assertEqual(route2.action, Action.KEEP_HEADING)

    def test_emergency_preemption_earcon_and_stereo_panning(self):
        audio = AudioTimeline(Bank(1.0), repeat_s=0, preempt_emergency=True, earcon=True, stereo=True)
        audio.update(decision(Action.SLOW, 0.0), 0.0)
        self.assertEqual(audio.active["decision"].action, Action.SLOW)
        # Emergency STOP arrives at 0.25s and must preempt SLOW immediately with zero queue wait!
        audio.update(decision(Action.STOP, 0.25), 0.25)
        self.assertEqual(audio.active["decision"].action, Action.STOP)
        self.assertEqual(audio.active["start"], 0.25)
        self.assertEqual(audio.segments[0]["end"], 0.25)
        self.assertTrue(any(e["event"] == "cancelled" and e["reason"] == "preempted_by_emergency_stop" for e in audio.events))
        earcon_len = len(synthesize_earcon(Bank.rate, "stop"))
        self.assertEqual(len(audio.active["samples"]), Bank.rate + earcon_len)

        # Verify stereo directional panning energy balance.
        mono = np.full(200, 10000, dtype="<i2")
        left_stereo = pan_stereo(mono, -0.65)
        right_stereo = pan_stereo(mono, +0.65)
        self.assertGreater(np.abs(left_stereo[:, 0]).mean(), np.abs(left_stereo[:, 1]).mean())
        self.assertGreater(np.abs(right_stereo[:, 1]).mean(), np.abs(right_stereo[:, 0]).mean())

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "stereo.wav"
            end_t = audio.finish_export(1.5)
            audio.save(wav_path, end_t)
            with wave.open(str(wav_path), "rb") as w:
                self.assertEqual(w.getnchannels(), 2)

    def test_heterogeneous_cpu_and_igpu_devices_execute_concurrently(self):
        import time
        from guide_core.config import Config, ModelSpec
        from guide_core.engine import GuideEngine
        from guide_core.models import _letterbox_norm
        from test_guide_core import IdentityTracker

        tensor, (nh, nw) = _letterbox_norm(np.zeros((296, 640, 3), dtype=np.uint8), 384)
        self.assertEqual(tensor.shape, (1, 3, 384, 384))
        self.assertEqual((nh, nw), (178, 384))

        class SlowDeviceAdapter:
            def __init__(self, role, dev):
                self.spec = ModelSpec(role, "yolo" if role == "detector" else "segformer", role, "fake")
                self.device = dev
            def load(self): pass
            def close(self): pass
            def predict(self, f):
                time.sleep(0.08)
                if self.spec.role == "detector":
                    return []
                h, w = f.bgr.shape[:2]
                return {"classes": np.ones((h, w), np.int16), "confidence": np.ones((h, w)), "labels": {1: "sidewalk"}}

        cfg = Config([
            ModelSpec("det", "openvino_yolo", "detector", "fake"),
            ModelSpec("sem", "openvino_segformer", "semantic", "fake"),
        ])
        engine = GuideEngine(cfg, [SlowDeviceAdapter("detector", "openvino:CPU"),
                                   SlowDeviceAdapter("semantic", "openvino:GPU")], IdentityTracker())
        t0 = time.perf_counter()
        results = engine.predict(frame(), {"detector", "semantic"})
        elapsed = time.perf_counter() - t0
        self.assertEqual([r.role for r in results], ["detector", "semantic"])
        # Sequential would take >=0.16s; concurrent CPU + iGPU finishes in ~0.085s.
        self.assertLess(elapsed, 0.145)

    def test_interleaved_fast_detector_frames_preserve_async_route_confirmation(self):
        c = config(depth=False)
        planner = CorridorPlanner(c)
        policy = DecisionPolicy(c)
        # Interleave context frames (full=True) with fast detector-only frames (full=False) as in realtime_replay.
        s0 = scene(0, 0.0); s0.depth = None
        self.assertEqual(policy.update(s0, planner.plan(s0), 0.0, full=True).action, Action.SLOW)
        s1 = scene(1, 0.05); s1.depth = None
        self.assertEqual(policy.update(s1, None, 0.05, full=False).action, Action.SLOW)
        s2 = scene(2, 0.22); s2.depth = None
        self.assertEqual(policy.update(s2, planner.plan(s2), 0.22, full=True).action, Action.SLOW)
        s3 = scene(3, 0.30); s3.depth = None
        self.assertEqual(policy.update(s3, None, 0.30, full=False).action, Action.SLOW)
        s4 = scene(4, 0.45); s4.depth = None
        self.assertEqual(policy.update(s4, planner.plan(s4), 0.45, full=True).action, Action.KEEP_HEADING)

    def test_live_pipeline_with_openvino_core_and_hud_render(self):
        import time
        from pipeline import APP_DIR, GuidePipeline, PipelineConfig, available_core_profiles
        from obstacle_detector import render_analysis

        profiles = available_core_profiles()
        self.assertIn("openvino_igpu", profiles)
        pipe = GuidePipeline(speech=False)
        try:
            cfg = PipelineConfig(
                source=str(APP_DIR / "data/videos/val2.mp4"),
                core_config=str(profiles["openvino_igpu"]["path"]),
                environment="alley",
            )
            pipe.start(cfg)
            deadline = time.perf_counter() + 12.0
            rendered = None
            while time.perf_counter() < deadline:
                fid, packet = pipe.latest_analysis()
                if packet is not None and fid >= 3:
                    rendered = render_analysis(packet["analysis"])
                    break
                time.sleep(0.05)
            self.assertIsNotNone(rendered)
            self.assertEqual(rendered.ndim, 3)
            snap = pipe.snapshot()
            self.assertGreaterEqual(snap["metrics"].get("processed_frames", 0), 3)
        finally:
            pipe.close()

    def test_desktop_ui_defaults_to_openvino_profile_headless(self):
        import tkinter as tk
        from main import ObstacleGuideApp

        try:
            root = tk.Tk()
        except tk.TclError:
            self.skipTest("Tk display unavailable")
        root.withdraw()
        app = ObstacleGuideApp(root)
        try:
            self.assertIn("OpenVINO", app.core_profile.get())
            self.assertEqual(app.environment.get(), "alley")
        finally:
            app.close()

    def test_smart_arbitration_stop_never_cuts_stop_and_enforces_post_stop_hold(self):
        from dataclasses import replace

        audio = AudioTimeline(Bank(1.0), smart_arbitration=True, post_stop_hold_s=0.8)
        # First STOP starts at t=0.0..1.0s
        audio.update(decision(Action.STOP, 0.0), 0.0)
        self.assertEqual(audio.active["start"], 0.0)
        # Second STOP at t=0.3s (e.g. class changed from person to car) MUST NOT cut active STOP!
        audio.update(replace(decision(Action.STOP, 0.3), hazard_hint="ô tô"), 0.3)
        self.assertEqual(audio.active["start"], 0.0)
        self.assertEqual(len(audio.segments), 1)
        # At t=1.0s STOP finishes; KEEP_HEADING arrives at t=1.1s (valid until 2.5s), held until t=1.8s!
        audio.tick(1.0)
        self.assertIsNone(audio.active)
        audio.update(decision(Action.KEEP_HEADING, 1.1, ttl=1.4), 1.1)
        self.assertIsNone(audio.active)
        audio.tick(1.5)
        self.assertIsNone(audio.active)  # Still inside 0.8s post-STOP hold (1.0 + 0.8 = 1.8s)
        audio.tick(1.81)
        self.assertIsNotNone(audio.active)
        self.assertEqual(audio.active["decision"].action, Action.KEEP_HEADING)

    def test_smart_arbitration_replaces_queued_stop_with_latest_equal_priority_turn(self):
        audio = AudioTimeline(Bank(1.0), smart_arbitration=True)
        audio.update(decision(Action.SHIFT_LEFT, 0.0), 0.0)
        audio.update(decision(Action.STOP, 0.3, ttl=1.0), 0.3)
        self.assertEqual(audio.pending.action, Action.STOP)
        audio.update(decision(Action.SHIFT_RIGHT, 0.4, ttl=1.0), 0.4)
        self.assertEqual(audio.pending.action, Action.SHIFT_RIGHT)
        self.assertEqual(audio.active["decision"].action, Action.SHIFT_LEFT)
        audio.tick(1.0)
        self.assertEqual(audio.active["decision"].action, Action.SHIFT_RIGHT)
        self.assertEqual(audio.segments[0]["end"], 1.0)

    def test_smart_arbitration_finishes_turn_and_slow_before_playing_pending_turn(self):
        audio = AudioTimeline(Bank(1.0), smart_arbitration=True, revocation_debounce_frames=2, earcon=True)
        audio.update(decision(Action.SHIFT_LEFT, 0.0), 0.0)
        # Multiple consecutive SLOW / UNCLEAR frames MUST NEVER cut active SHIFT_LEFT ("Lệch nhẹ sang trái")!
        for t in (0.2, 0.3, 0.4, 0.5, 0.6):
            audio.update(decision(Action.SLOW, t), t)
            self.assertEqual(audio.active["decision"].action, Action.SHIFT_LEFT)
            self.assertEqual(audio.active["end"], 1.0)
        # Complete SHIFT_LEFT cleanly at t=1.0s; queued SLOW may then play once speaker is free.
        audio.tick(1.0)
        self.assertEqual(audio.segments[0]["end"], 1.0)
        self.assertEqual(audio.active["decision"].action, Action.SLOW)
        slow_start = audio.active["start"]

        audio.update(decision(Action.SHIFT_RIGHT, slow_start + 0.4), slow_start + 0.4)
        self.assertEqual(audio.active["decision"].action, Action.SLOW)
        self.assertEqual(audio.pending.action, Action.SHIFT_RIGHT)
        audio.tick(audio.active["end"])
        self.assertEqual(audio.active["decision"].action, Action.SHIFT_RIGHT)
        self.assertFalse(any(e["event"] == "cancelled" for e in audio.events))

        # Asymmetric cooldown: UNCLEAR has 8.5s repeat cooldown, while delta direction change has 0s cooldown!
        end_sr = audio.active["end"]
        audio.tick(end_sr + 0.05)
        audio.update(decision(Action.UNCLEAR, end_sr + 0.1), end_sr + 0.1)
        self.assertEqual(audio.active["decision"].action, Action.UNCLEAR)
        end_unclear = audio.active["end"]
        audio.tick(end_unclear + 0.1)
        audio.update(decision(Action.UNCLEAR, end_unclear + 0.2), end_unclear + 0.2)
        self.assertIsNone(audio.active)  # Suppressed by 8.5s UNCLEAR repeat cooldown!
        audio.update(decision(Action.KEEP_HEADING, end_unclear + 0.3), end_unclear + 0.3)
        self.assertEqual(audio.active["decision"].action, Action.KEEP_HEADING)  # New info -> 0s cooldown!
        end_kh = audio.active["end"]
        audio.tick(end_kh + 0.05)
        audio.update(decision(Action.SHIFT_LEFT, end_kh + 0.1), end_kh + 0.1)
        self.assertEqual(audio.active["decision"].action, Action.SHIFT_LEFT)  # KEEP_HEADING -> SHIFT_LEFT: 0s!

    def test_auto_environment_and_smooth_obstacle_bypass_without_false_stop(self):
        c = config(depth=True)
        c.environment = "auto"
        s = scene()
        h, w = s.frame.bgr.shape[:2]
        # Indoor ceramic floor in group_probs -> auto environment resolves to "indoor" and routes cleanly.
        s.semantic["classes"][:] = 3  # "floor"
        s.semantic["group_probs"] = {
            "indoor": np.full((h, w), 0.92, dtype=np.float32),
            "sidewalk": np.zeros((h, w), dtype=np.float32),
            "alley": np.zeros((h, w), dtype=np.float32),
            "structure": np.zeros((h, w), dtype=np.float32),
        }
        planner = CorridorPlanner(c)
        policy = DecisionPolicy(c)
        for n, t in enumerate([0.0, 0.15, 0.30, 0.45]):
            sn = scene(n, t)
            sn.semantic = s.semantic
            d = policy.update(sn, planner.plan(sn), t, full=True)
        self.assertEqual(d.action, Action.KEEP_HEADING)

        # Now place a person on the left-center so route curves smoothly to SHIFT_RIGHT (no false STOP!).
        actions = []
        for idx, t in enumerate([0.60, 0.75, 0.90, 1.05], start=4):
            sn = scene(idx, t)
            sn.semantic = {
                "classes": s.semantic["classes"].copy(),
                "confidence": s.semantic["confidence"].copy(),
                "labels": s.semantic["labels"],
                "group_probs": {k: v.copy() for k, v in s.semantic["group_probs"].items()},
            }
            sn.detections = [Detection("person", 0.85, (10, 110, 155, 180), ("y",))]
            r = planner.plan(sn)
            self.assertEqual(r.action, Action.SHIFT_RIGHT)
            d = policy.update(sn, r, t, full=True)
            actions.append(d.action)
        self.assertNotIn(Action.STOP, actions)
        self.assertEqual(actions[-1], Action.SHIFT_RIGHT)
        self.assertIn("tránh người bên trái", d.hazard_hint)

    def test_post_stop_fast_recovery_for_latest_turn_without_replaying_old_turn(self):
        audio = AudioTimeline(
            Bank(1.0),
            preempt_emergency=True,
            smart_arbitration=True,
            post_stop_hold_s=0.8,
            post_stop_turn_hold_s=0.3,
        )
        # 1. Old SHIFT_LEFT starts at 0.0s and is cut by STOP at 0.25s; if no new turn arrives, old turn NEVER replays!
        audio.update(decision(Action.SHIFT_LEFT, 0.0, ttl=2.0), 0.0)
        audio.update(replace(decision(Action.STOP, 0.25), reason="spoken_route_blocked"), 0.25)
        self.assertEqual(audio.active["decision"].action, Action.STOP)
        self.assertIsNone(audio.pending)
        audio.tick(1.25)  # STOP finishes at 1.25s
        audio.tick(1.60)
        self.assertIsNone(audio.active)

        # 2. Now SHIFT_RIGHT starts at 2.5s, STOP cuts it at 2.8s (plays 2.8..3.8s), and a FRESH escape turn
        # SHIFT_RIGHT arrives at 3.55s (default ttl=0.7 -> valid_until=4.25s).
        # With post_stop_turn_hold_s=0.3s, it deploys automatically at 3.8 + 0.3 = 4.10s (< 4.25s)!
        audio.update(decision(Action.SHIFT_RIGHT, 2.5), 2.5)
        audio.update(replace(decision(Action.STOP, 2.8), reason="spoken_route_blocked"), 2.8)
        audio.update(decision(Action.SHIFT_RIGHT, 3.55, ttl=0.7), 3.55)
        audio.tick(3.80)  # STOP finishes at 3.80s
        audio.tick(4.00)
        self.assertIsNone(audio.active)  # Inside 0.30s breathing gap (3.80..4.10s)
        audio.tick(4.11)
        self.assertIsNotNone(audio.active)
        self.assertEqual(audio.active["decision"].action, Action.SHIFT_RIGHT)


if __name__ == "__main__":
    unittest.main()
