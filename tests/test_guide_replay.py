import unittest
import tempfile
import time
from pathlib import Path
from dataclasses import replace
import cv2
import numpy as np
from test_guide_core import config,scene,frame,decision,Bank,IdentityTracker
from guide_core.types import Detection,Action,Decision,SPEECH_PRIORITY
from guide_core.config import ModelSpec,Config
from guide_core.planning import CorridorPlanner
from guide_core.policy import DecisionPolicy
from guide_core.tracking import MotionTracker
from guide_core.audio import AudioTimeline
from guide_core.engine import GuideEngine
from guide_core.replay import run_replay
from guide_core.evaluate import score

class ReplayTests(unittest.TestCase):
    def test_observed_track_box_is_not_kalman_prediction(self):
        tracker=MotionTracker(config())
        s=scene();s.detections=[Detection("person",.9,(20,30,70,180),("y",))]
        a=tracker.update(s)
        self.assertEqual(a.detections[0].box,(20,30,70,180))
        s=scene(1,.1);s.detections=[Detection("person",.9,(25,30,75,180),("y",))]
        b=tracker.update(s)
        self.assertEqual(b.detections[0].box,(25,30,75,180))
        self.assertIsNotNone(b.detections[0].track_id)
        s=scene(2,.2);s.detections=[]
        self.assertEqual(tracker.update(s).detections,[])

    def test_camera_gap_invalidates_motion(self):
        tracker=MotionTracker(config())
        tracker.update(scene())
        s=tracker.update(scene(20,2))
        self.assertTrue(s.motion_invalid)

    def test_left_right_symmetric_geometry(self):
        c=config(False)
        s=scene();s.depth=None
        ids=s.semantic["classes"];ids[:]=0
        h,w=ids.shape
        y0,y1=int(h*.45),int(h*.96)
        for y in range(y0,h):
            x=int(160-100*(1-np.clip((y-y0)/(y1-y0),0,1)))
            ids[y,max(0,x-46):min(w,x+46)]=1
        left=CorridorPlanner(c).plan(s)
        s.semantic["classes"]=np.fliplr(ids).copy()
        right=CorridorPlanner(c).plan(s)
        self.assertEqual(left.action,Action.SHIFT_LEFT)
        self.assertEqual(right.action,Action.SHIFT_RIGHT)

    def test_stationary_processing_delay_does_not_confirm_direction(self):
        p=DecisionPolicy(config());planner=CorridorPlanner(config())
        for i in range(3):
            s=scene(i,i*.05)
            d=p.update(s,planner.plan(s),i*2.) # long compute time, only 0.1s evidence
        self.assertEqual(d.action,Action.SLOW)

    def test_pending_direction_disappears_before_play(self):
        a=AudioTimeline(Bank(1),repeat_s=0)
        a.update(decision(Action.STOP,0,3),0)
        a.update(decision(Action.SHIFT_LEFT,.2),.2)
        a.update(decision(Action.UNCLEAR,.8),.8)
        a.tick(1.)
        self.assertEqual(a.active["decision"].action,Action.UNCLEAR)
        self.assertFalse(any(s["decision"].action==Action.SHIFT_LEFT for s in a.segments))

    def test_fresh_priority_waits_for_routine_and_stop_is_not_displaced(self):
        audio = AudioTimeline(Bank(1), repeat_s=0)
        audio.update(decision(Action.SLOW, 0), 0)
        audio.update(decision(Action.SHIFT_LEFT, .2), .2)
        self.assertEqual(audio.active["decision"].action, Action.SLOW)
        audio.update(replace(decision(Action.STOP, .4), invalidates_route=True), .4)
        audio.update(decision(Action.UNCLEAR, .9), .9)
        audio.tick(1)
        self.assertEqual(audio.active["decision"].action, Action.STOP)
        audio.update(decision(Action.SHIFT_RIGHT, 1.5), 1.5)
        audio.tick(2)
        self.assertEqual(
            [(clip["decision"].action, clip["start"]) for clip in audio.segments],
            [(Action.SLOW, 0), (Action.STOP, 1), (Action.SHIFT_RIGHT, 2)],
        )
        self.assertEqual([clip["end"] for clip in audio.segments], [1, 2, 3])
        self.assertFalse(any(event["event"] == "cancelled" for event in audio.events))

    def test_stop_and_both_sides_share_priority_and_wait_for_valid_peers(self):
        for side in (Action.SHIFT_LEFT, Action.SHIFT_RIGHT):
            self.assertEqual(SPEECH_PRIORITY[Action.STOP], SPEECH_PRIORITY[side])
            for first, second in ((side, Action.STOP), (Action.STOP, side)):
                with self.subTest(first=first, second=second):
                    audio = AudioTimeline(Bank(.5), repeat_s=0)
                    audio.update(decision(first, 0), 0)
                    audio.update(decision(second, .2), .2)
                    self.assertEqual(audio.active["decision"].action, first)
                    self.assertEqual(audio.pending.action, second)
                    audio.tick(.5)
                    self.assertEqual(audio.active["decision"].action, second)
                    self.assertEqual(audio.active["start"], .5)
                    self.assertFalse(any(event["event"] == "cancelled" for event in audio.events))

    def test_repeat_cooldown_does_not_cut_the_routine_phrase(self):
        audio = AudioTimeline(Bank(.5), repeat_s=4)
        audio.update(decision(Action.SHIFT_LEFT, 0), 0)
        audio.tick(.5)
        audio.update(decision(Action.SLOW, .6), .6)
        audio.update(decision(Action.SHIFT_LEFT, .8), .8)
        self.assertEqual(audio.active["decision"].action, Action.SLOW)
        self.assertEqual(audio.active["end"], 1.1)
        self.assertFalse(any(event["event"] == "cancelled" for event in audio.events))

    def test_expired_priority_cannot_cut_valid_speech(self):
        audio = AudioTimeline(Bank(1))
        audio.update(decision(Action.SLOW, 0), 0)
        audio.update(decision(Action.SHIFT_RIGHT, .1, ttl=.2), .5)
        self.assertEqual(audio.active["decision"].action, Action.SLOW)
        self.assertEqual(audio.active["end"], 1)

    def test_stop_cooldown_applies_to_continuing_stop_only(self):
        audio = AudioTimeline(Bank(.2), repeat_s=4)
        audio.update(decision(Action.STOP, 0), 0)
        audio.tick(.2)
        audio.update(decision(Action.STOP, .25), .25)
        self.assertEqual(len(audio.segments), 1)
        audio.update(decision(Action.SLOW, .3), .3)
        audio.tick(.5)
        audio.update(decision(Action.STOP, .6), .6)
        self.assertEqual(audio.active["decision"].action, Action.STOP)
        self.assertEqual(audio.active["start"], .6)

    def test_late_tick_does_not_cancel_an_already_finished_direction(self):
        audio = AudioTimeline(Bank(.4))
        audio.update(decision(Action.SHIFT_RIGHT, 0), 0)
        audio.tick(.8)
        self.assertIsNone(audio.active)
        self.assertEqual([event["event"] for event in audio.events], ["requested", "scheduled", "finished"])

    def test_new_stop_waits_then_bypasses_old_stop_cooldown(self):
        a=AudioTimeline(Bank(.2),repeat_s=4)
        a.update(decision(Action.STOP,0),0)
        a.tick(.2)
        a.update(decision(Action.SHIFT_LEFT,.3),.3)
        a.update(replace(decision(Action.STOP,.4), invalidates_route=True),.4)
        self.assertEqual(a.active["decision"].action,Action.SHIFT_LEFT)
        a.tick(.5)
        self.assertEqual(a.active["decision"].action,Action.STOP)
        self.assertEqual(a.active["start"],.5)

    def test_export_keeps_final_phrase_intact_without_starting_pending(self):
        audio = AudioTimeline(Bank(1))
        audio.update(decision(Action.SHIFT_LEFT, 0), 0)
        audio.update(replace(decision(Action.STOP, .4), invalidates_route=True), .4)
        self.assertEqual(audio.finish_export(.7), 1)
        self.assertEqual(len(audio.segments), 1)
        self.assertEqual(audio.segments[0]["end"], 1)
        self.assertEqual(audio.events[-1]["event"], "finished")

    def test_speech_crosses_into_forbidden_interval(self):
        data={"schema_version":1,"reviewed":True,"duration_s":3,"session_id":"s","location_id":"l","split":"test","intervals":[
            {"start":1,"end":2,"kind":"hazard","allowed_actions":["STOP"],"forbidden_actions":["SHIFT_RIGHT"]}]}
        records=[dict(decision(Action.SHIFT_RIGHT,0).record(),emitted_at_s=0),
                 dict(decision(Action.STOP,1).record(),emitted_at_s=1)]
        segments=[{"start":0,"end":1.5,"action":"SHIFT_RIGHT"}]
        result=score(data,records,[],segments)
        self.assertEqual(result["unsafe_direction_speech_count"] if "unsafe_direction_speech_count" in result else result["unsafe_speech_segment_count"],1)

    def test_detector_emits_before_slow_context_and_stale_context_dropped(self):
        class Fake:
            device="cpu"
            def __init__(self,role):
                self.spec=ModelSpec(role,"fake",role,"fake")
            def load(self): pass
            def close(self): pass
            def predict(self,f):
                if self.spec.role=="detector": return []
                time.sleep(.8)
                h,w=f.bgr.shape[:2]
                return {"classes":np.ones((h,w),np.int16),"confidence":np.ones((h,w)),"labels":{1:"sidewalk"}}
        cfg=Config([ModelSpec("detector","fake","detector","fake"),
                    ModelSpec("semantic","fake","semantic","fake")],mode="realtime_replay")
        e=GuideEngine(cfg,[Fake("detector"),Fake("semantic")],IdentityTracker()).load()
        observations=[]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"input.avi"
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),10,(320,240))
            for _ in range(10):writer.write(frame().bgr)
            writer.release()
            started=time.perf_counter()
            stats=run_replay(e,path,lambda s,d,r,t,stage: observations.append((t,stage,d.action)),lambda t:None,seconds=.5,start=started)
        self.assertTrue(observations)
        self.assertEqual(observations[0][1],"detector")
        self.assertLess(observations[0][0],.6)
        self.assertGreaterEqual(stats["stale_context"],1)
        self.assertFalse(any(a in {Action.KEEP_HEADING,Action.SHIFT_LEFT,Action.SHIFT_RIGHT} for _,_,a in observations))
        e.close()

    def test_stale_after_device_wait_is_never_inferred(self):
        from contextlib import contextmanager
        from unittest.mock import patch

        class Adapter:
            device = "cpu"
            spec = ModelSpec("detector", "fake", "detector", "fake")

            def predict(self, frame):
                raise AssertionError("expired frame reached inference")

        cfg = Config([Adapter.spec], mode="realtime_replay")
        engine = GuideEngine(cfg, [Adapter()], IdentityTracker())
        fake_time = [0.2]

        @contextmanager
        def wait_for_device(detector):
            self.assertTrue(detector)
            fake_time[0] = .51
            yield

        engine.gates["cpu"].acquire = wait_for_device
        with patch("guide_core.engine.time.perf_counter", side_effect=lambda: fake_time[0]):
            result, = engine.predict(frame(), {"detector"})
        self.assertEqual(result.skipped, "expired_waiting_for_device")
        self.assertIsNone(result.error)
        self.assertEqual(dict(engine.timings), {})

if __name__=="__main__":unittest.main()

