import unittest
import time
import threading
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from guide_core.config import Config,ModelSpec
from guide_core.types import Frame,Detection,ModelResult,Scene,Action,Decision,Route
from guide_core.fusion import fuse,merge_detections
from guide_core.planning import CorridorPlanner,obstacle_mask,route_blocked
from guide_core.policy import DecisionPolicy
from guide_core.audio import AudioTimeline
from guide_core.engine import GuideEngine,DeviceGate
from guide_core.replay import LatestFrame
from guide_core.evaluate import score,check_split_leakage

def config(depth=True):
    specs=[ModelSpec("y","yolo","detector","fake")]
    if depth: specs.append(ModelSpec("d","depth_anything","depth","fake"))
    return Config(specs,environment="sidewalk")

def frame(seq=0,pts=0):
    image=np.random.default_rng(42).integers(40,220,(240,320,3),dtype=np.uint8)
    return Frame.prepare(image,"s",seq,pts,received_at=0)

def scene(seq=0,pts=0):
    f=frame(seq,pts)
    ids=np.ones(f.bgr.shape[:2],np.int16)
    return Scene(f,[],{"classes":ids,"confidence":np.ones_like(ids,dtype=float),"labels":{0:"wall",1:"sidewalk",2:"road",3:"floor",4:"door"}},
                 {"relative_nearness":np.tile(np.linspace(.1,.7,240)[:,None],(1,320)),"valid":np.ones_like(ids,dtype=bool),"unit":"relative"})

def result(f,role,payload):
    return ModelResult(role,role,f.key,f.geometry,0,.1,payload)

def decision(action,t=0,ttl=.7):
    return Decision(action,t,t+ttl,("s",int(t*10)),"test")

class Bank:
    rate=100
    def __init__(self,duration=1): self.duration=duration
    def samples(self,d): return np.ones(round(self.duration*self.rate),dtype="<i2")

class IdentityTracker:
    def update(self,s): return s

class CoreTests(unittest.TestCase):
    def test_bypass_waits_for_a_confirmed_side(self):
        for action, reason in ((Action.SHIFT_LEFT, "bypass_left"), (Action.SHIFT_RIGHT, "bypass_right")):
            with self.subTest(side=action):
                policy = DecisionPolicy(config())
                blocked = Route(None, reason="forward_corridor_blocked", evidence_complete=True)
                self.assertEqual(policy.update(scene(0, 0), blocked, 0).action, Action.STOP)
                obscured = Route(None, reason="no_continuous_corridor", evidence_complete=True)
                scan = policy.update(scene(1, .1), obscured, .1)
                self.assertEqual((scan.action, scan.reason), (Action.UNCLEAR, "scan_for_bypass"))
                lateral = Route(action, evidence_complete=True)
                for seq, t in ((2, .2), (3, .4)):
                    self.assertNotEqual(policy.update(scene(seq,t), lateral, t).action, action)
                confirmed = policy.update(scene(4, .6), lateral, .6)
                self.assertEqual((confirmed.action, confirmed.reason), (action, reason))

    def test_indoor_transverse_step_requires_two_distinct_frames(self):
        cfg = config()
        cfg.environment = "indoor"
        policy = DecisionPolicy(cfg)
        first = scene(1, .1)
        first.detections = [Detection("stairs", .53, (0, 0, 320, 130), ("y",))]
        self.assertNotEqual(policy.update(first, None, .1, full=False).action, Action.STOP)
        self.assertNotEqual(policy.update(first, None, .15).action, Action.STOP)
        second = scene(2, .2)
        second.detections = first.detections.copy()
        stop = policy.update(second, None, .2, full=False)
        self.assertEqual((stop.action, stop.reason), (Action.STOP, "transverse_step_ahead"))
        self.assertEqual(policy.update(scene(3, .3), None, .3, full=False).action, Action.STOP)
        self.assertNotEqual(policy.update(scene(6, .6), None, .6, full=False).action, Action.STOP)

    def test_confirmed_barrier_stops_but_unknown_ground_does_not(self):
        cfg = config(depth=False)
        planner, policy = CorridorPlanner(cfg), DecisionPolicy(cfg)
        trial = scene()
        trial.semantic["classes"][140:162, :] = 0
        trial.detections = [Detection("chair", .85, (0, 140, 320, 162), ("y",))]
        route = planner.plan(trial)
        self.assertEqual(route.reason, "forward_corridor_blocked")
        self.assertEqual(policy.update(trial, route, 0).action, Action.STOP)
        trial.detections.clear()
        route = planner.plan(trial)
        self.assertNotEqual(route.reason, "forward_corridor_blocked")
        self.assertNotEqual(policy.update(trial, route, .1).action, Action.STOP)
        unknown = Route(None, reason="no_continuous_corridor", evidence_complete=True)
        self.assertEqual(policy.update(trial, unknown, .2).action, Action.UNCLEAR)

    def test_new_detection_revokes_spoken_route_during_confirmation(self):
        audio = AudioTimeline(Bank(1.0), smart_arbitration=True)
        policy = DecisionPolicy(config())
        policy.spoken_direction = audio.direction_in_effect
        mask = np.zeros((240, 320), dtype=bool)
        mask[145:200, 250:300] = True
        right = Route(Action.SHIFT_RIGHT, mask=mask)
        spoken = replace(decision(Action.SHIFT_RIGHT), route=right)
        audio.update(spoken, 0.0)

        rechecking = policy.update(scene(2, .2), Route(Action.SHIFT_LEFT), .2)
        self.assertEqual(rechecking.action, Action.SLOW)
        audio.update(rechecking, .2)
        crossing = scene(8, .8)
        crossing.detections = [Detection("person", .9, (253, 145, 290, 190), ("y",))]
        stop = policy.update(crossing, None, .8, full=False)
        self.assertEqual(stop.reason, "spoken_route_blocked")
        audio.update(stop, .8)
        self.assertEqual(audio.segments[0]["end"], .8)
        self.assertEqual(audio.active["decision"].action, Action.STOP)
        self.assertEqual(audio.segments[1]["start"], .8)

    def test_cross_frame_fusion_rejected(self):
        f=frame()
        r=result(frame(1),"detector",[])
        with self.assertRaises(ValueError): fuse(f,[r])
    def test_cross_session_rejected(self):
        f=frame()
        with self.assertRaises(ValueError): fuse(f,[result(replace(f,session="different"),"detector",[])])
    def test_geometry_rejected(self):
        f=frame();r=result(f,"detector",[]);r.geometry=((480,640),(240,320),False)
        with self.assertRaises(ValueError): fuse(f,[r])
    def test_mirror_canonicalized(self):
        x=np.zeros((4,6,3),np.uint8);x[:,0]=255
        f=Frame.prepare(x,"s",0,0,mirror=True)
        self.assertTrue((f.bgr[:,-1]==255).all())
        self.assertTrue(f.mirrored_input)
    def test_merge_retains_provenance_not_sum(self):
        a=Detection("bike",.7,(1,2,20,30),("a",))
        b=Detection("bicycle",.9,(1,2,20,30),("b",))
        out=merge_detections([a,b])
        self.assertEqual(len(out),1);self.assertEqual(out[0].confidence,.9)
        self.assertEqual(out[0].sources,("a","b"))
    def test_full_continuous_ground(self):
        route=CorridorPlanner(config()).plan(scene())
        self.assertEqual(route.action,Action.KEEP_HEADING)
        self.assertTrue(route.evidence_complete)
    def test_disconnected_ground_cannot_route(self):
        s=scene();s.semantic["classes"][165:190]=0
        self.assertIsNone(CorridorPlanner(config()).plan(s).action)
    def test_door_does_not_mean_open(self):
        s=scene();s.semantic["classes"][:]=4
        self.assertIsNone(CorridorPlanner(config()).plan(s).action)
    def test_floor_only_indoor(self):
        s=scene();s.semantic["classes"][:]=3
        c=config()
        self.assertIsNone(CorridorPlanner(c).plan(s).action)
        c.environment="indoor"
        self.assertEqual(CorridorPlanner(c).plan(s).action,Action.KEEP_HEADING)
    def test_road_context(self):
        s=scene();s.semantic["classes"][:]=2
        c=config()
        self.assertIsNone(CorridorPlanner(c).plan(s).action)
        c.environment="alley"
        self.assertEqual(CorridorPlanner(c).plan(s).reason,"road_candidate_traffic_risk")
    def test_traffic_cancels_road_candidate(self):
        s=scene();s.semantic["classes"][:]=2;s.detections=[Detection("car",.9,(0,0,25,25),("a",))]
        c=config();c.environment="alley"
        self.assertIsNone(CorridorPlanner(c).plan(s).action)
    def test_invalid_depth_blocks_direction(self):
        s=scene();s.depth["valid"][:]=False
        self.assertEqual(CorridorPlanner(config()).plan(s).reason,"invalid_depth")
    def test_nan_depth_blocks_direction(self):
        s=scene();s.depth["relative_nearness"][:]=np.nan
        self.assertIsNone(CorridorPlanner(config()).plan(s).action)
    def test_missing_depth_ablation_explicit(self):
        s=scene();s.depth=None
        self.assertIsNone(CorridorPlanner(config()).plan(s).action)
        self.assertIsNotNone(CorridorPlanner(config(False)).plan(s).action)
    def test_unknown_not_free(self):
        s=scene();s.semantic["confidence"][:]=.1
        self.assertIsNone(CorridorPlanner(config()).plan(s).action)
    def test_narrow_slit_not_confirmed(self):
        s=scene();s.semantic["classes"][:]=0;s.semantic["classes"][:,157:163]=1
        self.assertIsNone(CorridorPlanner(config()).plan(s).action)
    def test_prediction_only_expands(self):
        s=scene();d=Detection("person",.8,(10,100,30,180),("y",),1,(100,0))
        s.detections=[d];mask=obstacle_mask(s)
        self.assertTrue(mask[150,15]);self.assertTrue(mask[150,70])
    def test_stable_observations_and_duration(self):
        c=config();p=DecisionPolicy(c);planner=CorridorPlanner(c)
        out=[]
        for n,t in enumerate([0,.1,.2,.4]):
            s=scene(n,t);out.append(p.update(s,planner.plan(s),t).action)
        self.assertEqual(out[:3],[Action.SLOW]*3)
        self.assertEqual(out[3],Action.KEEP_HEADING)
    def test_repeated_frame_not_three_observations(self):
        p=DecisionPolicy(config());s=scene();r=CorridorPlanner(config()).plan(s)
        for t in [0,.2,.4,.8]: d=p.update(s,r,t)
        self.assertEqual(d.action,Action.SLOW)
    def test_crossing_obstacle_veto_immediate(self):
        c=config();p=DecisionPolicy(c);planner=CorridorPlanner(c)
        for n,t in enumerate([0,.2,.4]):
            s=scene(n,t);d=p.update(s,planner.plan(s),t)
        self.assertEqual(d.action,Action.KEEP_HEADING)
        s=scene(3,.5);s.detections=[Detection("person",.7,(145,120,180,185),("y",))]
        d=p.update(s,None,.5,False)
        self.assertEqual(d.action,Action.STOP)
    def test_fast_detection_never_renews_route_expiry(self):
        c=config();p=DecisionPolicy(c);r=CorridorPlanner(c).plan(scene())
        for n,t in enumerate([0,.2,.4]): d=p.update(scene(n,t),r,t)
        old=d.valid_until
        d=p.update(scene(4,.6),None,.6,False);self.assertEqual(d.valid_until,old)
        d=p.update(scene(5,2),None,2,False);self.assertEqual(d.action,Action.STOP)
    def test_blur_and_missing_detector_unclear(self):
        p=DecisionPolicy(config());s=scene();s.blur_score=0
        self.assertEqual(p.update(s,None,0).action,Action.UNCLEAR)
        s.blur_score=100;s.detector_valid=False
        self.assertEqual(p.update(s,None,1).action,Action.UNCLEAR)
    def test_manhole_not_open_hole(self):
        d=Detection("manhole",.9,(20,20,50,50),("rod",))
        self.assertEqual(merge_detections([d])[0].label,"manhole")
    def test_audio_finish_then_latest(self):
        a=AudioTimeline(Bank(.5),repeat_s=0)
        a.update(decision(Action.SLOW,0),0)
        a.update(decision(Action.UNCLEAR,.1),.1)
        a.update(decision(Action.KEEP_HEADING,.2),.2)
        a.tick(.5)
        self.assertEqual([s["decision"].action for s in a.segments],[Action.SLOW,Action.KEEP_HEADING])
        self.assertGreaterEqual(a.segments[1]["start"],a.segments[0]["end"])
    def test_audio_invalidated_direction_finishes_before_stop(self):
        a=AudioTimeline(Bank())
        a.update(decision(Action.SHIFT_LEFT),0)
        a.update(decision(Action.STOP,.4),.4)
        self.assertEqual(a.active["decision"].action,Action.SHIFT_LEFT)
        a.tick(1)
        self.assertEqual(a.active["decision"].action,Action.STOP)
        self.assertEqual(a.segments[0]["end"],1)
    def test_pending_expiry(self):
        a=AudioTimeline(Bank(2))
        a.update(decision(Action.SLOW,0,3),0)
        a.update(decision(Action.UNCLEAR,.1),.1)
        a.tick(2)
        self.assertEqual(len(a.segments),1)
    def test_pending_refreshed_even_cooldown(self):
        a=AudioTimeline(Bank(1))
        a.update(decision(Action.SLOW,0,3),0)
        a.update(decision(Action.UNCLEAR,.1),.1)
        a.update(decision(Action.UNCLEAR,.7),.7)
        a.update(decision(Action.UNCLEAR,.9),.9)
        a.tick(1)
        self.assertEqual(a.active["decision"].action,Action.UNCLEAR)
        started=[e for e in a.events if e["event"]=="scheduled"][-1]
        self.assertAlmostEqual(started["queue_wait_s"],.9)
    def test_direction_expiry_does_not_interrupt_started_speech(self):
        a=AudioTimeline(Bank(2))
        a.update(decision(Action.SHIFT_RIGHT),0);a.tick(.71)
        self.assertEqual(a.active["decision"].action,Action.SHIFT_RIGHT)
        a.tick(2)
        self.assertIsNone(a.active)
        self.assertEqual(a.segments[0]["end"],2)
    def test_stop_mute_interrupts_and_clears(self):
        a=AudioTimeline(Bank());a.update(decision(Action.SLOW),0)
        a.update(decision(Action.UNCLEAR,.1),.1);a.stop(.2);a.tick(2)
        self.assertIsNone(a.active);self.assertIsNone(a.pending)
        self.assertEqual(a.segments[0]["end"],.2)
    def test_latest_buffer_bounded(self):
        q=LatestFrame()
        for n in range(50):q.put(frame(n,n/10))
        self.assertEqual(q.pop().sequence,49);self.assertEqual(q.dropped,49);self.assertIsNone(q.pop())
    def test_model_error_becomes_observable(self):
        class Bad:
            spec=ModelSpec("bad","test","detector","fake")
            device="cpu"
            def load(self):pass
            def predict(self,f):raise RuntimeError("test model failure")
            def close(self):pass
        e=GuideEngine(config(False),[Bad()],IdentityTracker()).load()
        _,d,results=e.process(frame())
        self.assertEqual(d.action,Action.UNCLEAR)
        self.assertIn("test model failure",results[0].error)
    def test_gpu_gate_serializes(self):
        gate=DeviceGate();concurrent=[0];peak=[0]
        def job():
            with gate.acquire():
                concurrent[0]+=1;peak[0]=max(peak[0],concurrent[0]);time.sleep(.02);concurrent[0]-=1
        threads=[threading.Thread(target=job) for _ in range(3)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(peak[0],1)
    def test_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            check_split_leakage([{"location_id":"a","session_id":"1","split":"validation"},{"location_id":"a","session_id":"2","split":"test"}])
    def test_always_stop_has_zero_direction_coverage(self):
        data={"schema_version":1,"reviewed":True,"duration_s":10,"session_id":"s","location_id":"l","split":"test","intervals":[
            {"start":0,"end":2,"kind":"hazard","allowed_actions":["STOP"],"forbidden_actions":["SHIFT_RIGHT"]},
            {"start":2,"end":10,"kind":"guidance","allowed_actions":["SHIFT_LEFT"],"forbidden_actions":["SHIFT_RIGHT"]}]}
        records=[dict(decision(Action.STOP,t).record(),emitted_at_s=t) for t in [1,3]]
        scored=score(data,records,[])
        self.assertEqual(scored["decision_event_recall"],1)
        self.assertEqual(scored["correct_direction_opportunity_coverage"],0)
        self.assertTrue(scored["preliminary"])
        self.assertFalse(scored["accuracy_targets_met"])
        # Even enough hazard/clear footage cannot pass with zero directional coverage.
        data["duration_s"]=2000
        data["intervals"]=[{"start":i*2,"end":i*2+1,"kind":"hazard",
            "allowed_actions":["STOP"],"forbidden_actions":["SHIFT_RIGHT"]} for i in range(100)]
        data["intervals"] += [{"start":200,"end":2000,"kind":"clear","allowed_actions":["KEEP_HEADING"],"forbidden_actions":["STOP"]},
                              {"start":200,"end":210,"kind":"guidance","allowed_actions":["KEEP_HEADING"],"forbidden_actions":["SHIFT_RIGHT"]}]
        records=[dict(decision(Action.STOP,i*2+.5).record(),emitted_at_s=i*2+.5) for i in range(100)]
        audio=[dict(r,event="scheduled") for r in records]
        scored=score(data,records,audio)
        self.assertFalse(scored["preliminary"])
        self.assertEqual(scored["scheduled_or_submitted_alert_event_recall"],1)
        self.assertEqual(scored["false_alerts_per_clear_minute"],0)
        self.assertFalse(scored["accuracy_targets_met"])

if __name__=="__main__":unittest.main()

