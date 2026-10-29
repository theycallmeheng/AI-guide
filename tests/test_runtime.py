import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from types import SimpleNamespace
import wave
import numpy as np
from obstacle_detector import LatestFrame, FrameAnalysis, Obstacle, analyze_frame, VIETNAMESE_NAMES
from guidance import AlertPolicy, describe
from performance import Measurements

class FakeOutput:
    name="test PCM"
    def __init__(self):
        self.played=[]
        self.until=0
    def play(self,audio):
        self.played.append(audio)
        self.until=time.perf_counter()+audio.duration
    def busy(self): return time.perf_counter()<self.until
    def stop(self): self.until=0
    def close(self): self.stop()

def analysis(objects=(), unclear=False, clear=()):
    return FrameAnalysis(np.zeros((100,100,3),dtype=np.uint8), tuple(objects), unclear, "", 100, clear)
def obstacle(name="Person", band="gần", box=(40,40,60,99), direction="phía trước"):
    return Obstacle(name,.9,direction,band,box)

class PerceptionTests(unittest.TestCase):
    def test_latest_frame_never_accumulates(self):
        latest=LatestFrame()
        for n in range(100): latest.publish(np.array([n]))
        seq,frame,closed,_=latest.next_after(0)
        self.assertEqual(seq,100); self.assertEqual(frame.item(),99)
        latest.close()
        self.assertTrue(latest.next_after(seq)[2])

    def test_latency_storage_is_bounded(self):
        m=Measurements(limit=32)
        for n in range(1000): m.add(n)
        result=m.snapshot()
        self.assertEqual(result["count"],1000)
        self.assertEqual(result["sample_count"],32)

    def test_wide_object_at_side_can_block_centre(self):
        result=describe(analysis([obstacle(box=(10,10,65,99),direction="bên trái")]),VIETNAMESE_NAMES)
        self.assertEqual(result.action,"stop")

    def test_empty_detections_never_confirm_safe_path(self):
        policy=AlertPolicy()
        for now in (0,1,10):
            decision,speak=policy.update(analysis(),VIETNAMESE_NAMES,now)
            self.assertEqual(decision.action,"observe"); self.assertFalse(speak)

    def test_unclear_never_proposes_route(self):
        decision,_=AlertPolicy().update(analysis(unclear=True,clear=("bên trái","phía trước")),{},0)
        self.assertEqual(decision.action,"unclear")

    def test_side_suggestion_requires_stable_walkway_mask(self):
        policy=AlertPolicy()
        scene=analysis([obstacle()],clear=("bên trái",))
        self.assertEqual(policy.update(scene,{},0)[0].action,"stop")
        self.assertEqual(policy.update(scene,{},.9)[0].action,"stop")
        self.assertEqual(policy.update(scene,{},1.1)[0].action,"suggest_side")
        scene=analysis([obstacle()])
        self.assertEqual(policy.update(scene,{},1.2)[0].action,"stop")

    def test_direction_precedes_obstacle_only_with_stable_walkway(self):
        for side,command in (("bên trái","Rẽ trái"),("bên phải","Rẽ phải")):
            policy=AlertPolicy()
            scene=analysis([obstacle()],clear=(side,))
            first,_=policy.update(scene,VIETNAMESE_NAMES,0)
            self.assertEqual(first.action,"stop")
            routed,_=policy.update(scene,VIETNAMESE_NAMES,1.1)
            self.assertTrue(routed.message.startswith("Dừng lại. "+command))
            self.assertLess(routed.message.index(command),routed.message.index("Có người"))
        policy=AlertPolicy()
        scene=analysis([obstacle(box=(0,40,20,99),direction="bên trái")],clear=("phía trước",))
        first,_=policy.update(scene,VIETNAMESE_NAMES,0)
        self.assertTrue(first.message.startswith("Đi chậm lại."))
        forward,_=policy.update(scene,VIETNAMESE_NAMES,1.3)
        self.assertEqual(forward.action,"suggest_forward")
        self.assertTrue(forward.message.startswith("Đi thẳng"))
        self.assertIn("Có người",forward.message)

    def test_detector_without_walkway_never_invents_turn_or_forward(self):
        policy=AlertPolicy()
        scene=analysis([obstacle(box=(0,40,20,99),direction="bên trái")])
        for now in (0,2,8):
            guidance,_=policy.update(scene,VIETNAMESE_NAMES,now)
            self.assertEqual(guidance.action,"observe")
            self.assertTrue(guidance.message.startswith("Đi chậm lại."))

    def test_stop_speech_does_not_restart_for_changing_classes(self):
        policy=AlertPolicy()
        self.assertTrue(policy.update(analysis([obstacle()]),{},1)[1])
        self.assertFalse(policy.update(analysis([obstacle("Car")]),{},1.1)[1])
        self.assertTrue(policy.update(analysis([obstacle("Car")]),{},4.2)[1])

    def test_road_mask_does_not_count_as_walkway(self):
        detection=SimpleNamespace(cls=np.array(0),conf=np.array(.95),xyxy=np.array([[0,0,100,100]]))
        class Model:
            def __init__(self,name): self.name=name
            def predict(self,**kwargs):
                return [SimpleNamespace(names={0:self.name},boxes=[detection],
                    masks=SimpleNamespace(data=np.ones((1,100,100),dtype=np.float32)))]
        frame=np.random.default_rng(10).integers(0,255,(100,100,3),dtype=np.uint8)
        self.assertEqual(analyze_frame(frame,Model("road"),.3).clear_corridors,())
        self.assertEqual(len(analyze_frame(frame,Model("walkway"),.3).clear_corridors),3)

    def test_small_uncertain_hazard_invalidates_corridor(self):
        walkway=SimpleNamespace(cls=np.array(0),conf=np.array(.95),xyxy=np.array([[0,0,100,100]]))
        pole=SimpleNamespace(cls=np.array(1),conf=np.array(.2),xyxy=np.array([[22,70,25,95]]))
        class Model:
            def predict(self,**kwargs):
                return [SimpleNamespace(names={0:"walkway",1:"pole"},boxes=[walkway,pole],
                    masks=SimpleNamespace(data=np.stack([np.ones((100,100)),np.zeros((100,100))])))]
        frame=np.random.default_rng(10).integers(0,255,(100,100,3),dtype=np.uint8)
        self.assertNotIn("bên trái",analyze_frame(frame,Model(),.3).clear_corridors)

class SpeechTests(unittest.TestCase):
    def test_cancel_invalidates_inflight_synthesis(self):
        import feedback
        gate=threading.Event(); started=threading.Event(); events=queue.Queue()
        class Engine:
            name="Fake"
            def synthesize(self,text,path):
                if text=="delayed":
                    started.set(); gate.wait(2)
                with wave.open(str(path),"wb") as audio:
                    audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(16000); audio.writeframes(b"\0"*3200)
        with tempfile.TemporaryDirectory() as directory, patch.object(feedback,"create_engine",return_value=Engine()):
            worker=feedback.SpeechWorker(events,cache_dir=directory,output_factory=FakeOutput)
            deadline=time.perf_counter()+3
            while not (Path(directory)/"disconnected.wav").exists() and time.perf_counter()<deadline: time.sleep(.01)
            worker.speak("delayed")
            self.assertTrue(started.wait(2))
            worker.cancel(); gate.set(); time.sleep(.1)
            self.assertIsNone(worker._play)
            self.assertFalse(any(e["type"]=="feedback_latency" for e in list(events.queue)))
            worker.close()
            self.assertFalse(worker._play_thread.is_alive())

class WebTests(unittest.TestCase):
    def test_stop_ignores_late_frames_and_speech(self):
        from pipeline import GuidePipeline
        from guidance import Guidance
        pipe=GuidePipeline(speech=False)
        with patch.object(pipe.speech,"speak") as speak:
            pipe.stop()
            pipe.put({"type":"analysis", "analysis":analysis(), "guidance":Guidance("stale","stale"),
                      "captured_at":time.perf_counter()})
            pipe.speak("stale")
            self.assertIsNone(pipe.latest_analysis()[1])
            speak.assert_not_called()
        pipe.close()

    def test_state_and_csrf_and_no_video_path_escape(self):
        from pipeline import GuidePipeline
        from web_app import GuideServer
        pipe=GuidePipeline(speech=False)
        server=GuideServer(("127.0.0.1",0),pipe)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base+"/api/state") as r: self.assertFalse(json.load(r)["running"])
            bad=Request(base+"/api/stop",data=b"{}",method="POST")
            with self.assertRaises(HTTPError) as error: urlopen(bad)
            self.assertEqual(error.exception.code,403)
            bad=Request(base+"/api/start",data=json.dumps({"model":"../../secret.pt"}).encode(),
                        headers={"X-Guide-Token":server.token,"Content-Type":"application/json"},method="POST")
            with self.assertRaises(HTTPError) as error: urlopen(bad)
            self.assertEqual(error.exception.code,400)
            with urlopen(base+"/") as r: self.assertIn(b'lang="vi"',r.read())
        finally:
            server.shutdown();server.server_close();pipe.close()

if __name__=="__main__": unittest.main()
