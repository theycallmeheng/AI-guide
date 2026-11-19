"""Bounded replay: one latest frame, one detector job, one context job; no GPU overlap."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time
import uuid
import cv2
from .types import Frame
from .fusion import fuse
from .planning import route_blocked, urgent

class LatestFrame:
    def __init__(self):
        self.lock=threading.Lock()
        self.value=None
        self.dropped=0
    def put(self,frame):
        with self.lock:
            if self.value is not None: self.dropped+=1
            self.value=frame
    def pop(self):
        with self.lock:
            value,self.value=self.value,None
            return value

def run_replay(engine,video,emit,tick,seconds=None,start=None):
    c=engine.config
    slot=LatestFrame()
    stopped=threading.Event()
    done=threading.Event()
    stats={"decoded_frames":0,"decode_seconds":0.,"stale_frames":0,"stale_context":0,
           "skipped_model_requests":0,"vetoed_context":0,"source_duration_s":0.}
    start=time.perf_counter() if start is None else start
    session="replay-"+uuid.uuid4().hex[:12]
    reader_error=[]
    def decode():
        cap=cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened(): raise RuntimeError("Cannot open video")
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.
            seq=0
            while not stopped.is_set():
                pts=seq/fps
                if seconds is not None and pts>=seconds: break
                wait=start+pts-time.perf_counter()
                if wait>0 and stopped.wait(wait): break
                t=time.perf_counter()
                ok,image=cap.read()
                stats["decode_seconds"]+=time.perf_counter()-t
                if not ok: break
                stats["decoded_frames"]+=1
                stats["source_duration_s"]=(seq+1)/fps
                slot.put(Frame.prepare(image,session,seq,pts,c.max_side,c.mirror_input,received_at=start+pts))
                seq+=1
        except Exception as exc:
            reader_error.append(str(exc))
        finally:
            cap.release();done.set()
    reader=threading.Thread(target=decode,daemon=True)
    reader.start()
    detector=None
    context=None
    last_scene=None
    last_context_seq=-1
    later_scenes=[]
    context_roles={model.role for model in c.models if model.enabled and model.role!="detector"}
    last_stale_emit=-1.
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            while True:
                now=time.perf_counter()
                if detector and detector[1].done():
                    frame,future=detector
                    results=future.result()
                    stats["skipped_model_requests"]+=sum(bool(result.skipped) for result in results)
                    detector=None
                    scene=fuse(frame,results)
                    if scene.detector_valid:
                        scene=engine.tracker.update(scene)
                    now=time.perf_counter()
                    last_scene=scene
                    if context and now-context[0].frame.received_at<=c.max_age_s:
                        later_scenes.append(scene)
                    if now-frame.received_at<=c.max_age_s:
                        decision=engine.policy.update(scene,None,now,False)
                        decision=replace(decision,valid_until=min(decision.valid_until,frame.received_at+c.max_age_s))
                        engine.policy.active=decision
                        emit(scene,decision,results,now-start,"detector")
                        if context is None and context_roles and scene.observation_valid:
                            future=pool.submit(engine.predict,frame,context_roles)
                            context=(scene,results,future)
                            later_scenes=[]
                    else:
                        decision=engine.policy.stale(frame,now)
                        emit(scene,decision,results,now-start,"stale_detector")
                if context and context[2].done():
                    anchor,det_results,future=context
                    context_results=future.result()
                    stats["skipped_model_requests"]+=sum(bool(result.skipped) for result in context_results)
                    results=det_results+context_results
                    context=None
                    frame=anchor.frame
                    # A cut/new session invalidates an outstanding context job.
                    usable=(now-frame.received_at<=c.max_age_s and frame.sequence>last_context_seq
                            and not any(result.skipped for result in context_results))
                    if usable:
                        scene=fuse(frame,results)
                        scene.detections=anchor.detections
                        scene.motion_invalid=anchor.motion_invalid
                        route=engine.planner.plan(scene)
                        now=time.perf_counter()
                        # New observations may veto old geometry, never validate it.
                        veto=any(not newer.observation_valid or newer.errors
                                 or route_blocked(route,newer) or urgent(newer) for newer in later_scenes)
                        if now-frame.received_at>c.max_age_s:
                            stats["stale_context"]+=1
                            continue
                        if veto:
                            stats["vetoed_context"]+=1
                            engine.policy.reset_confirmation()
                            later_scenes.clear()
                            continue
                        decision=engine.policy.update(scene,route,now,True)
                        # Evidence expiration remains tied to CAPTURE, not context completion.
                        decision=replace(decision,valid_until=frame.received_at+c.max_age_s)
                        engine.policy.active=decision
                        last_context_seq=frame.sequence
                        emit(scene,decision,results,now-start,"fused")
                    else:
                        stats["stale_context"]+=1
                    later_scenes.clear()
                if detector is None:
                    frame=slot.pop()
                    if frame:
                        if now-frame.received_at>c.max_input_age_s:
                            stats["stale_frames"]+=1
                        else:
                            detector=(frame,pool.submit(engine.predict,frame,{"detector"}))
                if last_scene and now-last_scene.frame.received_at>c.max_age_s and now-last_stale_emit>.2:
                    decision=engine.policy.stale(last_scene.frame,now,"no_fresh_camera_data")
                    emit(last_scene,decision,[],now-start,"watchdog")
                    last_stale_emit=now
                tick(time.perf_counter()-start)
                if done.is_set() and detector is None and context is None and slot.value is None:
                    break
                time.sleep(.005)
    finally:
        stopped.set();reader.join(timeout=2)
    if reader_error: raise RuntimeError("; ".join(reader_error))
    stats["dropped_frames"]=slot.dropped
    stats["elapsed_s"]=time.perf_counter()-start
    stats["clock_origin"]=start
    return stats

