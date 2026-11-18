"""Shared inference/fusion/planning API; no GUI or platform speech dependency."""
from contextlib import contextmanager
from collections import defaultdict
import threading
import time
from .models import REGISTRY, predict_result
from .types import ModelResult
from .fusion import fuse
from .tracking import MotionTracker
from .planning import CorridorPlanner
from .policy import DecisionPolicy

class DeviceGate:
    def __init__(self):
        self.condition=threading.Condition()
        self.busy=False
        self.waiting_detectors=0

    @contextmanager
    def acquire(self,detector=False):
        with self.condition:
            self.waiting_detectors+=int(detector)
            while self.busy or (not detector and self.waiting_detectors):
                self.condition.wait()
            self.waiting_detectors-=int(detector)
            self.busy=True
        try:
            yield
        finally:
            with self.condition:
                self.busy=False
                self.condition.notify_all()

class GuideEngine:
    def __init__(self,config,adapters=None,tracker=None):
        config.validate()
        self.config=config
        self.adapters=adapters if adapters is not None else [REGISTRY[s.adapter](s) for s in config.models if s.enabled]
        self.failures={}
        self.gates=defaultdict(DeviceGate)
        self.tracker=tracker or MotionTracker(config)
        self.planner=CorridorPlanner(config)
        self.policy=DecisionPolicy(config)
        self.timings=defaultdict(list)
        self._timing_lock=threading.Lock()

    def load(self):
        for adapter in self.adapters:
            try:
                adapter.load()
            except Exception as exc:
                self.failures[adapter.spec.id]=str(exc)
        if all(a.spec.id in self.failures for a in self.adapters if a.spec.role=="detector"):
            raise RuntimeError("No detector loaded: "+str(self.failures))
        return self

    def _predict_one(self,adapter,frame):
        if adapter.spec.id in self.failures:
            now=time.perf_counter()
            return ModelResult(adapter.spec.id,adapter.spec.role,frame.key,frame.geometry,now,now,
                               error=self.failures[adapter.spec.id])
        with self.gates[adapter.device].acquire(adapter.spec.role=="detector"):
            now = time.perf_counter()
            deadline = self.config.max_input_age_s if adapter.spec.role == "detector" else self.config.max_age_s
            if self.config.mode == "realtime_replay" and now - frame.received_at > deadline:
                return ModelResult(
                    adapter.spec.id, adapter.spec.role, frame.key, frame.geometry, now, now,
                    skipped="expired_waiting_for_device",
                )
            result=predict_result(adapter,frame)
        with self._timing_lock:
            self.timings[adapter.spec.id].append((result.finished_at-result.started_at)*1000)
        return result

    def predict(self,frame,roles):
        selected=[a for a in self.adapters if a.spec.role in roles]
        if len({a.device for a in selected}) <= 1:
            return [self._predict_one(a,frame) for a in selected]
        # Heterogeneous execution: run distinct hardware devices (e.g. CPU P-cores + Iris Xe iGPU) concurrently.
        by_device=defaultdict(list)
        for idx,adapter in enumerate(selected):
            by_device[adapter.device].append((idx,adapter))
        ordered=[None]*len(selected)
        def run_group(items):
            for idx,adapter in items:
                ordered[idx]=self._predict_one(adapter,frame)
        threads=[threading.Thread(target=run_group,args=(items,),daemon=True) for items in by_device.values()]
        for t in threads: t.start()
        for t in threads: t.join()
        return ordered

    def analyze(self,frame,results,clock,track=True,full=True):
        scene=fuse(frame,results)
        if track:
            scene=self.tracker.update(scene)
        route=self.planner.plan(scene) if full else None
        decision=self.policy.update(scene,route,clock,full)
        return scene,decision

    def process(self,frame):
        results=self.predict(frame,{"detector","semantic","depth"})
        clock=frame.pts if self.config.mode=="offline" else time.perf_counter()
        scene,decision=self.analyze(frame,results,clock)
        if self.config.mode!="offline" and clock-frame.received_at>self.config.max_age_s:
            decision=self.policy.stale(frame,clock)
        return scene,decision,results

    def close(self):
        for adapter in self.adapters:
            adapter.close()

