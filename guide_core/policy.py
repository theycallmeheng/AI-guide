from .types import Action, Decision, DIRECTIONAL, SPEECH_PRIORITY
from .planning import urgent_reason, route_blocked, _vi_label
from .fusion import is_hazard

class DecisionPolicy:
    def __init__(self,config):
        self.c=config
        self.active=None
        self.candidate=None
        self.since=0.
        self.count=0
        self.last_key=None
        self.candidate_route=None
        self.last_pts=None

    @staticmethod
    def consistent(a,b,width):
        if not a or not b or not a.path or not b.path: return True
        import numpy as np
        m=min(len(a.path),len(b.path))
        if m<6 or abs(len(a.path)-len(b.path))>10:
            return False
        x=np.array(a.path[:m])[:,0];y=np.array(b.path[:m])[:,0]
        return float(np.max(np.abs(x-y)))<=width*.07

    @staticmethod
    def _nearest_hazard_hint(scene):
        hazards=[d for d in scene.detections if is_hazard(d)]
        if not hazards:
            return ""
        top=max(hazards,key=lambda d:(d.box[3],d.confidence))
        w=max(1,scene.frame.bgr.shape[1])
        cx=(top.box[0]+top.box[2])/(2*w)
        side="bên trái" if cx<.38 else "bên phải" if cx>.62 else "phía trước"
        name=_vi_label(top.label).replace(" phía trước","")
        return f"{name} {side}"

    def _emit(self,action,frame,clock,reason,route=None,hazard_hint=""):
        priority=SPEECH_PRIORITY.get(action,1)
        decision=Decision(action,clock,clock+self.c.max_age_s,frame.key,reason,route,priority,
                          invalidates_route=action == Action.STOP,hazard_hint=hazard_hint)
        self.active=decision
        if action not in DIRECTIONAL and reason not in {"confirming_corridor","waiting_context"}:
            self.reset_confirmation()
        return decision

    def reset_confirmation(self):
        self.candidate = None
        self.count = 0

    def stale(self,frame,clock,reason="observation_expired"):
        action=Action.STOP if self.active and self.active.action in DIRECTIONAL else Action.UNCLEAR
        return self._emit(action,frame,clock,reason)

    def update(self,scene,route,clock,full=True):
        frame=scene.frame
        previous=self.active
        is_urg,urg_reason,urg_hint=urgent_reason(
            scene,self.c.ttc_stop_s,self.c.head_hazard_enabled,self.c.depth_obstacle_enabled
        )
        if is_urg:
            return self._emit(Action.STOP,frame,clock,urg_reason,hazard_hint=urg_hint)
        if not scene.observation_valid:
            return self.stale(frame,clock,"poor_or_missing_observation")
        if previous and previous.action in DIRECTIONAL and route_blocked(previous.route,scene):
            return self._emit(Action.STOP,frame,clock,"previous_route_blocked",
                              hazard_hint=self._nearest_hazard_hint(scene))
        if not full:
            # Fast detector can veto a confirmed or candidate corridor, never renew expiry.
            if self.candidate_route and route_blocked(self.candidate_route,scene):
                self.reset_confirmation()
            if previous and previous.action in DIRECTIONAL:
                if clock>previous.valid_until: return self.stale(frame,clock)
                return previous
            if self.candidate and self.count>0 and self.last_pts is not None and 0<=frame.pts-self.last_pts<=self.c.max_age_s:
                return self._emit(Action.SLOW,frame,clock,"confirming_corridor",self.candidate_route)
            hint=self._nearest_hazard_hint(scene) if scene.detections else ""
            return self._emit(Action.SLOW if scene.detections else Action.UNCLEAR,frame,clock,"waiting_context",
                              hazard_hint=hint)
        if route is None or route.action is None or route.narrow:
            if previous and previous.action in DIRECTIONAL:
                return self._emit(Action.STOP,frame,clock,"route_no_longer_confirmed")
            action=Action.SLOW if route and route.evidence_complete else Action.UNCLEAR
            hint=self._nearest_hazard_hint(scene)
            return self._emit(action,frame,clock,route.reason if route else "context_unavailable",route,hint)
        if previous and previous.action in DIRECTIONAL and previous.action!=route.action:
            return self._emit(Action.STOP,frame,clock,"direction_changed_requires_confirmation")
        if previous and previous.action in DIRECTIONAL and not self.consistent(previous.route,route,frame.bgr.shape[1]):
            return self._emit(Action.STOP,frame,clock,"corridor_changed_requires_confirmation")
        gap=self.last_pts is not None and not 0<=frame.pts-self.last_pts<=self.c.max_age_s
        if self.candidate!=route.action or gap or not self.consistent(self.candidate_route,route,frame.bgr.shape[1]):
            self.candidate=route.action
            self.since=frame.pts
            self.count=0
        self.candidate_route=route
        self.last_pts=frame.pts
        if frame.key!=self.last_key:
            self.count+=1
            self.last_key=frame.key
        if self.count>=self.c.stable_observations and frame.pts-self.since>=self.c.stable_s-1e-8:
            clock_hint=f"hướng {route.clock_hour} giờ" if route.clock_hour and route.clock_hour!=12 else ""
            return self._emit(route.action,frame,clock,route.reason,route,clock_hint)
        return self._emit(Action.SLOW,frame,clock,"confirming_corridor",route)
