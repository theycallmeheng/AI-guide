"""Conservative image-based warnings. No box alone establishes a safe route."""
from dataclasses import dataclass
from fast_alerts import FIXED


@dataclass(frozen=True)
class Guidance:
    message: str
    key: str
    priority: int = 1
    clip: str | None = None
    action: str = "observe"


def describe(analysis, names):
    width = analysis.frame.shape[1]
    # A wide object can block the central corridor even if its centre is at a side.
    central = [o for o in analysis.obstacles if o.box[0] < width * .62 and o.box[2] > width * .38]
    blocking = [o for o in central if o.distance_band == "gần"]
    objects = blocking or list(analysis.obstacles)
    obstacle = min(objects, key=lambda o: ({"gần": 0, "vừa": 1, "xa": 2}[o.distance_band], -o.confidence), default=None)
    if obstacle:
        label = names.get(obstacle.class_name.lower().replace("_", " "), obstacle.class_name)
        text = f"Có {label} {obstacle.distance_band} ở {obstacle.direction}."
        key = f"{obstacle.class_name}|{obstacle.distance_band}|{obstacle.direction}"
        if blocking:
            return Guidance("Dừng lại. " + text, key, 3, "stop", "stop")
        if analysis.unclear:
            return Guidance("Chưa nhìn rõ. Hãy dừng để kiểm tra.", "unclear", 2, "unclear", "unclear")
        return Guidance(FIXED["slow_down"] + " " + text, key,
                        2 if obstacle.distance_band == "gần" else 1, "slow_down")
    if analysis.unclear:
        return Guidance("Chưa nhìn rõ. Hãy dừng để kiểm tra.", "unclear", 2, "unclear", "unclear")
    return Guidance("Chưa xác nhận được lối đi an toàn.", "unconfirmed", 0)


class AlertPolicy:
    def __init__(self):
        self.candidate = None
        self.candidate_since = 0.0
        self.last_key = None
        self.last_spoken = float("-inf")
        self.last_priority = 0
        self.last_action = None
        self.last_action_at = float("-inf")
        self.route = None
        self.route_since = 0.0

    def update(self, analysis, names, now):
        base = describe(analysis, names)
        # Only real, connected walkway segmentation may supply a route candidate.
        clear = getattr(analysis, "clear_corridors", ())
        route = next((c for c in ("bên trái", "bên phải") if c in clear), None) if base.action == "stop" else ("phía trước" if "phía trước" in clear else None)
        if analysis.unclear:
            route = None
        if route != self.route:
            self.route, self.route_since = route, now
        guidance = base
        if route and now - self.route_since >= 1.0:
            if base.action == "stop":
                command = "turn_left" if route == "bên trái" else "turn_right"
                detail = base.message.removeprefix(FIXED["stop"] + " ")
                guidance = Guidance(f"{FIXED['stop']} {FIXED[command]} {detail}",
                                    f"{command}|{base.key}", 2, None, "suggest_side")
            elif not analysis.obstacles:
                guidance = Guidance(FIXED["go_forward"], "route|front", 1, "go_forward", "suggest_forward")
            else:
                detail = base.message.removeprefix(FIXED["slow_down"] + " ")
                guidance = Guidance(f"{FIXED['go_forward']} {detail}", f"go_forward|{base.key}",
                                    base.priority, None, "suggest_forward")
        if guidance.key != self.candidate:
            self.candidate, self.candidate_since = guidance.key, now
        ready = guidance.priority >= 2 or now - self.candidate_since >= .12
        cooldown = 3.0 if guidance.priority >= 2 else 5.0
        due = guidance.key != self.last_key or now - self.last_spoken >= cooldown
        # A changing class label must not restart emergency speech on every frame.
        if guidance.action == "stop" and self.last_action == "stop" and now - self.last_action_at < .8:
            due = False
        if guidance.priority < self.last_priority and now - self.last_spoken < .4:
            due = False
        if ready and due and guidance.priority:
            self.last_key, self.last_spoken, self.last_priority = guidance.key, now, guidance.priority
            self.last_action, self.last_action_at = guidance.action, now
            return guidance, True
        return guidance, False
