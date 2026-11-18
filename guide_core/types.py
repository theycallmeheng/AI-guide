from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import time
import cv2
import numpy as np

class Action(str, Enum):
    UNCLEAR = "UNCLEAR"
    STOP = "STOP"
    SLOW = "SLOW"
    KEEP_HEADING = "KEEP_HEADING"
    SHIFT_LEFT = "SHIFT_LEFT"
    SHIFT_RIGHT = "SHIFT_RIGHT"

SIDE_GUIDANCE = {Action.SHIFT_LEFT, Action.SHIFT_RIGHT}
DIRECTIONAL = {Action.KEEP_HEADING, *SIDE_GUIDANCE}
SPEECH_PRIORITY = {
    Action.STOP: 3,
    Action.SHIFT_LEFT: 3,
    Action.SHIFT_RIGHT: 3,
}
MESSAGES = {
    Action.UNCLEAR: "Chưa xác nhận được lối đi.",
    Action.STOP: "Dừng lại.",
    Action.SLOW: "Đi chậm lại.",
    Action.KEEP_HEADING: "Giữ hướng, đi chậm.",
    Action.SHIFT_LEFT: "Lệch nhẹ sang trái.",
    Action.SHIFT_RIGHT: "Lệch nhẹ sang phải.",
}

@dataclass(frozen=True)
class Frame:
    session: str
    sequence: int
    pts: float
    received_at: float
    bgr: np.ndarray = field(repr=False, compare=False)
    original_hw: tuple[int, int] = (0, 0)
    mirrored_input: bool = False

    @property
    def key(self):
        return (self.session, self.sequence)

    @property
    def geometry(self):
        return (self.original_hw, self.bgr.shape[:2], self.mirrored_input)

    @property
    def transform(self):
        h,w=self.bgr.shape[:2]
        oh,ow=self.original_hw if self.original_hw!=(0,0) else (h,w)
        sx,sy=w/ow,h/oh
        return {"original_hw":[oh,ow],"analysis_hw":[h,w],"mirror_corrected":self.mirrored_input,
                "original_to_analysis_edges":[[-sx if self.mirrored_input else sx,0,w if self.mirrored_input else 0],
                                              [0,sy,0],[0,0,1]]}

    @classmethod
    def prepare(cls, image, session, sequence, pts, max_side=640, mirror=False, received_at=None):
        original = image.shape[:2]
        if mirror:
            image = cv2.flip(image, 1)  # Canonical camera-relative left/right before all models.
        scale = min(1., max_side / max(original))
        if scale < 1:
            image = cv2.resize(image, (round(original[1]*scale), round(original[0]*scale)))
        return cls(session, sequence, pts, time.perf_counter() if received_at is None else received_at,
                   image, original, mirror)

@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    sources: tuple[str, ...]
    track_id: int | None = None
    velocity: tuple[float, float] = (0., 0.)
    scale_rate: float = 0.
    ttc_s: float | None = None
    zone: str = "ground"

@dataclass
class ModelResult:
    model_id: str
    role: str
    frame_key: tuple[str, int]
    geometry: tuple
    started_at: float
    finished_at: float
    payload: object = None
    error: str | None = None
    skipped: str | None = None

@dataclass
class Scene:
    frame: Frame
    detections: list[Detection]
    semantic: dict | None = None
    depth: dict | None = None
    errors: list[str] = field(default_factory=list)
    motion_invalid: bool = False
    blur_score: float = 100.
    brightness: float = 100.
    detector_valid: bool = True
    warp: np.ndarray | None = field(default=None, repr=False)
    depth_anomalies: dict | None = field(default=None, repr=False)

    @property
    def observation_valid(self):
        return self.detector_valid and not self.motion_invalid and self.blur_score >= 12 and self.brightness >= 18

@dataclass
class Route:
    action: Action | None
    path: list[tuple[int, int]] = field(default_factory=list)
    mask: np.ndarray | None = field(default=None, repr=False)
    states: np.ndarray | None = field(default=None, repr=False)
    clearance: float = 0.
    narrow: bool = False
    reason: str = ""
    evidence_complete: bool = False
    horizon_ratio: float = 0.
    clock_hour: int | None = None

@dataclass(frozen=True)
class Decision:
    action: Action
    clock: float
    valid_until: float
    frame_key: tuple[str, int]
    reason: str
    route: Route | None = field(default=None, compare=False)
    priority: int = 1
    invalidates_route: bool = False
    hazard_hint: str = ""

    @property
    def text(self):
        base = MESSAGES[self.action]
        if self.hazard_hint:
            return f"{base[:-1]} • {self.hazard_hint}."
        return base

    def record(self):
        return {"action": self.action.value, "time_s": self.clock, "valid_until": self.valid_until,
                "frame_key": list(self.frame_key), "reason": self.reason, "priority": self.priority,
                "invalidates_route": self.invalidates_route,
                "hazard_hint": self.hazard_hint,
                "clock_hour": self.route.clock_hour if self.route else None,
                "horizon_ratio": round(self.route.horizon_ratio, 3) if self.route else 0.,
                "path": self.route.path if self.route else []}

