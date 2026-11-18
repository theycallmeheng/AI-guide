from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

@dataclass
class ModelSpec:
    id: str
    adapter: str
    role: str
    path: str
    enabled: bool = True
    device: str = "auto"
    imgsz: int = 480
    confidence: float = .30
    label_map: dict[str, str] = field(default_factory=dict)

@dataclass
class Config:
    models: list[ModelSpec]
    environment: str = "sidewalk"
    mode: str = "offline"
    sample_fps: float = 10.
    max_side: int = 640
    mirror_input: bool = False
    max_age_s: float = .7
    max_input_age_s: float = .5
    stable_s: float = .4
    stable_observations: int = 3
    track_ttl_s: float = .5
    predict_horizon_s: float = .5
    semantic_threshold: float = .60
    grid_width: int = 64
    grid_height: int = 48
    margin_top_cells: float = .8
    margin_bottom_cells: float = 2.
    confirm_horizon_ratio: float = .70
    temporal_ema_alpha: float = .65
    ttc_stop_s: float = 1.6
    head_hazard_enabled: bool = True
    depth_obstacle_enabled: bool = True
    preempt_emergency_audio: bool = False
    smart_arbitration: bool = False
    post_stop_hold_s: float = .8
    tail_grace_s: float = .15
    revocation_debounce_frames: int = 2
    earcon_enabled: bool = False
    stereo_audio: bool = False
    audio_dir: str = "assets/audio/guide_core/khanh_vy"
    audio_enabled: bool = True
    seed: int = 42

    def validate(self):
        active = [m for m in self.models if m.enabled]
        if not active or not any(m.role == "detector" for m in active):
            raise ValueError("At least one detector is required.")
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError("Model IDs must be unique.")
        for role in ("semantic", "depth"):
            if sum(m.role == role for m in active) > 1:
                raise ValueError("Select one active " + role + " provider.")
        if self.environment not in {"sidewalk", "alley", "indoor"}:
            raise ValueError("Unknown environment.")
        if self.mode not in {"offline", "realtime_replay"}:
            raise ValueError("Unknown mode.")
        if not 0 < self.sample_fps <= 30 or not 0 < self.max_input_age_s <= self.max_age_s <= .7:
            raise ValueError("Invalid sample rate/freshness limits.")
        if self.grid_width < 16 or self.grid_height < 16 or self.max_side < 128:
            raise ValueError("Analysis/grid resolution is too small.")
        if self.stable_s < .4 or self.stable_observations < 3:
            raise ValueError("Route confirmation requires >=0.4s and >=3 observations.")
        if not .45 <= self.confirm_horizon_ratio <= 1.0 or not .1 <= self.temporal_ema_alpha <= 1.0:
            raise ValueError("Invalid corridor horizon or temporal smoothing.")
        if not .3 <= self.ttc_stop_s <= 5.0:
            raise ValueError("Invalid TTC stop threshold.")
        if not 0. <= self.post_stop_hold_s <= 3.0 or not 0. <= self.tail_grace_s <= .5 or self.revocation_debounce_frames < 1:
            raise ValueError("Invalid audio arbitration timing parameters.")
        for m in active:
            if m.role not in {"detector", "semantic", "depth"} or not 0 < m.confidence <= 1 or m.imgsz < 64:
                raise ValueError("Invalid model specification: " + m.id)
            roles={
                "yolo":"detector","openvino_yolo":"detector",
                "segformer":"semantic","openvino_segformer":"semantic",
                "depth_anything":"depth","openvino_depth":"depth",
            }
            if m.adapter in roles and roles[m.adapter]!=m.role:
                raise ValueError("Adapter/role mismatch: "+m.id)
        if not 0<self.semantic_threshold<=1 or not 0<self.track_ttl_s<=.5:
            raise ValueError("Invalid confidence/track age")

    def as_dict(self):
        return asdict(self)

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["models"] = [ModelSpec(**item) for item in data["models"]]
        config = cls(**data)
        config.validate()
        return config

