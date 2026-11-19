"""Single-voice scheduler with priority queuing, earcons and stereo panning."""
from pathlib import Path
import wave
import numpy as np
from .types import Action, DIRECTIONAL, SIDE_GUIDANCE, SPEECH_PRIORITY

CLIPS={a:a.value.lower()+".wav" for a in Action}

ARBITRATION_TIER = {
    **SPEECH_PRIORITY,
    Action.KEEP_HEADING: 2,
    Action.SLOW: 1,
    Action.UNCLEAR: 1,
}

# Asymmetric repeat cooldowns (seconds) to preserve ambient street hearing
DEFAULT_ACTION_COOLDOWNS = {
    Action.STOP: 2.5,
    Action.SHIFT_LEFT: 4.0,
    Action.SHIFT_RIGHT: 4.0,
    Action.KEEP_HEADING: 7.0,
    Action.SLOW: 7.5,
    Action.UNCLEAR: 8.5,
}

def synthesize_earcon(rate=24000, kind="stop"):
    """Generate a crisp, click-free acoustic earcon (55ms stop double-tone or 40ms revocation tone)."""
    duration = 0.055 if kind == "stop" else 0.040
    n = max(1, round(rate * duration))
    t = np.arange(n, dtype=np.float32) / rate
    window = np.hanning(n).astype(np.float32)
    if kind == "stop":
        signal = 0.55 * np.sin(2 * np.pi * 880.0 * t) + 0.45 * np.sin(2 * np.pi * 1320.0 * t)
        amp = 14000.0
    elif kind == "revoke":
        # Descending soft alert indicating previous directional prompt was cancelled
        freq = np.linspace(660.0, 420.0, n, dtype=np.float32)
        signal = np.sin(2 * np.pi * np.cumsum(freq) / rate)
        amp = 10000.0
    else:
        signal = 0.60 * np.sin(2 * np.pi * 520.0 * t)
        amp = 11000.0
    tone = (signal * window * amp).astype("<i2")
    gap = np.zeros(round(rate * 0.015), dtype="<i2")
    return np.concatenate([tone, gap])

def decision_pan(decision):
    """Map directional guidance or lateral hazard hints to stereo pan in [-0.75, +0.75]."""
    if decision.action == Action.SHIFT_LEFT:
        return -0.55
    if decision.action == Action.SHIFT_RIGHT:
        return 0.55
    hint = getattr(decision, "hazard_hint", "") or ""
    if "bên trái" in hint or "10 giờ" in hint or "11 giờ" in hint:
        return -0.65
    if "bên phải" in hint or "1 giờ" in hint or "2 giờ" in hint:
        return 0.65
    return 0.0

def pan_stereo(samples_mono, pan=0.0):
    """Constant-power stereo panning from mono PCM16 to interleaved (N, 2) PCM16."""
    mono = np.asarray(samples_mono, dtype=np.float32).reshape(-1)
    angle = (float(np.clip(pan, -0.85, 0.85)) + 1.0) * (np.pi / 4.0)
    left = np.clip(mono * np.cos(angle), -32768, 32767).astype("<i2")
    right = np.clip(mono * np.sin(angle), -32768, 32767).astype("<i2")
    return np.column_stack([left, right])

class ClipBank:
    def __init__(self,directory,sample_rate=24000):
        self.rate=sample_rate
        self.clips={}
        for action,name in CLIPS.items():
            with wave.open(str(Path(directory)/name),"rb") as f:
                if f.getsampwidth()!=2: raise ValueError("Voice clips must be PCM16 WAV.")
                audio=np.frombuffer(f.readframes(f.getnframes()),dtype="<i2").astype(np.float32)
                audio=audio.reshape(-1,f.getnchannels()).mean(1)
                if f.getframerate()!=self.rate:
                    audio=np.interp(np.arange(round(len(audio)*self.rate/f.getframerate()))*f.getframerate()/self.rate,
                                    np.arange(len(audio)),audio)
                self.clips[action]=audio.astype("<i2")
        self.extra={}
        for reason, name in {
            "narrow_corridor": "narrow",
            "forward_corridor_blocked": "forward_corridor_blocked",
            "transverse_step_ahead": "transverse_step_ahead",
            "scan_for_bypass": "scan_for_bypass",
            "bypass_left": "bypass_left",
            "bypass_right": "bypass_right",
        }.items():
            path = Path(directory) / (name + ".wav")
            if path.is_file():
                with wave.open(str(path), "rb") as f:
                    if f.getsampwidth()!=2 or f.getframerate()!=self.rate or f.getnchannels()!=1:
                        raise ValueError(f"{path.name} must be mono PCM16 at bank sample rate")
                    self.extra[reason] = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")

    def samples(self,decision):
        return self.extra.get(decision.reason,self.clips[decision.action])

class AudioTimeline:
    def __init__(
        self,
        bank,
        repeat_s=4.,
        sink=None,
        *,
        preempt_emergency=False,
        smart_arbitration=False,
        post_stop_hold_s=.8,
        post_stop_turn_hold_s=.30,
        tail_grace_s=.15,
        revocation_debounce_frames=2,
        action_cooldowns=None,
        earcon=False,
        stereo=False,
    ):
        self.bank=bank
        self.repeat_s=repeat_s
        self.sink=sink # Optional submit(samples, rate) / stop(); only live sinks log actual submission.
        self.smart_arbitration=bool(smart_arbitration)
        self.preempt_emergency=bool(preempt_emergency)
        self.post_stop_hold_s=float(post_stop_hold_s) if self.smart_arbitration else 0.
        self.post_stop_turn_hold_s=float(min(post_stop_turn_hold_s,post_stop_hold_s)) if self.smart_arbitration else 0.
        self.action_cooldowns=dict(DEFAULT_ACTION_COOLDOWNS if action_cooldowns is None else action_cooldowns)
        self.earcon=bool(earcon)
        self.stereo=bool(stereo)
        self.active=None
        self.pending=None
        self.pending_since=None
        self.latest=None
        self.segments=[]
        self.events=[]
        self.last_started={}
        self.last_side_started=None
        self.last_directional_started=None
        self.last_stop_ended_at=float("-inf")
        self.spoken_direction = None
        self.enabled=True

    def direction_in_effect(self, now):
        direction = self.spoken_direction
        if direction is None or not self.enabled:
            return None
        still_speaking = (
            self.active is not None
            and self.active["decision"] is direction
            and now < self.active["end"]
        )
        if still_speaking or now <= direction.valid_until:
            return direction
        return None

    def _log(self,event,now,decision,**extra):
        self.events.append(dict(event=event,time_s=now,action=decision.action.value,
                                evidence=decision.record(),**extra))

    def _cancel(self,now,why):
        if self.active:
            segment=self.active
            segment["end"]=min(segment["end"],now)
            if segment["decision"].action == Action.STOP:
                self.last_stop_ended_at = segment["end"]
            self._log("cancelled",now,segment["decision"],reason=why)
            if self.sink: self.sink.stop()
            self.active=None

    def stop(self,now):
        self._cancel(now,"stopped_or_muted")
        self.pending=None
        self.pending_since=None
        self.spoken_direction = None
        self.enabled=False

    def _finish_completed(self, now):
        if self.active is None or now < self.active["end"]:
            return
        finished_seg = self.active
        if finished_seg["decision"].action == Action.STOP:
            self.last_stop_ended_at = finished_seg["end"]
        self._log("finished", finished_seg["end"], finished_seg["decision"])
        self.active = None

    def _queue(self, decision, now):
        if self.pending is None or self.pending.action != decision.action:
            self.pending_since = now
        self.pending = decision

    def _clear_pending(self):
        self.pending = None
        self.pending_since = None

    def cooldown_for(self, action):
        if self.smart_arbitration and self.repeat_s > 0:
            return self.action_cooldowns.get(action, self.repeat_s)
        return self.repeat_s

    def _in_post_stop_hold(self, action, now):
        if not self.smart_arbitration or action == Action.STOP:
            return False
        hold_s = self.post_stop_turn_hold_s if action in SIDE_GUIDANCE else self.post_stop_hold_s
        if hold_s <= 0:
            return False
        return (now - self.last_stop_ended_at) < hold_s - 1e-9

    def _can_start(self, decision, now):
        action = decision.action
        if self._in_post_stop_hold(action, now):
            return False
        if decision.reason == "scan_for_bypass":
            return now - self.last_started.get("scan_for_bypass", float("-inf")) >= 8.5
        if action in SIDE_GUIDANCE and action != self.last_side_started:
            return True
        if self.smart_arbitration and action in DIRECTIONAL and action != self.last_directional_started:
            return True
        return now - self.last_started.get(action, float("-inf")) >= self.cooldown_for(action)

    def _prepare_samples(self, decision, earcon_kind=None):
        samples = self.bank.samples(decision)
        if self.earcon:
            if earcon_kind is not None:
                samples = np.concatenate([synthesize_earcon(self.bank.rate, earcon_kind), samples])
            elif decision.action == Action.STOP:
                samples = np.concatenate([synthesize_earcon(self.bank.rate, "stop"), samples])
        return samples

    def _start(self,decision,now,queued_at=None,earcon_kind=None):
        if not self.enabled or decision.valid_until<now:
            self._log("expired",now,decision)
            return
        samples=self._prepare_samples(decision,earcon_kind=earcon_kind)
        pan=decision_pan(decision)
        self.active={"start":now,"end":now+len(samples)/self.bank.rate,"decision":decision,
                     "samples":samples,"pan":pan}
        self.segments.append(self.active)
        self.last_started[decision.action]=now
        if decision.reason == "scan_for_bypass":
            self.last_started["scan_for_bypass"] = now
        if decision.action in SIDE_GUIDANCE:
            self.last_side_started = decision.action
        if decision.action in DIRECTIONAL:
            self.last_directional_started = decision.action
        elif decision.action == Action.STOP:
            self.last_side_started = None
            self.last_directional_started = None
        event="scheduled"
        submission_ms=None
        if self.sink:
            import time
            t=time.perf_counter()
            out_samples=pan_stereo(samples,pan).reshape(-1) if self.stereo else samples
            self.sink.submit(out_samples,self.bank.rate)
            submission_ms=(time.perf_counter()-t)*1000
            event="submitted_to_device"
        if decision.action in DIRECTIONAL:
            self.spoken_direction = decision
        elif decision.action == Action.STOP:
            self.spoken_direction = None
        self._log(event,now,decision,queue_wait_s=max(0,now-(now if queued_at is None else queued_at)),
                  submission_ms=submission_ms,pan=pan)

    def update(self, decision, now):
        if not self.enabled:
            return
        if decision.valid_until < now:
            self._log("expired", now, decision)
            self.tick(now)
            return

        self._finish_completed(now)
        previous = self.latest
        if decision.action == Action.STOP and (previous is None or previous.action != Action.STOP):
            # Coalesce observations of the same alarm. A new alarm is eligible immediately.
            self.last_started.pop(Action.STOP, None)
        if previous is None or previous.action != decision.action:
            self._log("requested", now, decision)
        self.latest = decision

        blocked_spoken_route = (
            decision.action == Action.STOP and decision.reason == "spoken_route_blocked"
        )
        if blocked_spoken_route:
            self.spoken_direction = None
        if decision.action == Action.STOP and self.active is not None:
            active_action = self.active["decision"].action
            revoke = self.smart_arbitration and blocked_spoken_route and active_action in DIRECTIONAL
            legacy_stop = (
                not self.smart_arbitration and self.preempt_emergency
                and active_action != Action.STOP and self._can_start(decision, now)
            )
            if revoke or legacy_stop:
                reason = "preempted_revoked_direction" if revoke else "preempted_by_emergency_stop"
                self._cancel(now, reason)
                if self.pending is not None:
                    self._log("pending_replaced", now, self.pending)
                    self._clear_pending()
                self._start(decision, now, now, earcon_kind="stop")
                return

        # Keep just one pending utterance. A newer decision can revoke a queued route.
        if self.pending:
            pending = self.pending
            expired = pending.valid_until < now
            revoked = pending.frame_key[0] != decision.frame_key[0] or (
                pending.action in DIRECTIONAL
                and (pending.action != decision.action or decision.invalidates_route)
            )
            priorities = ARBITRATION_TIER if self.smart_arbitration else SPEECH_PRIORITY
            keep_pending = priorities.get(pending.action, 1) > priorities.get(decision.action, 1)
            if not self.smart_arbitration and pending.action == Action.STOP and decision.action != Action.STOP:
                keep_pending = True
            if expired or revoked:
                self._log("expired" if expired else "pending_replaced", now, pending)
                self._clear_pending()
            elif keep_pending:
                self.tick(now)
                return
            elif pending.action != decision.action:
                self._log("pending_replaced", now, self.pending)
                self._clear_pending()

        # In smart_arbitration, hold off queuing transient 0.4s pre-confirmation SLOW chatter so the
        # upcoming confirmed direction (SHIFT_LEFT/RIGHT or KEEP_HEADING) gets an idle speaker immediately.
        if (self.smart_arbitration and decision.action == Action.SLOW
                and decision.reason in {"confirming_corridor", "waiting_context"}):
            self.tick(now)
            return

        if self.active is None or self.active["decision"].action != decision.action:
            self._queue(decision, now)
        self.tick(now)

    def tick(self, now):
        self._finish_completed(now)
        if self.pending and now > self.pending.valid_until:
            self._log("expired", now, self.pending)
            self._clear_pending()
        if not self.active and self.pending and self._can_start(self.pending, now):
            decision, queued_at = self.pending, self.pending_since
            self._clear_pending()
            self._start(decision, now, queued_at)

    def finish_export(self, video_end):
        """Close the current phrase and drop queued commands at end of footage."""
        self._clear_pending()
        end = max(video_end, self.active["end"]) if self.active else video_end
        self._finish_completed(end)
        self.enabled = False
        return end

    def save(self,path,duration):
        # Stream chunks to disk instead of allocating a full long-video WAV in RAM.
        channels = 2 if self.stereo else 1
        with wave.open(str(path),"wb") as out:
            out.setnchannels(channels);out.setsampwidth(2);out.setframerate(self.bank.rate)
            total=round(duration*self.bank.rate)
            for start in range(0,total,self.bank.rate):
                end=min(total,start+self.bank.rate)
                chunk=np.zeros((end-start,channels) if self.stereo else (end-start,),dtype="<i2")
                for segment in self.segments:
                    a=round(segment["start"]*self.bank.rate)
                    b=min(round(segment["end"]*self.bank.rate),a+len(segment["samples"]))
                    lo,hi=max(a,start),min(b,end)
                    if hi>lo:
                        seg_slice=segment["samples"][lo-a:hi-a]
                        chunk[lo-start:hi-start]=pan_stereo(seg_slice,segment.get("pan",0.)) if self.stereo else seg_slice
                out.writeframes(chunk.tobytes())
