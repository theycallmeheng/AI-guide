"""Bounded, observed-only IoU tracking for obstacle alerts."""
from dataclasses import replace


def iou(a, b):
    overlap = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = lambda x: max(0, x[2]-x[0])*max(0, x[3]-x[1])
    return overlap / max(1, area(a)+area(b)-overlap)


def category(name):
    name = name.lower().replace('_', ' ')
    return {'bike': 'bicycle', 'stair': 'stairs', 'steps': 'stairs'}.get(name, name)


class TemporalTracker:
    def __init__(self, ttl=.5, confirmation_window=.3, stable_seconds=.12, emergency_confidence=.8):
        self.ttl, self.window, self.stable = ttl, confirmation_window, stable_seconds
        self.emergency_confidence = emergency_confidence
        self.tracks = {}
        self.next_id = 1

    def update(self, analysis, now):
        self.tracks = {k: v for k, v in self.tracks.items() if 0 <= now-v['time'] <= self.ttl}
        used, visible = set(), []
        width = analysis.frame.shape[1]
        for obj in sorted(analysis.obstacles, key=lambda x: -x.confidence):
            matches = [(iou(obj.box, state['object'].box), key) for key, state in self.tracks.items()
                       if key not in used and category(obj.class_name) == category(state['object'].class_name)]
            score, key = max(matches, default=(0, 0))
            if score < .3:
                key = self.next_id
                self.next_id += 1
                state = dict(object=obj, time=now, observations=[], candidate=None, since=now)
                self.tracks[key] = state
            else:
                state = self.tracks[key]
            used.add(key)
            emergency = (obj.distance_band == 'gần' and obj.box[0] < width*.62 and
                         obj.box[2] > width*.38 and obj.confidence >= self.emergency_confidence)
            candidate = (obj.direction, obj.distance_band)
            previous = state['object']
            if candidate != state['candidate']:
                state['candidate'], state['since'] = candidate, now
            direction, band = candidate
            if not emergency and now-state['since'] < self.stable:
                direction, band = previous.direction, previous.distance_band
            box = tuple(round(.65*a+.35*b) for a, b in zip(obj.box, previous.box))
            fresh = replace(obj, box=obj.box if emergency else box, direction=direction,
                            distance_band=band, track_id=key, observed_at=now)
            state['observations'] = [t for t in state['observations'] if 0 <= now-t <= self.window] + [now]
            state.update(object=fresh, time=now)
            if emergency or len(state['observations']) >= 2:
                visible.append(fresh)
        return replace(analysis, obstacles=tuple(visible), captured_at=now)
