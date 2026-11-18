"""Score emitted audio alerts against explicitly annotated video intervals."""
import argparse
import json
import math
from pathlib import Path


def evaluate(truth, alerts):
    duration = truth['duration_s']
    assert math.isfinite(duration) and duration > 0
    events, clear = truth['events'], truth['clear_intervals']
    ids = [e['id'] for e in events]
    assert len(ids) == len(set(ids)), 'Duplicate event IDs'
    for e in events:
        assert 0 <= e['start_s'] < e['end_s'] <= duration
        assert e['action'] in ('stop', 'slow_down', 'unclear')
    for i, interval in enumerate(clear):
        a, b = interval
        assert 0 <= a < b <= duration
        assert all(b <= c or a >= d for c, d in clear[:i]), 'Overlapping clear intervals'
        assert all(b <= e['start_s'] or a >= e['end_s'] for e in events), 'Clear interval overlaps hazard'
    hits, duplicates, false = {}, 0, 0
    for alert in sorted(alerts, key=lambda a: a['time_s']):
        t = alert['time_s']
        assert math.isfinite(t) and 0 <= t <= duration
        matches = [e for e in events if e['start_s'] <= t <= e['end_s']
                   and alert['action'] == e['action'] and alert.get('direction') == e.get('direction')
                   and alert.get('class_name') == e.get('class_name')]
        fresh = [e for e in matches if e['id'] not in hits]
        if fresh:
            event = min(fresh, key=lambda e: e['end_s'])
            hits[event['id']] = t-event['start_s']
        elif matches:
            duplicates += 1
        elif any(a <= t <= b for a, b in clear):
            false += 1
    clear_minutes = sum(b-a for a,b in clear)/60
    recall = len(hits)/len(events) if events else None
    return dict(event_count=len(events), detected_events=len(hits), recall=recall,
                false_alerts=false, clear_minutes=clear_minutes,
                false_alerts_per_minute=false/clear_minutes if clear_minutes else None,
                repeated_alerts=duplicates, first_alert_delays_s=hits,
                preliminary=len(events)<100 or clear_minutes<30,
                note='Event timing uses emitted audio timestamps aligned to video, not queued requests. Repeats may include intended cooldown reminders.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--truth', required=True)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--alerts')
    group.add_argument('--session', help='Runtime session report with audio_events')
    p.add_argument('--video-offset', type=float, help='Session clock time corresponding to video time zero; must be measured')
    args=p.parse_args()
    if args.session:
        if args.video_offset is None: p.error('--session requires measured --video-offset')
        session=json.loads(Path(args.session).read_text(encoding='utf-8'))
        alerts=[]
        for event in session['audio_events']:
            parts=(event.get('key') or '').split('|')
            action='stop' if event['priority'] >= 3 else 'slow_down'
            if event.get('key') == 'unclear': action='unclear'
            t=event['time_s']-args.video_offset
            if t < 0: continue
            alerts.append(dict(time_s=t,action=action,class_name=parts[0] if len(parts)==3 else None,
                               direction=parts[2] if len(parts)==3 else None))
    else:
        alerts=json.loads(Path(args.alerts).read_text(encoding='utf-8'))
    print(json.dumps(evaluate(json.loads(Path(args.truth).read_text(encoding='utf-8')),
                              alerts), ensure_ascii=False, indent=2))
