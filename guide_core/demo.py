"""python -m guide_core.demo --config configs/guide_kaggle.json --video path.mp4 --output runs/guide_v2"""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import time
import uuid
import cv2
import numpy as np
from .config import Config
from .types import Frame, DIRECTIONAL
from .engine import GuideEngine
from .audio import AudioTimeline, ClipBank
from .render import render
from .replay import run_replay

def distribution(values):
    return {"count":len(values),"p50":float(np.percentile(values,50)) if values else None,
            "p95":float(np.percentile(values,95)) if values else None}

def fingerprint(path):
    path=Path(path)
    if path.is_file():
        files=[path]
    else:
        files=sorted({*path.glob("*.safetensors"),*path.glob("*.bin"),*path.glob("*.xml")})
    digests={}
    for item in files:
        digest=hashlib.sha256()
        with item.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""): digest.update(chunk)
        digests[item.name]=digest.hexdigest()
    return digests

def run(config,video,output,seconds=None,progress=True):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"decisions.jsonl").exists():
        raise FileExistsError("Choose a fresh output directory to preserve previous measurements: "+str(output))
    video=Path(video)
    if not video.is_file(): raise FileNotFoundError(video)
    config.validate()
    import os
    import torch
    import psutil
    # Default to 4 threads so hybrid Intel CPUs stay on fast P-cores instead of stalling on E-cores.
    torch.set_num_threads(max(1,int(os.environ.get("GUIDE_TORCH_THREADS","4"))))
    cv2.setNumThreads(1)
    process=psutil.Process()
    rss=[]
    def memory():
        rss.append(process.memory_info().rss/1024**2)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        for _dev in range(torch.cuda.device_count()):
            try:
                torch.cuda.reset_peak_memory_stats(_dev)
            except RuntimeError:
                pass
    engine=GuideEngine(config)
    load_start=time.perf_counter()
    try:
        engine.load()
    except Exception:
        engine.close()
        raise
    load_seconds=time.perf_counter()-load_start
    bank=ClipBank(config.audio_dir) if config.audio_enabled else None
    audio=AudioTimeline(
        bank,
        preempt_emergency=config.preempt_emergency_audio,
        smart_arbitration=config.smart_arbitration,
        post_stop_hold_s=config.post_stop_hold_s,
        post_stop_turn_hold_s=config.post_stop_turn_hold_s,
        tail_grace_s=config.tail_grace_s,
        revocation_debounce_frames=config.revocation_debounce_frames,
        earcon=config.earcon_enabled,
        stereo=config.stereo_audio,
    ) if bank else None
    (output/"config.json").write_text(json.dumps(config.as_dict(),ensure_ascii=False,indent=2),encoding="utf-8")
    records=[]
    pipeline_ms=[]
    stale_count=0
    frames_used=[]
    received_latency=[]
    counts=Counter()
    writer=None
    video_frames=0
    previous_image=None
    writer_fps=config.sample_fps
    origin=0.
    elapsed_start=time.perf_counter()
    stats={}
    deferred_renders=[]
    def write_until(image,t):
        nonlocal writer,video_frames,previous_image
        if writer is None:
            writer=cv2.VideoWriter(str(output/"preview_raw.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),writer_fps,(image.shape[1],image.shape[0]))
            if not writer.isOpened(): raise RuntimeError("Video encoder unavailable")
        target=math.floor(t*writer_fps+1e-6)
        while video_frames<target:
            writer.write(previous_image if previous_image is not None else image)
            video_frames+=1
        previous_image=image

    def emit(scene,decision,results,at,stage):
        nonlocal stale_count
        memory()
        if config.mode=="realtime_replay":
            now=time.perf_counter()
            at=now-origin
            if now>decision.valid_until:
                decision=engine.policy.stale(scene.frame,now,"expired_before_output")
                stage="stale_before_output"
            decision=replace(decision,clock=decision.clock-origin,valid_until=decision.valid_until-origin)
            latency=(now-scene.frame.received_at)*1000
            received_latency.append(latency)
        else:
            latency=None
        if "stale" in stage: stale_count+=1
        if audio: audio.update(decision,at)
        record=decision.record()
        record.update(emitted_at_s=at,video_time_s=scene.frame.pts,stage=stage,
                      geometry=scene.frame.transform,received_at_monotonic=scene.frame.received_at,
                      frame_to_decision_ms=latency,errors=scene.errors,
                      models=[{"id":r.model_id,"role":r.role,"frame_key":list(r.frame_key),
                               "started_at_monotonic":r.started_at,"finished_at_monotonic":r.finished_at,
                               "ms":(r.finished_at-r.started_at)*1000,"error":r.error,
                               "skipped":r.skipped} for r in results],
                      detections=[{"label":d.label,"confidence":d.confidence,"box":d.box,
                                   "track_id":d.track_id,"sources":d.sources,"velocity":d.velocity,
                                   "ttc_s":d.ttc_s,"zone":d.zone} for d in scene.detections])
        records.append(record)
        counts[decision.action.value]+=1
        frames_used.append(scene.frame.sequence)
        if config.mode=="realtime_replay" and len(deferred_renders)<600:
            deferred_renders.append((scene,decision,at,stage))
        else:
            write_until(render(scene,decision,stage),at)
        if progress and len(records)%10==0:
            print(f"Processed {len(records)} outputs | video {scene.frame.pts:.1f}s | {decision.action.value}",flush=True)
    try:
        if config.mode=="offline":
            cap=cv2.VideoCapture(str(video))
            if not cap.isOpened(): raise RuntimeError("Cannot open video")
            fps=cap.get(cv2.CAP_PROP_FPS) or 30.
            stride=max(1,math.ceil(fps/config.sample_fps))
            writer_fps=fps/stride
            seq=0
            decode_seconds=0.
            decoded=0
            last_pts=0.
            session="offline-"+uuid.uuid4().hex[:12]
            try:
                while True:
                    pts=seq/fps
                    if seconds is not None and pts>=seconds: break
                    t=time.perf_counter()
                    ok,image=cap.read()
                    decode_seconds+=time.perf_counter()-t
                    if not ok: break
                    decoded+=1
                    if seq%stride==0:
                        frame=Frame.prepare(image,session,seq,pts,config.max_side,config.mirror_input)
                        t=time.perf_counter()
                        scene,decision,results=engine.process(frame)
                        pipeline_ms.append((time.perf_counter()-t)*1000)
                        emit(scene,decision,results,pts,"offline_fused")
                        last_pts=pts
                    seq+=1
            finally:
                cap.release()
            duration=decoded/fps
            stats={"decoded_frames":decoded,"decode_seconds":decode_seconds,
                   "sample_stride":stride,"sampled_frames":len(frames_used),"dropped_frames":0}
        else:
            origin=time.perf_counter()
            stats=run_replay(engine,video,emit,lambda t: audio.tick(t) if audio else None,seconds,start=origin)
            duration=stats["elapsed_s"]
        if not frames_used: raise RuntimeError("Video produced no usable frames")
        active_elapsed = duration if config.mode=="realtime_replay" else None
        for d_scene,d_dec,d_at,d_stage in deferred_renders:
            write_until(render(d_scene,d_dec,d_stage),d_at)
        deferred_renders.clear()
        export_duration = audio.finish_export(duration) if audio else duration
        if previous_image is not None:
            while video_frames<math.ceil(export_duration*writer_fps):
                writer.write(previous_image);video_frames+=1
        if writer:
            writer.release();writer=None
        if audio:
            audio.save(output/"voice.wav",export_duration)
        elapsed=active_elapsed if active_elapsed is not None else (time.perf_counter()-elapsed_start)
        with (output/"decisions.jsonl").open("w",encoding="utf-8") as f:
            for record in records: f.write(json.dumps(record,ensure_ascii=False)+"\n")
        with (output/"audio_events.jsonl").open("w",encoding="utf-8") as f:
            for record in audio.events if audio else []: f.write(json.dumps(record,ensure_ascii=False)+"\n")
        (output/"speech_segments.json").write_text(json.dumps([
            {"start":s["start"],"end":s["end"],"action":s["decision"].action.value,
             "pan":round(float(s.get("pan",0.0)),3),"evidence":s["decision"].record()} for s in audio.segments] if audio else [],
             ensure_ascii=False,indent=2),encoding="utf-8")
        import imageio_ffmpeg
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-y","-loglevel","error","-i",str(output/"preview_raw.mp4")]
        if audio: cmd+=["-i",str(output/"voice.wav")]
        cmd+=["-c:v","libx264","-pix_fmt","yuv420p","-preset","fast","-movflags","+faststart"]
        if audio: cmd+=["-c:a","aac","-shortest"]
        cmd+=[str(output/"demo.mp4")]
        subprocess.run(cmd,check=True,capture_output=True)
        versions={}
        for name in ("torch","ultralytics","transformers","opencv-python","numpy","imageio-ffmpeg"):
            versions[name]=importlib.metadata.version(name)
        cuda_alloc_by_dev={}
        cuda_res_vals=[]
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            for i in range(torch.cuda.device_count()):
                try:
                    cuda_alloc_by_dev[f"cuda:{i}"]=round(torch.cuda.max_memory_allocated(i)/1024**2,2)
                    cuda_res_vals.append(torch.cuda.max_memory_reserved(i)/1024**2)
                except RuntimeError:
                    pass
        report={
            "status":"unlabelled_smoke_or_demo_not_acceptance","mode":config.mode,
            "video":str(video),"video_sha256":fingerprint(video),"video_duration_s":duration,
            "export_duration_s":export_duration,
            "source_video_duration_s":stats.get("source_duration_s",duration),
            "frame_sequences":sorted(set(frames_used)),"load_seconds":load_seconds,"processing_wall_s":elapsed,
            "pipeline_outputs_per_second":len(records)/max(elapsed,1e-6),
            "unique_inference_frames_per_second":len(set(frames_used))/max(elapsed,1e-6),
            "decode_fps":stats["decoded_frames"]/max(stats["decode_seconds"],1e-6),
            "pipeline_ms":distribution(pipeline_ms),"frame_to_decision_ms":distribution(received_latency),
            "model_ms":{k:distribution(v) for k,v in engine.timings.items()},
            "model_fps":{k:1000/np.mean(v) if v and np.mean(v)>0 else None for k,v in engine.timings.items()},
            "rss_peak_sampled_mb":max(rss,default=0),
            "cuda_peak_allocated_mb":max(cuda_alloc_by_dev.values(),default=None),
            "cuda_peak_reserved_mb":max(cuda_res_vals,default=None),
            "cuda_peak_allocated_by_device_mb":cuda_alloc_by_dev,
            "device_names":[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "versions":versions,"model_sha256":{m.id:fingerprint(m.path) for m in config.models if m.enabled},
            "load_failures":engine.failures,"action_counts":dict(counts),"source":stats,"stale_outputs":stale_count,
            "audio":{"mode":"scheduled_in_export_not_hardware_playback" if audio else "disabled",
                     "queue_wait_ms":distribution([e["queue_wait_s"]*1000 for e in audio.events if e["event"]=="scheduled"] if audio else []),
                     "started_by_action":dict(Counter(e["action"] for e in audio.events if e["event"]=="scheduled")) if audio else {},
                     "interruptions_by_reason":dict(Counter(e["reason"] for e in audio.events
                                                           if e["event"]=="cancelled" and e["reason"]!="stopped_or_muted")) if audio else {},
                     "device_submission_ms":None,"idle_emergency_capture_to_device_p95_ms":None},
            "acceptance":{"evaluated":False,"reason":"Requires independent event-labelled videos, >=100 hazards and >=30 clear minutes."}
        }
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        return report
    finally:
        if writer: writer.release()
        engine.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",required=True);p.add_argument("--video",required=True)
    p.add_argument("--output",required=True);p.add_argument("--seconds",type=float)
    p.add_argument("--mode",choices=["offline","realtime_replay"])
    p.add_argument("--environment",choices=["auto","alley","sidewalk","indoor"])
    p.add_argument("--no-audio",action="store_true")
    args=p.parse_args()
    config=Config.read(args.config)
    if args.mode: config.mode=args.mode
    if args.environment: config.environment=args.environment
    if args.no_audio: config.audio_enabled=False
    if args.seconds is not None and args.seconds<=0: p.error("--seconds must be positive")
    report=run(config,args.video,args.output,args.seconds)
    print(json.dumps({"output":args.output,"fps":report["unique_inference_frames_per_second"],"failures":report["load_failures"]},ensure_ascii=False))
if __name__=="__main__": main()

