"""BoT-SORT identity association with observed boxes and compensated velocity."""
from dataclasses import replace
from types import SimpleNamespace
import cv2
import numpy as np

class MotionTracker:
    def __init__(self, config):
        self.config=config
        self.previous=None
        self.history={}
        self.labels={}
        self.tracker=None

    def reset(self):
        if self.tracker is not None:
            self.tracker.reset()
        self.previous=None
        self.history.clear()

    def _create(self):
        from ultralytics.trackers.bot_sort import BOTSORT
        self.tracker=BOTSORT(SimpleNamespace(
            tracker_type="botsort", track_high_thresh=.25, track_low_thresh=.1,
            new_track_thresh=.25, track_buffer=30, match_thresh=.8, fuse_score=True,
            gmc_method="sparseOptFlow", proximity_thresh=.5, appearance_thresh=.8,
            with_reid=False, model="auto"))

    def update(self, scene):
        frame=scene.frame
        gray=cv2.cvtColor(frame.bgr,cv2.COLOR_BGR2GRAY)
        warp=np.eye(2,3,dtype=np.float32)
        invalid=False
        if self.previous:
            old,key,pts=self.previous
            dt=frame.pts-pts
            if key[0]!=frame.session or old.shape!=gray.shape or not 0<dt<=self.config.track_ttl_s:
                invalid=True
            else:
                # Reject cuts and strong motion; scale walking-gait tolerance smoothly with dt.
                diff=float(np.mean(cv2.absdiff(cv2.resize(old,(64,48)),cv2.resize(gray,(64,48)))))
                points=cv2.goodFeaturesToTrack(old,100,.02,8)
                track_old,track_new=old,gray
                if points is None or len(points)<12:
                    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
                    track_old,track_new=clahe.apply(old),clahe.apply(gray)
                    points=cv2.goodFeaturesToTrack(track_old,120,.004,6)
                if points is not None and len(points)>=8:
                    new,status,_=cv2.calcOpticalFlowPyrLK(track_old,track_new,points,None)
                    good=status.ravel()==1
                    affine,inliers=cv2.estimateAffinePartial2D(points[good],new[good],method=cv2.RANSAC) if good.sum()>=8 else (None,None)
                    if affine is None:
                        invalid=diff>16
                    else:
                        warp=affine.astype(np.float32)
                        angle=abs(np.degrees(np.arctan2(warp[1,0],warp[0,0])))
                        shift=np.linalg.norm(warp[:,2])/max(gray.shape)
                        max_angle=max(8.,min(18.,65.*dt))
                        max_shift=max(.12,min(.25,.90*dt))
                        invalid=angle>max_angle or shift>max_shift or float(inliers.mean())<.35
                else:
                    # Clean indoor painted walls/floors may lack corners; allow only when frame diff is calm.
                    invalid=diff>16
                invalid=invalid or diff>65
        if invalid:
            self.reset()
        if self.tracker is None:
            self._create()
        from ultralytics.engine.results import Boxes
        for obj in scene.detections:
            if obj.label not in self.labels:
                self.labels[obj.label]=len(self.labels)
        rows=np.array([list(d.box)+[d.confidence,self.labels[d.label]] for d in scene.detections],dtype=np.float32).reshape(-1,6)
        tracks=self.tracker.update(Boxes(rows,gray.shape).numpy(),frame.bgr)
        observed=list(scene.detections)
        new_history={}
        seen_ids=set()
        zoom=max(.5,min(2.,float(np.hypot(warp[0,0],warp[1,0]))))
        for row in tracks:
            index=int(row[-1])
            if not 0<=index<len(observed):
                continue
            obj=observed[index]
            track_id=int(row[4])
            seen_ids.add(track_id)
            center=np.array([(obj.box[0]+obj.box[2])/2,(obj.box[1]+obj.box[3])/2,1.])
            scale=float(np.sqrt(max(1.,obj.box[2]-obj.box[0])*max(1.,obj.box[3]-obj.box[1])))
            velocity=(0.,0.)
            scale_rate=0.
            ttc_s=None
            obs_count=1
            prior=self.history.get(track_id)
            if prior and not invalid:
                old_center,old_scale,prior_vel,prior_rs,prior_obs,pts=prior
                dt=frame.pts-pts
                if 0<dt<=self.config.track_ttl_s:
                    obs_count=prior_obs+1
                    raw_vel=(center[:2]-warp@old_center)/dt
                    vel_arr=.65*raw_vel+.35*np.asarray(prior_vel) if prior_obs>=2 else raw_vel
                    velocity=tuple(np.clip(vel_arr,-max(gray.shape),max(gray.shape)).tolist())
                    expected_scale=max(1.,old_scale*zoom)
                    raw_rs=(scale-expected_scale)/(expected_scale*dt)
                    scale_rate=float(np.clip(.6*raw_rs+.4*prior_rs if prior_obs>=2 else raw_rs,-5.,5.))
                    if (obs_count>=3 and scale_rate>.22) or (obs_count>=2 and scale_rate>.50):
                        ttc_s=round(1./scale_rate,3)
            observed[index]=replace(obj,track_id=track_id,velocity=velocity,scale_rate=scale_rate,ttc_s=ttc_s)
            new_history[track_id]=(center,scale,velocity,scale_rate,obs_count,frame.pts)
        # Retain short-term motion state across 1-frame detector flickers without emitting ghost boxes.
        if not invalid:
            for tid,prior in self.history.items():
                if tid in seen_ids:
                    continue
                old_center,old_scale,prior_vel,prior_rs,prior_obs,pts=prior
                if 0<frame.pts-pts<=self.config.track_ttl_s:
                    warped_xy=warp@old_center
                    new_history[tid]=(np.array([warped_xy[0],warped_xy[1],1.]),old_scale*zoom,prior_vel,prior_rs,prior_obs,pts)
        self.history=new_history
        self.previous=(gray,frame.key,frame.pts)
        scene.detections=observed
        scene.motion_invalid=invalid
        scene.warp=None if invalid else warp
        return scene

