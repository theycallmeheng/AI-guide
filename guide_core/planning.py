"""Conservative image-space topology with 3D ground-plane residuals and multi-horizon DP."""
import cv2
import numpy as np
from .types import Action, Route
from .fusion import is_hazard

GROUND={
    "sidewalk":{"sidewalk","walkway","footpath","pavement","path","pedestrian walkway"},
    "alley":{"sidewalk","walkway","footpath","pavement","path","dirt track","road"},
    "indoor":{"floor","flooring","rug"},
}
STRUCTURES={"wall","fence","door","stairs","stairway","escalator","step","building","railing",
            "bannister","person","car","bus","truck","van","minibike","bicycle","tree",
            "column","pole","streetlight","cabinet","chair","table","bench","box","barrel",
            "ashcan","water","river","swimming pool"}
STAIR_NAMES={"stairs","stairway","escalator","step"}
VEHICLES={"car","bus","truck","motorcycle","bicycle","van","minibike"}
VI_HINTS={
    "person":"người phía trước","bicycle":"xe đạp","motorcycle":"xe máy","car":"ô tô",
    "bus":"xe buýt","truck":"xe tải","dog":"chó phía trước","stairs":"cầu thang",
    "step":"bậc thềm","pole":"cột chắn","barrier":"rào chắn","cone":"cọc tiêu",
    "tree":"cây phía trước","chair":"ghế chắn lối","bench":"ghế dài","manhole":"miệng cống",
    "pothole":"hố mặt đường","dustbin":"thùng rác","plant pot":"chậu cây",
}

def _vi_label(label):
    return VI_HINTS.get(str(label).lower(), "vật cản")

def obstacle_mask(scene, horizon=.5):
    h,w=scene.frame.bgr.shape[:2]
    mask=np.zeros((h,w),np.uint8)
    for obj in scene.detections:
        if not is_hazard(obj): continue
        x1,y1,x2,y2=obj.box
        dx,dy=np.array(obj.velocity)*horizon
        # Expand both along translational velocity and radial looming expansion.
        grow=.5*min(.5,max(0.,getattr(obj,"scale_rate",0.))*horizon)
        gx,gy=(x2-x1)*grow,(y2-y1)*grow
        box=(min(x1,x1+dx)-gx,min(y1,y1+dy)-gy,max(x2,x2+dx)+gx,max(y2,y2+dy)+gy)
        pad=max(2,round(w*.015))
        a,b=max(0,int(box[0])-pad),max(0,int(box[1])-pad)
        c,d=min(w,int(box[2])+pad+1),min(h,int(box[3])+pad+1)
        mask[b:d,a:c]=1
    anomalies=getattr(scene,"depth_anomalies",None)
    if anomalies is not None:
        pos=anomalies.get("positive")
        drop=anomalies.get("dropoff")
        if pos is not None and pos.shape==(h,w):
            mask[pos]=1
        if drop is not None and drop.shape==(h,w):
            mask[drop]=1
    return mask.astype(bool)

def analyze_depth_geometry(near, valid, ground_hint=None):
    """Fit a ground plane in relative inverse-depth space to detect class-agnostic obstacles & drop-offs."""
    h,w=near.shape
    gh_d,gw_d=min(60,max(16,h//4)),min(80,max(16,w//4))
    near_f=near.astype(np.float32)
    near_s=cv2.resize(near_f,(gw_d,gh_d),interpolation=cv2.INTER_AREA)
    valid_s=cv2.resize(valid.astype(np.float32),(gw_d,gh_d),interpolation=cv2.INTER_AREA)>=.95
    v_s=(np.arange(gh_d,dtype=np.float32)[:,None]+.5)/gh_d
    u_s=(np.arange(gw_d,dtype=np.float32)[None,:]+.5)/gw_d-.5

    cand=np.zeros_like(valid_s)
    if ground_hint is not None and ground_hint.shape==(h,w):
        hint_s=cv2.resize(ground_hint.astype(np.float32),(gw_d,gh_d),interpolation=cv2.INTER_AREA)>=.5
        cand=hint_s & valid_s & (v_s>=.45) & (v_s<=.96)
    if int(cand.sum())<40:
        cand=valid_s & (v_s>=.55) & (v_s<=.96) & (np.abs(u_s)<=.40)

    ys,xs=np.nonzero(cand)
    best_params=None
    best_inliers=None
    if len(ys)>=24:
        A=np.column_stack([v_s[ys,0],u_s[0,xs],np.ones(len(ys),dtype=np.float32)])
        z=near_s[ys,xs]
        # Candidate 0: direct least-squares (exact for clean planar walkways), followed by RANSAC triplets.
        proposals=[]
        try:
            ols,*_=np.linalg.lstsq(A,z,rcond=None)
            proposals.append(ols)
        except np.linalg.LinAlgError:
            pass
        rng=np.random.default_rng(42)
        for _ in range(28):
            idx=rng.choice(len(ys),size=3,replace=False)
            sub=A[idx]
            if abs(float(np.linalg.det(sub)))<1e-4: continue
            proposals.append(np.linalg.solve(sub,z[idx]))
        best_count=0
        for params in proposals:
            if not (.14<=params[0]<=2.5 and abs(params[1])<=.22): continue
            inliers=np.abs(A@params-z)<.085
            cnt=int(inliers.sum())
            if cnt>best_count:
                best_count,best_inliers=cnt,inliers
        if best_inliers is not None and best_count>=max(24,int(.32*len(z))):
            refined,*_=np.linalg.lstsq(A[best_inliers],z[best_inliers],rcond=None)
            if .12<=refined[0]<=2.6 and abs(refined[1])<=.25:
                best_params=refined

    dx=cv2.Sobel(near_f,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(near_f,cv2.CV_32F,0,1,ksize=3)
    grad=np.hypot(dx,dy)
    kernel2=np.ones((2,2),np.uint8)

    if best_params is not None:
        plane_s=np.clip(best_params[0]*v_s+best_params[1]*u_s+best_params[2],0.,1.)
        res_s=near_s-plane_s
        v_pos_min=.22 if h>w else .42
        pos_s=cv2.morphologyEx(((res_s>.18)&(v_s>=v_pos_min)&valid_s).astype(np.uint8),cv2.MORPH_OPEN,kernel2)
        dy_s=cv2.Sobel(near_s,cv2.CV_32F,0,1,ksize=3)
        below_cliff=np.maximum.accumulate((dy_s<-.12)&(v_s>=.38),axis=0)
        drop_raw=(res_s<-.18)&(near_s<.72)&(v_s>=.52)&valid_s&below_cliff
        drop_s=cv2.morphologyEx(drop_raw.astype(np.uint8),cv2.MORPH_OPEN,kernel2)
        pos_full=cv2.resize(pos_s,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        drop_full=cv2.resize(drop_s,(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        res_full=cv2.resize(res_s,(w,h),interpolation=cv2.INTER_LINEAR)
        cliff=((grad>.95)&(np.abs(res_full)>.11)).astype(np.uint8)
        uncertain=cv2.dilate(cliff,np.ones((3,3),np.uint8)).astype(bool)
        return {"positive":pos_full,"dropoff":drop_full,"uncertain":uncertain,"plane_valid":True,
                "residual":res_full,"grad":grad,"plane_slope":float(best_params[0])}

    pos_full=np.zeros((h,w),dtype=bool)
    c_patch=near_s[int(gh_d*.55):int(gh_d*.95),int(gw_d*.35):int(gw_d*.65)]
    if c_patch.size and float(np.median(c_patch))>=.83 and float(np.std(c_patch))<.12:
        pos_full[int(h*.55):int(h*.95),int(w*.35):int(w*.65)]=True
    edges=(grad>.75).astype(np.uint8)
    uncertain=cv2.dilate(edges,np.ones((3,3),np.uint8)).astype(bool)
    return {"positive":pos_full,"dropoff":np.zeros((h,w),dtype=bool),"uncertain":uncertain,"plane_valid":False,
            "residual":None,"grad":grad,"plane_slope":0.}

def urgent_reason(scene, ttc_stop_s=1.6, head_enabled=True, depth_enabled=True):
    h,w=scene.frame.bgr.shape[:2]
    depth_near=None
    if scene.depth is not None and scene.depth.get("valid") is not None:
        valid=scene.depth["valid"]
        near=scene.depth["relative_nearness"]
        if valid.shape==(h,w) and near.shape==(h,w) and np.isfinite(near).all() and valid.mean()>=.95:
            depth_near=near

    for obj in scene.detections:
        if not is_hazard(obj): continue
        x1,y1,x2,y2=obj.box
        # 1. Near ground/body obstacle in central corridor.
        if x1<w*.62 and x2>w*.38 and y2>h*.80 and (y2-y1>h*.18 or y2>h*.92):
            return True,"near_center_obstacle",_vi_label(obj.label)
        # 1b. Wide transverse stairs / step / barrier blocking forward path.
        if obj.label in {"stairs","step","barrier"} and (x2-x1)>=w*.55 and x1<w*.45 and x2>w*.55:
            if y2>=h*(.52 if h>w else .68):
                return True,"near_stairs_hazard",f"{_vi_label(obj.label)} phía trước"
        # 2. Rapidly approaching obstacle via Looming Time-To-Collision (TTC).
        ttc=getattr(obj,"ttc_s",None)
        if ttc is not None and 0<ttc<=ttc_stop_s and obj.confidence>=.45 and y2>h*.48:
            vx=obj.velocity[0]*min(ttc,1.)
            if min(x1,x1+vx)<w*.65 and max(x2,x2+vx)>w*.35:
                return True,"approaching_obstacle_ttc",f"{_vi_label(obj.label)} đang tiến gần"
        # 3. Overhanging / head-to-chest hazard fused with depth proximity.
        if head_enabled and getattr(obj,"zone","ground")=="head" and x1<w*.60 and x2>w*.40 and y1<h*.48 and h*.28<=y2<=h*.80:
            big_box=(y2-y1)>=h*.20 or (x2-x1)>=w*.28
            if big_box and depth_near is not None:
                patch=depth_near[max(0,int(y1)):min(h,int(y2)),max(0,int(x1)):min(w,int(x2))]
                if patch.size and float(np.median(patch))>=.76:
                    return True,"overhead_center_obstacle",f"{_vi_label(obj.label)} ngang tầm đầu"
            elif (y2-y1)>=h*.44 and obj.confidence>=.70 and obj.label in {"tree","barrier","pole","signboard","awning"}:
                return True,"overhead_center_obstacle",f"{_vi_label(obj.label)} ngang tầm đầu"

    # 4. Close stairs or open-world depth obstacle / drop-off directly ahead.
    if scene.semantic is not None:
        gp=scene.semantic.get("group_probs")
        if gp and "stairs" in gp:
            foot_stairs=gp["stairs"][int(h*.76):int(h*.96),int(w*.36):int(w*.64)]
            if foot_stairs.size and float(foot_stairs.mean())>=.65:
                if depth_near is None or float(np.median(depth_near[int(h*.76):int(h*.96),int(w*.36):int(w*.64)]))>=.55:
                    return True,"near_stairs_hazard","cầu thang phía trước"

    if depth_enabled and depth_near is not None:
        anomalies=scene.depth_anomalies
        if anomalies is None:
            anomalies=analyze_depth_geometry(depth_near,scene.depth["valid"])
            scene.depth_anomalies=anomalies
        pos_box=anomalies["positive"][int(h*.68):int(h*.95),int(w*.38):int(w*.62)]
        near_box=depth_near[int(h*.68):int(h*.95),int(w*.38):int(w*.62)]
        if pos_box.size and float(pos_box.mean())>=.35 and float(np.median(near_box))>=.82:
            return True,"near_depth_obstacle","vật cản sát phía trước"
        drop_box=anomalies["dropoff"][int(h*.72):int(h*.96),int(w*.36):int(w*.64)]
        if drop_box.size and float(drop_box.mean())>=.30:
            return True,"near_dropoff_hazard","bậc hụt hoặc hố phía trước"
    return False,"",""

def urgent(scene):
    return urgent_reason(scene)[0]


def transverse_step_ahead(scene):
    """Identify a step-like surface crossing nearly the whole view."""
    h, w = scene.frame.bgr.shape[:2]
    return any(
        obj.label in {"stairs", "step"}
        and obj.confidence >= .45
        and obj.box[0] <= w * .12 and obj.box[2] >= w * .88
        and obj.box[3] >= h * (.30 if h > w else .50)
        and obj.box[3] - obj.box[1] >= h * .10
        for obj in scene.detections
    )

def route_blocked(route,scene,horizon=.5):
    return bool(route and route.mask is not None and
                route.mask.shape==scene.frame.bgr.shape[:2] and
                np.any(route.mask & obstacle_mask(scene,horizon)))

def _clock_from_shift(shift):
    if shift<-.20: return 10
    if shift<-.08: return 11
    if shift<=.08: return 12
    if shift<=.20: return 1
    return 2

class CorridorPlanner:
    def __init__(self,config):
        self.config=config
        self._prev_walk=None
        self._prev_session=None
        self._prev_pts=None
        self._env_session=None
        self._indoor_ema=0.
        self._alley_ema=0.

    def reset(self):
        self._prev_walk=None
        self._prev_session=None
        self._prev_pts=None

    def _resolve_environment(self,sem,session):
        gp=sem.get("group_probs")
        h=sem["classes"].shape[0]
        al_now=0.
        if gp and "indoor" in gp and "alley" in gp and "sidewalk" in gp:
            sl=slice(int(h*.45),int(h*.96))
            in_now=float(gp["indoor"][sl].mean())
            al_now=float(gp["alley"][sl].mean())
            sw_now=float(gp["sidewalk"][sl].mean())
            if session!=self._env_session:
                self._indoor_ema=in_now
                self._alley_ema=al_now-sw_now
                self._env_session=session
            else:
                in_alpha=.45 if in_now>=self._indoor_ema else .12
                self._indoor_ema=in_alpha*in_now+(1.-in_alpha)*self._indoor_ema
                self._alley_ema=.35*(al_now-sw_now)+.65*self._alley_ema
        if self.config.environment!="auto":
            return self.config.environment
        if self._indoor_ema>=max(.18,al_now*1.15):
            return "indoor"
        if self._alley_ema>=.06:
            return "alley"
        return "sidewalk"

    def plan(self,scene):
        c=self.config
        empty=lambda why: Route(None,reason=why)
        if scene.errors:
            self.reset(); return empty("model_output_error")
        if not scene.detector_valid:
            self.reset(); return empty("detector_unavailable")
        if scene.motion_invalid:
            self.reset(); return empty("camera_motion_or_cut")
        if scene.blur_score<12 or scene.brightness<18:
            self.reset(); return empty("poor_image")
        if scene.semantic is None:
            self.reset(); return empty("semantic_unavailable")
        sem=scene.semantic
        ids=sem["classes"]
        confidence=sem["confidence"]
        if confidence.shape!=ids.shape or not np.isfinite(confidence).all():
            self.reset(); return empty("invalid_semantic")
        h,w=ids.shape
        eff_env=self._resolve_environment(sem,scene.frame.session)
        names={int(k):str(v).lower().strip() for k,v in sem["labels"].items()}
        ground_ids=[k for k,v in names.items() if v in GROUND[eff_env]]
        structure_ids=[k for k,v in names.items() if v in STRUCTURES]

        gp=sem.get("group_probs")
        if gp and eff_env in gp and gp[eff_env].shape==(h,w):
            walk_prob=gp[eff_env].astype(np.float32,copy=True)
            struct_prob=gp["structure"] if "structure" in gp else np.where(np.isin(ids,structure_ids),confidence,0.)
            blocked=struct_prob>=c.semantic_threshold
        else:
            walk_prob=np.where(np.isin(ids,ground_ids),confidence,0.).astype(np.float32)
            blocked=np.isin(ids,structure_ids)&(confidence>=c.semantic_threshold)

        # Ego-motion-compensated temporal EMA stabilizes semantic flicker without delaying obstacle vetoes.
        frame=scene.frame
        if (self._prev_walk is not None and self._prev_walk.shape==(h,w)
                and self._prev_session==frame.session and self._prev_pts is not None
                and 0<frame.pts-self._prev_pts<=c.track_ttl_s and scene.warp is not None):
            warped=cv2.warpAffine(self._prev_walk,scene.warp,(w,h),flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT,borderValue=0.)
            alpha=float(c.temporal_ema_alpha)
            has_prior=warped>.05
            walk_prob=np.where(has_prior,alpha*walk_prob+(1.-alpha)*warped,walk_prob)

        depth_required=any(m.enabled and m.role=="depth" for m in c.models)
        if depth_required:
            if scene.depth is None: return empty("depth_unavailable")
            depth=scene.depth
            near=depth["relative_nearness"]
            valid=depth["valid"]&np.isfinite(near)
            if valid.shape!=ids.shape or valid.mean()<.95: return empty("invalid_depth")
            anomalies=analyze_depth_geometry(near,valid,ground_hint=walk_prob>=c.semantic_threshold)
            scene.depth_anomalies=anomalies
            # 3D Coplanar Specular-Tile Recovery: glossy indoor floor tiles often trigger false 2D "wall"/"ceiling"
            # predictions despite lying strictly flat on the verified 3D RANSAC ground plane.
            if (eff_env=="indoor" and gp and self._indoor_ema>=.25
                    and anomalies.get("plane_valid") and anomalies.get("plane_slope",0.)>=.30
                    and anomalies.get("residual") is not None):
                ambiguous_ids=[k for k,v in names.items() if v in {"wall","ceiling","screen door","building"}]
                if ambiguous_ids:
                    coplanar=(valid & np.isin(ids,ambiguous_ids) & (np.abs(anomalies["residual"])<.095)
                              & (anomalies["grad"]<.36) & ~anomalies["positive"]
                              & ~anomalies["dropoff"] & ~anomalies["uncertain"])
                    walk_prob=np.where(coplanar,np.maximum(walk_prob,np.clip(confidence,.65,.88)),walk_prob)
                    blocked=blocked&~coplanar
            if c.depth_obstacle_enabled:
                blocked|=anomalies["positive"]|anomalies["dropoff"]
            walk_prob=np.where(valid & ~anomalies["uncertain"],walk_prob,0.)

        self._prev_walk=walk_prob.copy()
        self._prev_session=frame.session
        self._prev_pts=frame.pts

        blocked |= obstacle_mask(scene,c.predict_horizon_s)
        walk_prob[blocked]=0.
        free=walk_prob>=c.semantic_threshold

        y0,y1=int(h*.45),int(h*.96)
        if h>w:
            y_ext=int(h*.22)
            if float(free[y_ext:y0].mean())>=.18:
                y0=y_ext
        gw,gh=c.grid_width,c.grid_height
        grid_prob=cv2.resize(walk_prob[y0:y1],(gw,gh),interpolation=cv2.INTER_AREA)
        grid_cov=cv2.resize(free[y0:y1].astype(np.float32),(gw,gh),interpolation=cv2.INTER_AREA)
        grid_block=cv2.resize(blocked[y0:y1].astype(np.float32),(gw,gh),interpolation=cv2.INTER_AREA)>.02
        grid_free=(grid_prob>=c.semantic_threshold)&(grid_cov>=.85)&~grid_block

        # Replicate only vertically; both lateral image edges remain blocked.
        padded=np.pad(grid_free.astype(np.uint8),((gh,gh),(1,1)),mode="edge")
        padded[:,0]=0; padded[:,-1]=0
        clearance=cv2.distanceTransform(padded,cv2.DIST_L2,5)[gh:2*gh,1:-1]
        required=np.linspace(c.margin_top_cells,c.margin_bottom_cells,gh)[:,None]
        # Normalize clearance by perspective foreshortening so vanishing-point convergence is not misclassified as narrow.
        norm_clearance=clearance*(c.margin_bottom_cells/np.maximum(required,1e-6))
        admissible=grid_free & (clearance>required)
        states=np.zeros((gh,gw),np.uint8) # 0 unknown, 1 evidenced ground, 2 blocked
        states[grid_free]=1
        states[grid_block]=2

        # Multi-horizon Dynamic Programming from anchored footbox toward deepest clear horizon.
        score=np.full((gh,gw),-np.inf)
        cost=np.full((gh,gw),np.inf)
        parent=np.full((gh,gw),-1,int)
        center=(gw-1)/2
        starts=[x for x in range(gw) if abs(x-center)<=2 and admissible[-1,x]]
        for x in starts:
            score[-1,x]=norm_clearance[-1,x]
            cost[-1,x]=abs(x-center)
        top_y=gh
        if starts:
            top_y=gh-1
        for y in range(gh-2,-1,-1):
            for x in np.flatnonzero(admissible[y]):
                for p in range(max(0,x-1),min(gw,x+2)):
                    if not np.isfinite(score[y+1,p]): continue
                    capacity=min(score[y+1,p],norm_clearance[y,x])
                    effort=cost[y+1,p]+abs(x-p)+abs(x-center)*.015
                    if capacity>score[y,x]+1e-6 or (abs(capacity-score[y,x])<1e-6 and effort<cost[y,x]):
                        score[y,x],cost[y,x],parent[y,x]=capacity,effort,p
            if np.any(np.isfinite(score[y])):
                top_y=y

        max_confirm_row=int(round((gh-1)*(1.-c.confirm_horizon_ratio)))
        reached_ratio=0. if top_y>=gh else float((gh-1-top_y)/max(1,gh-1))
        if top_y>max_confirm_row:
            why="short_corridor_ahead" if top_y<=int(gh*.62) else "no_continuous_corridor"
            # Unknown pixels or insufficient width alone do not prove a blockage.
            # Require object or depth evidence. Segmentation can mistake glossy floor for a wall.
            observed_obstacles = obstacle_mask(scene, horizon=0.)
            front = observed_obstacles[int(h*.40):int(h*.88), int(w*.12):int(w*.88)]
            if front.size:
                dense_rows = front.mean(axis=1) >= .9
                band = max(1, round(h*.04))
                if len(dense_rows) >= band and np.convolve(dense_rows, np.ones(band), mode="valid").max() >= band:
                    why = "forward_corridor_blocked"
            return Route(None,states=states,reason=why,evidence_complete=True,horizon_ratio=reached_ratio)

        candidates=np.flatnonzero(np.isfinite(score[top_y]))
        best=sorted(candidates,key=lambda x:(-score[top_y,x],cost[top_y,x],abs(x-center)))[0]
        # Equal alternatives on opposite sides do not justify an arbitrary turn.
        ties=[x for x in candidates if abs(score[top_y,x]-score[top_y,best])<1e-6 and abs(cost[top_y,x]-cost[top_y,best])<1e-6]
        if min(ties)<center-3 and max(ties)>center+3:
            return Route(None,states=states,reason="ambiguous_corridors",evidence_complete=True,horizon_ratio=reached_ratio)

        points=[]
        path_clearances=[]
        x=best
        for y in range(top_y,gh):
            points.append((int((x+.5)*w/gw),int(y0+(y+.5)*(y1-y0)/gh)))
            path_clearances.append(float(norm_clearance[y,x]))
            if y<gh-1: x=parent[y,x]
        points.reverse()
        path_clearances.reverse()
        corridor=np.zeros((h,w),np.uint8)
        for i in range(len(points)-1):
            radius=max(2,round(c.margin_bottom_cells*w/gw))
            cv2.line(corridor,points[i],points[i+1],1,2*radius)
        upper_x=float(np.mean([p[0] for p in points[len(points)//2:]]))
        start_x=float(points[0][0])
        shift=.65*(upper_x/w-.5)+.35*((upper_x-start_x)/w)
        action=Action.SHIFT_LEFT if shift<-.058 else Action.SHIFT_RIGHT if shift>.058 else Action.KEEP_HEADING
        clock_hour=_clock_from_shift(shift)
        # Evaluate corridor width over the near/mid body path (excluding the far horizon termination tip).
        body_span=path_clearances[:max(1,int(len(path_clearances)*.85))]
        bottleneck=float(min(body_span)) if body_span else float(score[top_y,best])
        narrow=bottleneck<c.margin_bottom_cells+.35
        reason="narrow_corridor" if narrow else "continuous_ground"
        if eff_env=="alley":
            road_present=(gp and "alley" in gp and "sidewalk" in gp and float(np.mean(gp["alley"]-gp["sidewalk"]))>.02) or any(
                v=="road" and np.any(ids==k) for k,v in names.items()
            )
            if road_present:
                if not narrow:
                    reason="road_candidate_traffic_risk"
                if any(d.label in VEHICLES for d in scene.detections):
                    return Route(None,states=states,reason="traffic_on_candidate_ground",
                                 evidence_complete=True,horizon_ratio=reached_ratio)
        return Route(action,points,corridor.astype(bool),states,bottleneck,narrow,reason,True,reached_ratio,clock_hour)
