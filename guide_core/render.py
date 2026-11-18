from functools import lru_cache
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

@lru_cache(maxsize=16)
def font(size):
    candidates = [
        "assets/fonts/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    spec = importlib.util.find_spec("matplotlib")
    if spec and spec.origin:
        candidates.append(str(Path(spec.origin).parent / "mpl-data/fonts/ttf/DejaVuSans.ttf"))
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def render(scene,decision,stage="fused"):
    image=scene.frame.bgr.copy()
    h,w=image.shape[:2]
    for obj in scene.detections:
        x1,y1,x2,y2=map(int,obj.box)
        ttc=getattr(obj,"ttc_s",None)
        box_color=(0,50,255) if (ttc is not None and ttc<=1.6) or getattr(obj,"zone","ground")=="head" else (0,140,255)
        cv2.rectangle(image,(x1,y1),(x2,y2),box_color,2)
        ttc_tag=f" TTC:{ttc:.1f}s" if ttc is not None else ""
        head_tag=" [HEAD]" if getattr(obj,"zone","ground")=="head" else ""
        text=f"{obj.label}{head_tag} #{obj.track_id or '-'} {obj.confidence:.2f}{ttc_tag}"
        cv2.putText(image,text,(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,230,255),1,cv2.LINE_AA)
    route=decision.route if decision.frame_key==scene.frame.key else None
    if route and route.path:
        cv2.polylines(image,[np.array(route.path,np.int32)],False,(50,230,100),3)
    panels=[]
    semantic=np.zeros_like(image)
    if scene.semantic:
        ids=scene.semantic["classes"].astype(np.int32)
        semantic=np.stack(((ids*53)%220+20,(ids*97)%220+20,(ids*131)%220+20),axis=2).astype(np.uint8)
        semantic=cv2.addWeighted(scene.frame.bgr,.45,semantic,.55,0)
    panels.append(semantic)
    depth=np.zeros_like(image)
    if scene.depth:
        depth=cv2.applyColorMap((scene.depth["relative_nearness"]*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
        anomalies=getattr(scene,"depth_anomalies",None)
        if anomalies:
            if anomalies.get("positive") is not None and anomalies["positive"].shape==(h,w):
                depth[anomalies["positive"]]=[40,40,255]
            if anomalies.get("dropoff") is not None and anomalies["dropoff"].shape==(h,w):
                depth[anomalies["dropoff"]]=[255,220,0]
    panels.append(depth)
    states=np.zeros_like(image)
    if route and route.states is not None:
        palette=np.array([[70,70,70],[50,170,60],[50,50,220]],np.uint8)
        states=cv2.resize(palette[route.states],(w,h),interpolation=cv2.INTER_NEAREST)
    panels.append(states)
    # Fixed 960px width; enough space for Vietnamese subtitles and three context panels.
    pw=320
    main_h=max(180,round(h*960/w))
    image=cv2.resize(image,(960,main_h))
    ph=180
    context=np.hstack([cv2.resize(p,(pw,ph)) for p in panels])
    canvas=np.vstack([image,context,np.zeros((116,960,3),np.uint8)])
    pil=Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    draw=ImageDraw.Draw(pil)
    draw.text((12,8),f"frame {scene.frame.sequence} | video {scene.frame.pts:.2f}s | {stage}",font=font(18),fill="white",stroke_width=1,stroke_fill="black")
    for i,label in enumerate(("Ngữ nghĩa","Chiều sâu tương đối","Nền / Chắn / Chưa rõ")):
        draw.text((i*pw+8,main_h+5),label,font=font(17),fill="white",stroke_width=1,stroke_fill="black")
    draw.text((12,main_h+ph+8),decision.text,font=font(26),fill="#ffdc8a")
    draw.text((12,main_h+ph+48),decision.reason,font=font(17),fill="white")
    draw.text((12,main_h+ph+79),"Demo nghiên cứu • Không đo mét • Chưa xác nhận an toàn đi bộ",font=font(17),fill="#b7c7d5")
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)

