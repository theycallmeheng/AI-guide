"""Create an allowlisted, POSIX-path Kaggle bundle; never package a Python environment."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
def inside(path):
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT): raise ValueError("Bundle assets must be inside project: "+str(path))
    return path

def build(output,video=None,models=True):
    output=inside(output)
    included=set()
    for directory,pattern in (("guide_core","*.py"),("configs","guide_*.json"),("assets/audio/guide_core/khanh_vy","*.wav")):
        included.update(ROOT.joinpath(directory).glob(pattern))
    for name in ("requirements-guide-core.txt","GUIDE_V2.md","GUIDE_V2_REPORT.md","kaggle_demo_yolo_segformer.ipynb",
                 "tools/compare_guide_models.py","tools/download_guide_depth.py","tools/check_guide_environment.py",
                 "tools/annotate_guide.html","tools/check_guide_splits.py",
                 "data/guide_events.template.json","data/guide_eval_manifest.template.json",
                 "assets/audio/guide_core/khanh_vy/manifest.json"):
        included.add(ROOT/name)
    if models:
        cfg=json.loads((ROOT/"configs/guide_kaggle.json").read_text(encoding="utf-8"))
        for spec in cfg["models"]:
            path=inside(ROOT/spec["path"])
            if path.is_dir():
                for pattern in ("*.json","*.safetensors","README.md","LICENSE*"):
                    included.update(path.glob(pattern))
            else: included.add(path)
    if video:
        included.add(inside(video))
        source=inside(video).parent/"source.json"
        if source.exists(): included.add(source)
    files=sorted(inside(p) for p in included)
    missing=[str(p) for p in files if not p.is_file()]
    if missing: raise FileNotFoundError("Missing bundle files: "+str(missing))
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists(): raise FileExistsError("Use a new bundle filename: "+str(output))
    manifest={"schema_version":1,"files":[],"video":inside(video).relative_to(ROOT).as_posix() if video else None}
    with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for path in files:
            relative=path.relative_to(ROOT).as_posix()
            if "\\" in relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise ValueError("Unsafe archive path")
            digest=hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda:f.read(1024*1024),b""):digest.update(chunk)
            archive.write(path,relative)
            manifest["files"].append({"path":relative,"bytes":path.stat().st_size,"sha256":digest.hexdigest()})
        archive.writestr("bundle_manifest.json",json.dumps(manifest,indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None: raise RuntimeError("ZIP integrity failed")
    print(json.dumps({"bundle":str(output),"files":len(files),"MiB":round(output.stat().st_size/1024**2,2)},ensure_ascii=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="dist/guide_kaggle_v2.zip")
    p.add_argument("--video");p.add_argument("--code-only",action="store_true")
    a=p.parse_args();build(a.output,a.video,not a.code_only)

