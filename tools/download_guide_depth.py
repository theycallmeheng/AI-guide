"""Fetch the official relative-depth model and record the resolved immutable revision."""
from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download
repo="depth-anything/Depth-Anything-V2-Small-hf"
if __name__=="__main__":
    revision=HfApi().model_info(repo).sha
    path=Path("models/depth_anything_v2_small")
    snapshot_download(repo,revision=revision,local_dir=str(path),
                      allow_patterns=["config.json","preprocessor_config.json","model.safetensors","README.md","LICENSE*"])
    (path/"source.json").write_text(json.dumps({"repo":repo,"revision":revision,"depth_type":"relative"},indent=2),encoding="utf-8")
    print("Depth model downloaded; immutable revision: "+revision)

