from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from guide_core.evaluate import validate_annotations,check_split_leakage
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("files",nargs="+")
    a=p.parse_args()
    data=[json.loads(Path(f).read_text(encoding="utf-8")) for f in a.files]
    for item in data: validate_annotations(item)
    check_split_leakage(data)
    print("No shared location/session across validation/test in supplied manifests.")

