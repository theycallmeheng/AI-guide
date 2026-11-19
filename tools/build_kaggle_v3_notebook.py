"""Build a self-contained Kaggle GPU notebook (dist/guide_v3_kaggle_gpu.ipynb) with SHA-256 verified Guide v3 code overlay.

Allows running Guide v3 immediately on Kaggle on top of the existing `guide-v2-models` dataset
without re-uploading 117 MiB of model weights.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

ROOT = Path(__file__).resolve().parents[1]


def collect_overlay_files() -> dict[str, dict[str, str]]:
    rels: list[Path] = []
    for p in sorted((ROOT / "guide_core").glob("*.py")):
        rels.append(p.relative_to(ROOT))
    for name in (
        "configs/guide_kaggle.json",
        "configs/guide_optimized.json",
        "requirements-guide-core.txt",
        "tools/check_guide_environment.py",
        "tools/compare_guide_models.py",
        "assets/audio/guide_core/khanh_vy/manifest.json",
        "assets/audio/guide_core/khanh_vy/forward_corridor_blocked.wav",
        "assets/audio/guide_core/khanh_vy/transverse_step_ahead.wav",
        "assets/audio/guide_core/khanh_vy/scan_for_bypass.wav",
        "assets/audio/guide_core/khanh_vy/bypass_left.wav",
        "assets/audio/guide_core/khanh_vy/bypass_right.wav",
    ):
        rels.append(Path(name))

    files_meta: dict[str, dict[str, str]] = {}
    for rel in rels:
        full = ROOT / rel
        raw = full.read_bytes()
        files_meta[rel.as_posix()] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "b64": base64.b64encode(raw).decode("ascii"),
        }
    return files_meta


def build_overlay_snippet(files_meta: dict[str, dict[str, str]]) -> list[str]:
    blob_bytes = json.dumps(files_meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(blob_bytes, level=9)
    archive_sha256 = hashlib.sha256(compressed).hexdigest()
    b64_lines = textwrap.wrap(base64.b64encode(compressed).decode("ascii"), width=100)

    lines = [
        "\n",
        "# --- Tự động nạp bản nâng cấp mã nguồn Guide v3 (kiểm tra SHA-256 từng tệp) ---\n",
        "import base64 as _b64, zlib as _zlib\n",
        "_V3_ARCHIVE_SHA256 = '" + archive_sha256 + "'\n",
        "_V3_COMPRESSED_B64 = (\n",
    ]
    for chunk in b64_lines:
        lines.append(f"    '{chunk}'\n")
    lines.extend([
        ")\n",
        "_compressed_bytes = _b64.b64decode(_V3_COMPRESSED_B64)\n",
        "assert hashlib.sha256(_compressed_bytes).hexdigest() == _V3_ARCHIVE_SHA256, 'Lỗi toàn vẹn gói overlay Guide v3'\n",
        "_v3_files = json.loads(_zlib.decompress(_compressed_bytes).decode('utf-8'))\n",
        "for _rel_path, _entry in _v3_files.items():\n",
        "    _raw = _b64.b64decode(_entry['b64'])\n",
        "    assert hashlib.sha256(_raw).hexdigest() == _entry['sha256'], f'Lỗi SHA-256 overlay: {_rel_path}'\n",
        "    _dest = (WORK / _rel_path).resolve()\n",
        "    assert _dest.is_relative_to(WORK.resolve())\n",
        "    _dest.parent.mkdir(parents=True, exist_ok=True)\n",
        "    _dest.write_bytes(_raw)\n",
        "print(f'Đã nạp & xác thực SHA-256 thành công {len(_v3_files)} tệp mã nguồn Guide v3 mới nhất!')\n",
    ])
    return lines


def build_notebook(output_path: Path) -> dict:
    template = json.loads((ROOT / "kaggle_demo_yolo_segformer.ipynb").read_text(encoding="utf-8"))
    nb = copy.deepcopy(template)

    # Update header note to emphasize zero-reupload convenience
    nb["cells"][0]["source"][-1] = (
        "**Chạy ngay không cần upload lại 117 MB trọng số:** Chỉ cần Import file `.ipynb` này lên Kaggle, "
        "bật **Accelerator = GPU T4 x2**, **Internet = ON**, gắn Dataset **`guide-v2-models`** sẵn có "
        "(hoặc `guide-v3-models`) cùng Dataset chứa video `.mp4`/`.mov`, rồi bấm **Run All**."
    )

    files_meta = collect_overlay_files()
    overlay_lines = build_overlay_snippet(files_meta)

    # Find Cell 1 code cell (unpack_and_verify) and append the self-verifying Guide v3 overlay
    injected = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code" and any("unpack_and_verify" in line for line in cell["source"]):
            src = list(cell["source"])
            if src and not src[-1].endswith("\n"):
                src[-1] = src[-1] + "\n"
            src.extend(overlay_lines)
            cell["source"] = src
            injected = True
            break
    if not injected:
        raise RuntimeError("Could not locate unpack_and_verify cell in template notebook.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "notebook": str(output_path),
        "overlay_files": len(files_meta),
        "size_kb": round(output_path.stat().st_size / 1024, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist" / "guide_v3_kaggle_gpu.ipynb"))
    args = parser.parse_args()
    summary = build_notebook(Path(args.output))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
