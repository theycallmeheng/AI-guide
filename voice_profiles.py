"""Persist the selected prerecorded voice without changing the working Windows voice bank."""
import json
from pathlib import Path
from fast_alerts import AlertBank, FIXED, DIRECTIONS, BANDS, word_id

APP_DIR=Path(__file__).resolve().parent
SETTINGS=APP_DIR/"voice_settings.json"
PRESETS={
    "an":{"name":"Microsoft An","engine":"windows","voice":None,"bank":"assets/audio/fast"},
    "khanh_vy":{"name":"Khánh Vy","engine":"korva","voice":"khanh_vy","bank":"assets/audio/voices/khanh_vy"},
    "hoang_nam":{"name":"Hoàng Nam","engine":"korva","voice":"hoang_nam","bank":"assets/audio/voices/hoang_nam"},
}

def selected_id():
    try:
        result=json.loads(SETTINGS.read_text(encoding="utf-8"))
        value=result.get("voice","an")
        return value if value in PRESETS else "an"
    except (OSError,ValueError,AttributeError,TypeError):
        return "an"

def selected_profile():
    identifier=selected_id()
    preset=dict(PRESETS[identifier])
    preset["id"]=identifier
    return preset

def validate_bank(identifier, labels):
    if identifier not in PRESETS:raise ValueError("Giọng không hợp lệ.")
    bank=AlertBank(APP_DIR/PRESETS[identifier]["bank"],labels)
    required=set(FIXED) | {word_id(t) for t in set(labels.values())|set(DIRECTIONS)|set(BANDS)}
    missing=required-set(bank.audio)
    if missing:raise ValueError(f"Giọng {identifier} còn thiếu {len(missing)} đoạn hợp lệ. Chạy --prepare trước.")
    return bank

def activate(identifier,labels):
    validate_bank(identifier,labels)  # Leave the old selection intact if generation failed.
    temporary=SETTINGS.with_suffix(".tmp.json")
    try:
        temporary.write_text(json.dumps({"voice":identifier},ensure_ascii=False,indent=2),encoding="utf-8")
        temporary.replace(SETTINGS)
    finally:temporary.unlink(missing_ok=True)
