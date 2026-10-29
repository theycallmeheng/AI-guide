import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave
import voice_profiles
from fast_alerts import AlertBank

class Engine:
    def synthesize(self,text,path):
        with wave.open(str(path),"wb") as audio:
            audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(24000)
            audio.writeframes(b"\0"*4800)

class VoiceProfileTests(unittest.TestCase):
    def test_incomplete_profile_keeps_previous_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);settings=root/"voice_settings.json"
            settings.write_text('{"voice":"an"}')
            with patch.object(voice_profiles,"APP_DIR",root),patch.object(voice_profiles,"SETTINGS",settings):
                before=settings.read_bytes()
                with self.assertRaises(ValueError):voice_profiles.activate("khanh_vy",{"person":"người"})
                self.assertEqual(settings.read_bytes(),before)
                self.assertEqual(voice_profiles.selected_id(),"an")

    def test_valid_profile_can_switch_without_overwriting_default_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);settings=root/"voice_settings.json"
            original=root/"assets/audio/fast"
            original.mkdir(parents=True)
            (original/"sentinel").write_text("original")
            with patch.object(voice_profiles,"APP_DIR",root),patch.object(voice_profiles,"SETTINGS",settings):
                bank=AlertBank(root/voice_profiles.PRESETS["khanh_vy"]["bank"],{"person":"người"})
                bank.prepare(Engine())
                voice_profiles.activate("khanh_vy",{"person":"người"})
                self.assertEqual(voice_profiles.selected_profile()["name"],"Khánh Vy")
                self.assertEqual((original/"sentinel").read_text(),"original")

if __name__=="__main__":unittest.main()
