import queue
import tempfile
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
import wave
import numpy as np
from audio_output import Audio, RATE, read_audio
from fast_alerts import AlertBank, word_id
from feedback import SpeechWorker
from test_runtime import FakeOutput

def await_condition(predicate, seconds=3):
    deadline=time.perf_counter()+seconds
    while time.perf_counter()<deadline:
        if predicate(): return True
        time.sleep(.005)
    return False

class AudioTests(unittest.TestCase):
    def test_trim_preserves_speech_and_resamples(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"voice.wav"
            signal=np.concatenate([np.zeros(2000),np.ones(3200)*1000,np.zeros(12000)]).astype("<i2")
            with wave.open(str(path),"wb") as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(signal.tobytes())
            audio=read_audio(path)
            self.assertAlmostEqual(audio.duration,.26,delta=.002)
            self.assertEqual(np.max(np.frombuffer(audio.pcm,dtype="<i2")),1000)

    def test_command_is_first_for_routine_warning(self):
        bank=AlertBank(".",{"person":"người"})
        for text,value in (("người",1000),("phía trước",2000),("gần",3000)):
            bank.audio[word_id(text)]=Audio(np.full(240,value,dtype="<i2").tobytes())
        bank.audio["stop"]=Audio(np.full(240,4000,dtype="<i2").tobytes())
        bank.audio["slow_down"]=Audio(np.full(240,5000,dtype="<i2").tobytes())
        normal=bank.compose("Person|gần|phía trước",2)
        urgent=bank.compose("Person|gần|phía trước",3)
        self.assertEqual(np.frombuffer(normal.pcm,dtype="<i2")[0],5000)
        self.assertEqual(np.frombuffer(normal.pcm,dtype="<i2")[840],1000)
        self.assertEqual(np.frombuffer(urgent.pcm,dtype="<i2")[0],4000)
        self.assertIsNone(bank.compose("unknown|gần|phía trước",2))

    def test_route_audio_keeps_command_before_obstacle_and_stop_before_turn(self):
        bank=AlertBank(".",{"person":"người"})
        for key,value in (("stop",1000),("go_forward",2000),("turn_left",3000),("turn_right",4000),
                          (word_id("người"),5000),(word_id("phía trước"),6000),(word_id("gần"),7000)):
            bank.audio[key]=Audio(np.full(240,value,dtype="<i2").tobytes())
        samples=lambda key,priority=2:np.frombuffer(bank.compose(key,priority).pcm,dtype="<i2")
        self.assertEqual(samples("route|front")[0],2000)
        self.assertEqual(samples("go_forward|Person|gần|phía trước")[0],2000)
        for command,value in (("turn_left",3000),("turn_right",4000)):
            audio=samples(command+"|Person|gần|phía trước")
            self.assertEqual((audio[0],audio[840],audio[1680]),(1000,value,5000))
        self.assertEqual(samples("go_forward|Person|gần|phía trước",3)[0],1000)

    def test_cached_hazard_preempts_slow_synthesis_and_deduplicates(self):
        gate=threading.Event(); synthesizing=threading.Event(); events=queue.Queue()
        class Engine:
            name="fake"
            def synthesize(self,text,path):
                if text=="delayed":
                    synthesizing.set();gate.wait(3)
                with wave.open(str(path),"wb") as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(RATE)
                    w.writeframes(np.full(2400,1000,dtype="<i2").tobytes())
        with tempfile.TemporaryDirectory() as directory, patch("feedback.create_engine",return_value=Engine()):
            output=FakeOutput()
            worker=SpeechWorker(events,cache_dir=directory,output_factory=lambda:output)
            try:
                self.assertTrue(worker.audio_ready.wait(2))
                self.assertTrue(await_condition(lambda:worker.bank.compose("Person|gần|phía trước",3) is not None))
                worker.speak("delayed")
                self.assertTrue(synthesizing.wait(2))
                worker.speak("Dừng lại. Có người gần ở phía trước.",priority=3,clip="stop",key="Person|gần|phía trước")
                self.assertTrue(await_condition(lambda:len(output.played)>=2))
                # Synthesis is still blocked, yet the complete urgent alert was submitted.
                self.assertFalse(gate.is_set())
                count=len(output.played)
                worker.speak("duplicate",priority=3,key="Person|gần|phía trước")
                time.sleep(.02)
                self.assertEqual(len(output.played),count)
                worker.cancel()
                self.assertTrue(await_condition(lambda:not output.busy()))
                gate.set()
                self.assertTrue(await_condition(lambda:worker._pending is None))
            finally:
                gate.set();worker.close()
                worker._synth_thread.join(2)

if __name__=="__main__":unittest.main()
