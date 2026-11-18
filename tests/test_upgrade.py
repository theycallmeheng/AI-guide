import queue
import tempfile
import time
import unittest
import threading
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch
from dataclasses import replace
from test_runtime import analysis, obstacle, FakeOutput
from test_audio import await_condition
from temporal_tracking import TemporalTracker
from guidance import AlertPolicy
from feedback import SpeechWorker
from audio_output import Audio
from tools.evaluate_alert_events import evaluate
from tools.audit_detection_dataset import audit
from obstacle_detector import LatestFrame, InferenceWorker
from PIL import Image


class UpgradeTests(unittest.TestCase):
    def test_expired_input_is_not_inferred(self):
        latest=LatestFrame()
        latest.publish(analysis().frame)
        latest._captured_at=time.perf_counter()-1
        latest.close()
        events, speech=queue.Queue(),Mock()
        fake={'torch':SimpleNamespace(set_num_threads=lambda n:None),
              'ultralytics':SimpleNamespace(YOLO=Mock())}
        with patch.dict('sys.modules',fake), patch('obstacle_detector.analyze_frame') as analyze:
            InferenceWorker('fake.pt',.3,latest,threading.Event(),events,speech).run()
            analyze.assert_not_called()
            speech.speak.assert_not_called()
        self.assertIn('analysis_stale',[e['type'] for e in events.queue])

    def test_slow_result_does_not_speak(self):
        latest=LatestFrame()
        latest.publish(analysis().frame)
        latest.close()
        events, speech=queue.Queue(),Mock()
        fake={'torch':SimpleNamespace(set_num_threads=lambda n:None),
              'ultralytics':SimpleNamespace(YOLO=Mock())}
        def slow(*args,**kwargs):
            time.sleep(.72)
            return analysis([obstacle()])
        with patch.dict('sys.modules',fake), patch('obstacle_detector.analyze_frame',side_effect=slow):
            InferenceWorker('fake.pt',.3,latest,threading.Event(),events,speech).run()
        speech.speak.assert_not_called()
        self.assertIn('analysis_stale',[e['type'] for e in events.queue])

    def test_dataset_audit_rejects_invalid_box_and_location_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[]
            for index,split in enumerate(('train','val','test')):
                (root/'images'/split).mkdir(parents=True)
                (root/'labels'/split).mkdir(parents=True)
                Image.new('RGB',(8,8),(index*60,20,10)).save(root/'images'/split/'a.png')
                (root/'labels'/split/'a.txt').write_text('0 0.5 0.5 0.5 0.5\n')
                rows.append(dict(image=f'images/{split}/a.png',split=split,location_id=split,session_id=split,verified_empty='0'))
            data=root/'data.yaml'
            data.write_text(json.dumps(dict(names=['Person'],train='images/train',val='images/val',test='images/test')))
            manifest=root/'manifest.csv'
            def save():
                with manifest.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            save()
            self.assertEqual(audit(data,manifest)['images']['train'],1)
            rows[1]['location_id']='train';save()
            with self.assertRaisesRegex(AssertionError,'Split leakage'):audit(data,manifest)
            rows[1]['location_id']='val';save()
            (root/'labels/train/a.txt').write_text('0 1 1 0.5 0.5\n')
            with self.assertRaisesRegex(AssertionError,'Invalid box'):audit(data,manifest)

    def test_observed_only_and_confirmation_window(self):
        tracker=TemporalTracker()
        scene=analysis([replace(obstacle(band='vừa'), confidence=.5)])
        self.assertFalse(tracker.update(scene,0).obstacles)
        first=tracker.update(scene,.1).obstacles[0]
        self.assertFalse(tracker.update(analysis(),.2).obstacles)
        self.assertEqual(tracker.update(scene,.25).obstacles[0].track_id,first.track_id)
        self.assertFalse(tracker.update(scene,1).obstacles)

    def test_emergency_immediate_and_direction_hysteresis(self):
        tracker=TemporalTracker()
        scene=analysis([obstacle()])
        self.assertEqual(len(tracker.update(scene,0).obstacles),1)
        scene=analysis([replace(obstacle(band='vừa'),direction='bên trái',confidence=.5)])
        self.assertEqual(tracker.update(scene,.05).obstacles[0].direction,'phía trước')
        self.assertEqual(tracker.update(scene,.2).obstacles[0].direction,'bên trái')

    def test_default_never_directs_turn_or_forward(self):
        policy=AlertPolicy()
        scene=analysis([obstacle()],clear=('bên trái','phía trước'))
        for now in (0,1,2):
            self.assertEqual(policy.update(scene,{},now)[0].action,'stop')

    def test_audio_finishes_current_and_drops_expired_pending(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(SpeechWorker,'_synthesize',return_value=None):
            output=FakeOutput()
            worker=SpeechWorker(queue.Queue(),cache_dir=directory,output_factory=lambda:output)
            try:
                self.assertTrue(worker.audio_ready.wait(2))
                worker.bank.audio['stop']=Audio(bytes(24000))
                worker.speak('active',clip='stop',priority=3,key='active')
                self.assertTrue(await_condition(lambda:worker._active_key=='active'))
                worker.speak('stale',clip='stop',priority=3,key='old',max_age=.02)
                time.sleep(.04)
                self.assertEqual(worker._active_key,'active')
                self.assertTrue(await_condition(lambda:worker._active_key is None))
                time.sleep(.03)
                self.assertEqual(len(output.played),2) # warmup + active only
            finally: worker.close()

    def test_audio_keeps_only_latest_observed_pending(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(SpeechWorker,'_synthesize',return_value=None):
            output=FakeOutput()
            worker=SpeechWorker(queue.Queue(),cache_dir=directory,output_factory=lambda:output)
            try:
                self.assertTrue(worker.audio_ready.wait(2))
                worker.bank.audio['stop']=Audio(bytes(16000))
                worker.speak('active',clip='stop',priority=3,key='active')
                self.assertTrue(await_condition(lambda:worker._active_key=='active'))
                worker.speak('old',clip='stop',priority=3,key='old')
                worker.observe('new',time.perf_counter(),3)
                self.assertIsNone(worker._play)
                worker.speak('new',clip='stop',priority=3,key='new')
                worker.observe('new',time.perf_counter(),3)
                self.assertTrue(await_condition(lambda:worker._active_key=='new'))
                self.assertEqual(len(output.played),3)
                worker.cancel()
                self.assertTrue(await_condition(lambda:not output.busy()))
            finally: worker.close()

    def test_event_metrics_do_not_count_wrong_direction(self):
        truth={'duration_s':60,'events':[{'id':'e1','start_s':1,'end_s':5,'action':'stop','direction':'front','class_name':'person'}], 'clear_intervals':[[10,60]]}
        alerts=[{'time_s':2,'action':'stop','direction':'left','class_name':'person'},
                {'time_s':20,'action':'stop','direction':'front','class_name':'person'}]
        result=evaluate(truth,alerts)
        self.assertEqual(result['recall'],0)
        self.assertEqual(result['false_alerts'],1)
        self.assertTrue(result['preliminary'])

if __name__=='__main__': unittest.main()
