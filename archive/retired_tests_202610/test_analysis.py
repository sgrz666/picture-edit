import tempfile
import unittest
from pathlib import Path
from vram_lab.common import write_json
from vram_lab.analysis import capacity,select

class AcceptanceTests(unittest.TestCase):
    def test_capacity_requires_both_inference_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def add(name,c,s):
                write_json(root/'runs'/name/'config.json',dict(id=name,resolution=512,**c))
                write_json(root/'runs'/name/'summary.json',s)
            for i in range(3):
                add(f'stable{i}',dict(group='stable',kind='train',encoding='cache',checkpointing=True),
                    dict(status='PASS',validation_passed=True,train_steps=100,peak_nvml_gib=20,step_p50_seconds=1,
                         stages=[dict(stage='steady_train',peak_allocated_gib=18,peak_reserved_gib=19)]))
            for task in ('S0','D0'):
                add('cache'+task,dict(kind='cache',task=task),dict(status='PASS',cache_roundtrip_exact=True,peak_nvml_gib=16))
            add('inferS0',dict(kind='adapter_infer',task='S0'),dict(status='PASS',peak_nvml_gib=17))
            add('inferD0',dict(kind='adapter_infer',task='D0'),dict(status='OOM',peak_nvml_gib=83))
            self.assertEqual(capacity(root),[])
            add('inferD0',dict(kind='adapter_infer',task='D0'),dict(status='PASS',peak_nvml_gib=18))
            self.assertEqual(capacity(root)[0]['suggested_capacity_gib'],22)
    def test_partial_scores_do_not_silently_select(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scores={}
            for task in ('S2','D1'):
                for seed in (42,1024):
                    name=f'{task}_{seed}'
                    write_json(root/'runs'/name/'config.json',dict(id=name,kind='quality',task=task,seed=seed,group='Q0',
                               preprocess='letterbox',cfg=4.5,steps=35,negative_prompt=''))
                    write_json(root/'runs'/name/'summary.json',dict(status='PASS'))
                    scores[name]={'action':2}
            write_json(root/'scores.json',scores)
            with self.assertRaisesRegex(ValueError,'Invalid or missing score'):
                select(root)

if __name__=='__main__':unittest.main()
