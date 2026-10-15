import collections
import unittest
from PIL import Image
from vram_lab.common import preprocess,quality_manifest,training_manifest

class ExperimentContractTests(unittest.TestCase):
    def test_factorial_comparisons_and_reference_reuse(self):
        rows=quality_manifest()
        self.assertEqual(collections.Counter(x['group'] for x in rows),{'Q0':24,'Q1':4,'Q2':20,'Q3':4})
        self.assertEqual(len({x['id'] for x in rows}),52)
        for task in ('S2','D1'):
            eligible=[x for x in rows if x['task']==task and x['preprocess']=='letterbox' and x['group'] in ('Q0','Q2')]
            self.assertEqual(len(eligible),12)
            self.assertEqual({(x['cfg'],x['steps'],x['seed']) for x in eligible},
                             {(c,n,s) for c in (2.5,4.5,6.0) for n in (35,50) for s in (42,1024)})
    def test_letterbox_preserves_content_and_records_geometry(self):
        for w,h in ((640,425),(480,640)):
            result,box=preprocess(Image.new('RGB',(w,h),'red'),512,'letterbox')
            self.assertEqual(result.size,(512,512))
            content=result.crop(box)
            self.assertLess(abs(content.width/content.height-w/h),.004)
            self.assertEqual(content.getpixel((0,0)),(255,0,0))
            self.assertEqual(result.getpixel((0,0)),(128,128,128))
    def test_training_design(self):
        rows=training_manifest()
        self.assertEqual(len(rows),24)
        self.assertEqual(len(training_manifest(True)),12)
        for row in rows:
            self.assertEqual(row['warmup']+row['updates'],25)
            self.assertTrue(row['control_injected'])

if __name__=='__main__':unittest.main()
