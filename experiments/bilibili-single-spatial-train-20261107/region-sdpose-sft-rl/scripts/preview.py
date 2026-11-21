"""Save read-only Source / SDPose / GT / ROI input panels for all val pairs."""
import argparse
import json
import os
from pathlib import Path
import sys

REPO=Path(os.environ.get('PICTUREEDIT_REPO',str(Path(__file__).resolve().parents[4])))
sys.path.insert(0,str(REPO))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    from PIL import Image,ImageDraw,ImageOps
    from src.data.canvas_contract import CanvasSpec
    pairs=[r for r in json.loads((a.root/'data/evaluation.json').read_text()) if r['split']=='val']
    rows=[]
    for r in pairs:
        source=CanvasSpec.from_record(r['source_canvas']);target=CanvasSpec.from_record(r['target_canvas'])
        with Image.open(r['source_rgb']['path']) as image:s=source.image(image.convert('RGB'))
        with Image.open(r['target_rgb']['path']) as image:g=target.image(image.convert('RGB'))
        condition=json.loads((a.root/f'conditions/{r["pair_id"]}.json').read_text())
        with Image.open(condition['images'][1]['file']['path']) as image:guide=image.convert('RGB')
        with np.load(r['region']['path'],allow_pickle=False) as region:
            overlay=np.asarray(g,np.float32)*.55
            overlay[:,:,1]+=region['person'][0]*55
            overlay[:,:,:2]+=region['joint'][0,:,:,None]*75
            overlay[:,:,0]+=region['hands'][0]*160
            overlay[:,:,2]+=region['hands'][1]*160
        mask=Image.fromarray(np.uint8(np.clip(overlay,0,255)))
        strip=Image.new('RGB',(960,450),'#171c22');draw=ImageDraw.Draw(strip)
        for col,(label,img) in enumerate((('Source',s),('SDPose condition',guide),('Paired GT',g),('Region supports',mask))):
            tile=ImageOps.contain(img,(236,408));strip.paste(tile,(240*col+(240-tile.width)//2,28))
            draw.text((240*col+5,5),label,fill='white')
        draw.text((5,434),r['frame_id']+' | '+r['pair_id'][:12],fill='white');rows.append(strip)
    panel=Image.new('RGB',(960,450*len(rows)))
    for i,row in enumerate(rows):panel.paste(row,(0,450*i))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:panel.save(f,format='PNG')
    print(json.dumps({'path':str(a.output),'validation_pairs':len(rows),'target_rgb_used_as_condition':False}))


if __name__=='__main__':main()
