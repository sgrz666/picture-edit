PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
D=/home/shangguanrz/project/pic-edit/datasets/iPER/iper_assets_512_nvdiffrast
echo "== manifest_512 wc + first line keys"
wc -l $D/manifest_512.jsonl
$PY -c "
import json
l=open('$D/manifest_512.jsonl').readline(); d=json.loads(l)
def walk(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v,p+'.'+k)
    elif isinstance(o,list): print(p,'list len',len(o), str(o[:2])[:120])
    else: print(p,'=',str(o)[:110])
walk(d)
"
echo "== _ASSETS_DONE.json keys"
$PY -c "
import json
d=json.load(open('$D/024_6/_ASSETS_DONE.json'))
print(list(d.keys()))
s=json.dumps(d,indent=1)
print(s[:2500])
"
echo "== channel file counts in 024_6"
for c in rgb_512 normal_512 depth_npy_512 depth_vis_512 dwpose_512 mask_512 part_512; do echo -n "$c: "; find $D/024_6/$c -type f ! -name '*README*' ! -name '目录*' | wc -l; done
echo "== npy shape"
F=$(find $D/024_6/depth_npy_512/target -name '*.npy' | head -1); echo $F
$PY -c "
import numpy as np; a=np.load('$F'); print(a.shape,a.dtype,a.min(),a.max())
"
echo "== normal png / mask png dims"
$PY -c "
from PIL import Image
import os
for c,ext in [('normal_512','png'),('mask_512','png'),('part_512','png'),('depth_vis_512','png'),('rgb_512','png'),('dwpose_512','png')]:
    p='$D/024_6/'+c+'/target'
    fs=[f for f in os.listdir(p) if f.endswith('.'+ext)]
    im=Image.open(p+'/'+fs[0]); print(c, im.size, im.mode, len(fs), 'files')
"
echo "== v6_conditions.pt"
$PY -c "
import torch
d=torch.load('$D/024_6/v6_conditions.pt',map_location='cpu',weights_only=False)
print(type(d))
if isinstance(d,dict):
  for k,v in d.items():
    try: print(k, tuple(v.shape), v.dtype)
    except Exception: print(k, type(v), str(v)[:200])
"
echo "== v6_detail_conditions.pt (keys only)"
$PY -c "
import torch
d=torch.load('$D/024_6/v6_detail_conditions.pt',map_location='cpu',weights_only=False)
print(type(d))
if isinstance(d,dict):
  for k,v in list(d.items())[:25]:
    try: print(k, tuple(v.shape), v.dtype)
    except Exception: print(k, type(v), str(v)[:150])
"
echo "== v65_face_features.pt"
$PY -c "
import torch
d=torch.load('$D/024_6/v65_face_features.pt',map_location='cpu',weights_only=False)
print(type(d))
if isinstance(d,dict):
  for k,v in list(d.items())[:15]:
    try: print(k, tuple(v.shape), v.dtype)
    except Exception: print(k, type(v), str(v)[:150])
"
echo "== du top files of 024_6"
du -sh --si $D/024_6/* | sort -h | tail -12
