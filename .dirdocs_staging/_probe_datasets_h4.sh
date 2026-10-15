PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
H=/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1
A=$H/processed_512_1500/pair01/fight01
echo "== channel subdir structure"
for c in rgb normal depth pose person_mask person_id scene_person_id smpl_mask occlusion; do echo "-- $c"; ls $A/$c | grep -v '\.md$' | tr '\n' ' '; echo; for sub in $(find $A/$c -maxdepth 1 -type d | tail -n +2); do echo "   $(basename $sub): $(find $sub -type f ! -name '*.md' | wc -l) files: $(ls $sub | grep -v '\.md$' | head -2 | tr '\n' ' ')"; done; done
echo "== npy shapes"
$PY -c "
import numpy as np, glob
for pat in ['normal/A_npy/*','depth/A_npy/*','pose/A_npy/*','pose/A/*','occlusion/A_by_B/*','normal/A/*']:
    fs=sorted(glob.glob('$A/'+pat))
    fs=[f for f in fs if not f.endswith('.md')]
    if not fs: print(pat,'NONE'); continue
    f=fs[0]
    if f.endswith('.npy'):
        a=np.load(f); print(pat, f.split('/')[-1], a.shape, a.dtype, float(a.min()), float(a.max()))
    else:
        from PIL import Image; im=Image.open(f); print(pat, f.split('/')[-1], im.size, im.mode)
"
echo "== person_id png values"
$PY -c "
import numpy as np
from PIL import Image
for p in ['$A/person_id/000001.png','$A/scene_person_id/000001.png','$A/person_mask/'+__import__('os').listdir('$A/person_mask')[0]+'/000001.png' if __import__('os').path.isdir('$A/person_mask/'+__import__('os').listdir('$A/person_mask')[0]) else '$A/person_mask/000001.png']:
    try:
        a=np.array(Image.open(p)); print(p, a.shape, a.dtype, sorted(set(a.reshape(-1).tolist()))[:6])
    except Exception as e: print(p,'ERR',e)
"
echo "== is rgb hardlinked to raw images?"
stat -c '%h %i %s' $A/rgb/000001.png $H/raw/pair01/fight01/images/52/000001.jpg
echo "== frame counts across processed"; cat $A/sequence_meta.json | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['num_frames'],d['camera_id'])"
echo "== total processed frames"; find $H/processed_512_1500 -path '*/rgb/*' -name '*.png' | wc -l
echo "== sequence_meta count"; find $H/processed_512_1500 -name 'sequence_meta.json' | wc -l
echo "== num_frames sum"
$PY -c "
import glob,json
s=glob.glob('$H/processed_512_1500/*/*/sequence_meta.json')
tot=0; sp={}
for f in s:
    d=json.load(open(f)); tot+=d['num_frames']; sp[d['split']]=sp.get(d['split'],0)+d['num_frames']
print('seqs',len(s),'frames',tot,'by split',sp)
print('iou sample min', min(json.load(open(f)).get('alignment_iou',1) for f in s))
"
