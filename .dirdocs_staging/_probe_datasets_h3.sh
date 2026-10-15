PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
H=/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1
A=$H/processed_512_1500/pair01/fight01
echo "== sequence_meta.json"; head -c 1600 $A/sequence_meta.json; echo
echo "== channel file counts"
for c in rgb normal depth pose person_mask person_id scene_person_id smpl_mask occlusion; do echo -n "$c: "; find $A/$c -type f ! -name '*README*' ! -name '目录*' | wc -l; done
echo "== file names per channel"
for c in rgb normal depth pose person_id occlusion smpl_mask; do echo "$c -> $(ls $A/$c | grep -v -E 'md$' | head -2 | tr '\n' ' ')"; done
echo "== dims/dtypes"
$PY -c "
import numpy as np, os
from PIL import Image
A='$A'
for c in ['rgb','normal','depth','pose','person_mask','person_id','scene_person_id','smpl_mask','occlusion']:
    fs=[f for f in os.listdir(A+'/'+c) if not f.endswith('.md')]
    p=A+'/'+c+'/'+fs[0]
    if p.endswith('.png') or p.endswith('.jpg'):
        im=Image.open(p); print(c, im.size, im.mode, fs[0])
    else:
        a=np.load(p) if p.endswith('.npy') else np.load(p)
        print(c, a.shape, a.dtype, a.min(), a.max(), fs[0])
"
echo "== du per channel"; du -sh --si $A/* | sort -h
echo "== raw smpl npz keys"
$PY -c "
import numpy as np
a=np.load('$H/raw/pair01/fight01/smpl/000001.npz')
for k in a.files: print(k, a[k].shape, a[k].dtype)
"
echo "== raw meta.npz + cameras"
$PY -c "
import numpy as np
for f in ['$H/raw/pair01/fight01/meta.npz','$H/raw/pair01/fight01/cameras/rgb_cameras.npz']:
    a=np.load(f); print(f.split('/')[-2:], [(k,a[k].shape,a[k].dtype) for k in a.files])
"
echo "== raw images naming"; ls $H/raw/pair01/fight01/images/; ls $H/raw/pair01/fight01/images/52 | head -4; ls $H/raw/pair01/fight01/images/52 | wc -l
echo "== raw seg dirs"; ls $H/raw/pair01/fight01/seg; find $H/raw/pair01/fight01/seg -maxdepth 2 -type f -name '*.png' | wc -l
echo "== raw totals"; find $H/raw -type f ! -name '*.md' | wc -l; find $H/processed_512_1500 -type f ! -name '*.md' | wc -l
