PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
D=/home/shangguanrz/project/pic-edit/datasets/iPER
S=$D/iper_sampled_6src64tgt/024_6
echo "== npz keys (keypoints)"
$PY -c "
import numpy as np
a=np.load('$S/keypoints/target/target_f000057.npz')
for k in a.files: print(k, a[k].shape, a[k].dtype)
"
echo "== candidate_cache npz"
$PY -c "
import numpy as np
for f in ['motion_candidates.npz','source_Apose_candidates.npz']:
    a=np.load('$S/candidate_cache/'+f)
    print(f, [(k,a[k].shape,a[k].dtype) for k in a.files])
"
echo "== smplestx params.pt"
ls -la $S/smplestx
$PY -c "
import torch
d=torch.load('$S/smplestx/params.pt',map_location='cpu',weights_only=False)
print(type(d), list(d.keys())[:20] if isinstance(d,dict) else d)
for k,v in d.items():
    try: print(k, tuple(v.shape), v.dtype)
    except Exception: print(k, type(v), str(v)[:100])
"
echo "== frame_map.json head"; head -c 400 $S/smplestx/frame_map.json; echo
echo "== params_schema.json head"; head -c 700 $S/smplestx/params_schema.json; echo
echo "== image dims"
$PY -c "
from PIL import Image
for p in ['$S/rgb_1024/target/target_f000057.png','$S/dwpose_512/target/'+sorted([x for x in __import__('os').listdir('$S/dwpose_512/target') if x.endswith('.png')])[0]]:
    im=Image.open(p); print(p, im.size, im.mode)
"
echo "== overlay sample name"; ls $S/overlay_512/target | grep -v md | head -3
echo "== per-subdir counts of each appearance (sample 3)"
for a in 001_1 024_6 029_1; do echo "$a: $(find $D/iper_sampled_6src64tgt/$a -type f -name '*.png' | wc -l) png, $(find $D/iper_sampled_6src64tgt/$a -type f -name '*.npz' | wc -l) npz, subdirs: $(ls $D/iper_sampled_6src64tgt/$a | grep -v md | tr '\n' ' ')"; done
echo "== du of one appearance"; du -sh --si $D/iper_sampled_6src64tgt/024_6
echo "== hardlink / inode check"
stat -c '%h %i %s' $S/rgb_1024/target/target_f000057.png
