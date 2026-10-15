D=/home/shangguanrz/project/pic-edit/datasets/iPER
S=$D/iper_sampled_6src64tgt/024_6
echo "== 024_6 tree with counts"
for s in $(find $S -maxdepth 1 -type d | sort); do echo "$s : $(find $s -type f | wc -l) files"; done
echo "== sub files sample names"
ls $S/rgb_1024/target | head -4; ls $S/rgb_1024/target | wc -l
ls $S/dwpose_512/source_standard | head -3
ls $S/keypoints/target | head -3
ls $S/smplestx | head -20
echo "== npz keys"
python3 -c "
import numpy as np
a=np.load('$S/keypoints/target/target_f000050.npz')
for k in a.files: print(k, a[k].shape, a[k].dtype)
"
echo "== candidate_cache npz"
python3 -c "
import numpy as np
for f in ['motion_candidates.npz','source_Apose_candidates.npz']:
    a=np.load('$S/candidate_cache/'+f)
    print(f, [(k,a[k].shape,a[k].dtype) for k in a.files][:12])
"
echo "== smplestx npz sample"
ls $S/smplestx | sed -n 1,12p
F=$(find $S/smplestx -name '*.npz' | head -1); echo "FILE $F"
python3 -c "
import numpy as np
a=np.load('$F')
for k in a.files: print(k, a[k].shape, a[k].dtype)
"
echo "== hardlink check inode"
stat -c '%h %s %n' $S/rgb_1024/target/$(ls $S/rgb_1024/target | head -1)
echo "== images sizes"
python3 -c "
from PIL import Image
import glob
for p in ['$S/rgb_1024/target/'+sorted(__import__('os').listdir('$S/rgb_1024/target'))[0]]:
    im=Image.open(p); print(p, im.size, im.mode)
"
echo "== manifest pair_count/unique"
python3 -c "
import json
d=json.load(open('$S/manifest.json'))
print('pair_count',d['pair_count'],'sources',len(d['sources']),'standard',len(d['standard_sources']),'arbitrary',len(d['arbitrary_sources']),'targets',len(d['targets']),'unique_selected_frames',len(d['unique_selected_frames']))
print('target keys sample', list(d['targets'][0].keys()))
"
