Q=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
R=/data/shangguanrz
echo "### control pt dump"
$Q -c "
import torch
p='$R/pic-edit-caches/qwen21_hi4d_controls/control_000d4c3aa7eb1504623c181b.pt'
d=torch.load(p,map_location='cpu',weights_only=True)
print('TYPE',type(d),'NKEYS',len(d))
for k,v in d.items():
    try: print(k, tuple(v.shape), v.dtype, 'nonzero_elems', int(v.count_nonone()) if False else int(torch.count_nonzero(v)), 'bytes', v.numel()*v.element_size())
    except Exception as e: print(k, type(v), e)
" 2>&1 | head -40
echo "### du of that file"; du -h --apparent-size --si $R/pic-edit-caches/qwen21_hi4d_controls/control_000d4c3aa7eb1504623c181b.pt; stat -c '%s %b' $R/pic-edit-caches/qwen21_hi4d_controls/control_000d4c3aa7eb1504623c181b.pt
echo "### add_migration scripts"
head -50 $R/add_migration/add_pull_from_autodl.sh
echo "---- other.sh ----"; head -40 $R/add_migration/add_pull_other.sh
echo "---- log ----"; cat $R/add_migration/add_pull_other.log
echo "### SynBody presence"; ls -la $R/dataset_samples/ ; ls $R/dataset_samples/SynBody 2>&1
echo "### pku sample dirs"; find $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans -maxdepth 3 | head -60
echo "### pku counts"; find $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans/full_sample -type f | wc -l; du -sh --si $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans/full_sample $R/dataset_samples/abc_showcase_20261003/pku_dymvhumans/Dance_Dunhuang_Single_f14_sample
echo "### json 000075"; head -c 1098 $R/dataset_samples/MVHumanNet_repo/visual_smpl_data/000075.json; echo
echo "### intrinsics"; cat $R/dataset_samples/MVHumanNet_repo/visual_smpl_data/camera_intrinsics.json
echo "### text_description_24 keys"; $Q -c "
import json;d=json.load(open('$R/dataset_samples/MVHumanNet_repo/textual_description/text_description_24.json'));print(type(d),len(d));
import itertools
if isinstance(d,dict): print(list(itertools.islice(d,3)))
print(str(d)[:600])
"
echo "### sample_100_description keys"; $Q -c "
import json;d=json.load(open('$R/dataset_samples/MVHumanNet_repo/textual_description/sample_100_description.json'));print(type(d), len(d));print(str(d)[:700])
"
echo "### smplx_vis data"; ls -la $R/dataset_samples/MVHumanNet++_repo/smplx_visualization/data/
echo "### video probes"; which ffprobe; for f in $R/dataset_samples/abc_showcase_20261003/humanvid/HumanVid_pexels_12908928.mp4 $R/dataset_samples/abc_showcase_20261003/dna_rendering/DNA_sample_20220419_yuxinrui_f_yuxinrui_yf2_dz9.mp4; do ffprobe -v error -show_entries stream=width,height,nb_frames,duration -of default=nw=1 "$f" 2>&1 | head -8; done
echo "### image dims"; $Q -c "
from PIL import Image
import glob
for f in sorted(glob.glob('$R/dataset_samples/abc_showcase_20261003/*/*.jpg'))[:12]: print(f.split('/')[-1], Image.open(f).size)
for f in sorted(glob.glob('$R/dataset_samples/abc_showcase_20261003/*/*.png'))[:8]: print(f.split('/')[-1], Image.open(f).size)
"
