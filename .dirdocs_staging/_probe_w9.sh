cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== git ls-files data inputs logs caches outputs ==="
git ls-files data inputs logs caches outputs | head -20
echo "=== data/README.md (handwritten?) ==="
cat data/README.md
echo "=== inputs/README.md ==="
head -5 inputs/README.md
echo "=== caches/README.md ==="
head -5 caches/README.md
echo "=== outputs/README.md ==="
head -5 outputs/README.md
echo "=== logs/README.md ==="
head -5 logs/README.md
echo "=== iper_debug_001 detail ==="
find data/iper_debug_001 -maxdepth 2 -printf '%y %s %TY-%Tm-%Td %p\n' | head -30
echo "-- pairs.json"; cat data/iper_debug_001/pairs.json
echo "-- controls ls"; ls data/iper_debug_001/controls | head
echo "-- images ls"; ls data/iper_debug_001/images | head
echo "=== data pairs.json head ==="
head -c 1500 data/debug_samples/pairs.json
echo "=== input image sizes ==="
python3 -c "
from PIL import Image
for p in ('inputs/person1.jpg','inputs/dual_person.jpg','data/ref-02.png','data/champ_overfit/tgt_frame_0100.png','data/champ_overfit/ref-08.png','data/champ_overfit/dwpose/0001_all.png','data/champ_overfit/depth/0001_all.png','data/champ_overfit/normal/0001_all.png','data/champ_overfit/semantic_map/0001_all.png','data/debug_samples/images/sample_000_src.png'):
    im=Image.open(p); print(p, im.size, im.mode)
" 2>&1 | tail -20
echo "=== npy shape ==="
python3 -c "
import numpy as np
a=np.load('data/debug_samples/controls/sample_000_control8ch.npy'); print('ctrl', a.shape, a.dtype, a.min(), a.max())
" 2>&1 | tail -5
