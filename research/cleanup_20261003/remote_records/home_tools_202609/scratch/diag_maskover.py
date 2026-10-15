import glob
import numpy as np
from PIL import Image

SIZE = 512
SRC = {'pA': '/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-A',
       'pB': '/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-B'}
for n in ('pA', 'pB'):
    d = np.load(f'/home/shangguanrz/project/pic-edit/experiments/16-test/conditions_ab/{n}/conditions.npz')
    mask = d['human_mask'] > 0.5
    p = glob.glob(f'{SRC[n]}/*.jpg')[0]
    im = Image.open(p).convert('RGB')
    s = min(SIZE/im.width, SIZE/im.height)
    ow, oh = round(im.width*s), round(im.height*s)
    letter = Image.new('RGB', (SIZE, SIZE), (255,255,255))
    letter.paste(im.resize((ow,oh), Image.LANCZOS), ((SIZE-ow)//2, (SIZE-oh)//2))
    a = np.asarray(letter).astype(np.float32)
    a[mask] = a[mask] * 0.45 + np.array([255, 30, 30], np.float32) * 0.55
    Image.fromarray(a.clip(0,255).astype(np.uint8)).save(f'/tmp/mask_{n}.png')
    print(f'{n}: mask 覆盖 {int(mask.sum())} px  ({100*mask.mean():.1f}% 画面)')
