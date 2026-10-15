import glob
import numpy as np
from PIL import Image

SIZE = 512
COND = '/home/shangguanrz/project/pic-edit/experiments/16-test/conditions_ab'
SRCA = glob.glob('/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-A/*.jpg')[0]
for tag, name in (('A 自身', 'pA'), ('B 重框到 A 机位', 'pB')):
    d = np.load(f'{COND}/{name}/conditions.npz')
    mask = d['human_mask'] > 0.5
    im = Image.open(SRCA).convert('RGB')          # 都叠在 A 的照片上
    s = min(SIZE/im.width, SIZE/im.height)
    ow, oh = round(im.width*s), round(im.height*s)
    letter = Image.new('RGB', (SIZE, SIZE), (255,255,255))
    letter.paste(im.resize((ow,oh), Image.LANCZOS), ((SIZE-ow)//2, (SIZE-oh)//2))
    a = np.asarray(letter).astype(np.float32)
    col = np.array([255, 30, 30], np.float32) if name == 'pA' else np.array([30, 90, 255], np.float32)
    a[mask] = a[mask] * 0.5 + col * 0.5
    Image.fromarray(a.clip(0,255).astype(np.uint8)).save(f'/tmp/rf_{name}.png')
    print(f'{tag}: {int(mask.sum())} px')
