import numpy as np
from PIL import Image

SIZE = 512
DEMO = '/tmp/demo_{}.jpg'
COND = '/home/shangguanrz/project/pic-edit/experiments/16-test/conditions_ab'
def letterbox(im):
    s = min(SIZE/im.width, SIZE/im.height)
    ow, oh = round(im.width*s), round(im.height*s)
    out = Image.new('RGB', (SIZE, SIZE), (255,255,255))
    out.paste(im.resize((ow,oh), Image.LANCZOS), ((SIZE-ow)//2, (SIZE-oh)//2))
    return out
for n in ('pA', 'pB'):
    demo = letterbox(Image.open(DEMO.format(n)).convert('RGB'))
    mask = np.load(f'{COND}/{n}/conditions.npz')['human_mask'] > 0.5
    a = np.asarray(demo).astype(np.float32)
    a[mask] = a[mask]*0.55 + np.array([255,40,40], np.float32)*0.45
    Image.fromarray(a.clip(0,255).astype(np.uint8)).save(f'/tmp/vs_{n}.png')
    print(f'{n}: mask={int(mask.sum())}')
