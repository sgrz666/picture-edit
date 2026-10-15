import numpy as np
from PIL import Image
import cv2

SIZE = 512
SRC = {'pA': '/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-A',
       'pB': '/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-B'}
import glob
for n in ('pA', 'pB'):
    d = np.load(f'/home/shangguanrz/project/pic-edit/experiments/16-test/conditions_ab/{n}/conditions.npz')
    mask = d['human_mask']; depth = d['depth']
    print(f'=== {n}: mask 像素={int((mask>0).sum())}  depth 非零={int((depth>0).sum())} '
          f'range=[{depth.min():.3f},{depth.max():.3f}]')
    p = glob.glob(f'{SRC[n]}/*.jpg')[0]
    im = Image.open(p).convert('RGB')
    s = min(SIZE/im.width, SIZE/im.height)
    ow, oh = round(im.width*s), round(im.height*s)
    letter = Image.new('RGB', (SIZE, SIZE), (255,255,255))
    letter.paste(im.resize((ow,oh), Image.LANCZOS), ((SIZE-ow)//2, (SIZE-oh)//2))
    arr = np.asarray(letter).copy()
    m = (mask > 0.5).astype(np.uint8)
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(arr, cnts, -1, (255,0,0), 3)
    Image.fromarray(arr).save(f'/tmp/overlay_{n}.png')
    print(f'   写出 overlay_{n}.png')
