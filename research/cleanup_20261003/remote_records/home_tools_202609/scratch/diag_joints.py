import glob
import numpy as np
from PIL import Image, ImageDraw

SIZE = 512
SRC = {'pA': '/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-A',
       'pB': '/home/shangguanrz/project/pic-edit/experiments/16-test/source/p-B'}
BONES = [(0,1),(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(10,11),
         (8,12),(12,13),(13,14),(0,15),(0,16),(11,24),(11,22),(14,21),(14,19)]
for n in ('pA', 'pB'):
    d = np.load(f'/home/shangguanrz/project/pic-edit/experiments/16-test/conditions_ab/{n}/conditions.npz')
    kp, cf = d['kp'], d['conf']
    p = glob.glob(f'{SRC[n]}/*.jpg')[0]
    im = Image.open(p).convert('RGB')
    s = min(SIZE/im.width, SIZE/im.height)
    ow, oh = round(im.width*s), round(im.height*s)
    letter = Image.new('RGB', (SIZE, SIZE), (255,255,255))
    letter.paste(im.resize((ow,oh), Image.LANCZOS), ((SIZE-ow)//2, (SIZE-oh)//2))
    dr = ImageDraw.Draw(letter)
    for i, j in BONES:
        if cf[i] > 0 and cf[j] > 0:
            dr.line([tuple(kp[i]), tuple(kp[j])], fill=(255,0,0), width=3)
    for i in range(25):
        if cf[i] > 0:
            x, y = kp[i]
            dr.ellipse([x-4,y-4,x+4,y+4], fill=(0,200,255), outline=(0,0,0))
    letter.save(f'/tmp/joint_{n}.png')
    print(f'{n}: 有效关节 {int((cf>0).sum())}/25')
