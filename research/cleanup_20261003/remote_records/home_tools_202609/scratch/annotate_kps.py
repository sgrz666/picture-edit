import numpy as np
from PIL import Image, ImageDraw

H = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1'
rgb = Image.open(f'{H}/raw/pair00/hug00/images/4/000108.jpg').convert('RGB')
W, Hh = rgb.size
print('RGB size', rgb.size)

for tag, color in (('A', (255, 60, 60)), ('B', (60, 140, 255))):
    d = np.load(f'{H}/native_interaction_v2/dwpose_native/pair00/hug00/000108_{tag}.npz', allow_pickle=True)
    xy, sc = d['xy_native'], d['scores']
    dr = ImageDraw.Draw(rgb)
    for i in range(25):
        x, y = float(xy[i, 0]), float(xy[i, 1])
        if not (0 <= x < W and 0 <= y < Hh):
            continue
        s = 4 if sc[i] > 0.3 else 2
        dr.ellipse([x - s, y - s, x + s, y + s], outline=color, width=2)
        dr.text((x + 5, y - 7), str(i), fill=color)
rgb.save('/tmp/kp_annotated.png')
print('wrote /tmp/kp_annotated.png')
