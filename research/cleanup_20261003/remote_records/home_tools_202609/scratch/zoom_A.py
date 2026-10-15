from PIL import Image, ImageDraw
BASE = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test'
import json
man = {p['pair']: p for p in json.load(open(f'{BASE}/data/testset/manifest.json'))['pairs']}
H = 700
ims = []
for nm in ('hard_01', 'hard_02'):
    im = Image.open(man[nm]['A_path']).convert('RGB')
    w = round(im.width * H / im.height)
    ims.append((nm, im.resize((w, H), Image.LANCZOS), im.size))
W = sum(i[1].width for i in ims) + 6 * (len(ims) + 1)
lab = Image.new('RGB', (W, H + 26), (255, 255, 255))
dr = ImageDraw.Draw(lab)
x = 6
for nm, im, orig in ims:
    lab.paste(im, (x, 26))
    dr.rectangle([x, 26, x + im.width, 26 + H], outline=(150, 150, 150))
    dr.text((x + 4, 8), f'{nm}  A_source  original {orig[0]}x{orig[1]}', fill=(10, 10, 10))
    x += im.width + 6
lab.save(f'{BASE}/experiment/results/A_zoom.png', optimize=True)
print('wrote A_zoom.png', lab.size)
