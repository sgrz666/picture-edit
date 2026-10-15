import glob, os
from PIL import Image, ImageDraw
TEST = '/home/shangguanrz/project/pic-edit/experiments/16-test'
B1, B2 = f'{TEST}/stage1_baselines/images', f'{TEST}/stage2_adapters/images'
H = 410
KEYS = ['zh_simple','zh_complex','en_simple','en_complex']
def ld(*paths):
    for p in paths:
        if os.path.isfile(p):
            im = Image.open(p).convert('RGB')
            return im.resize((max(1,round(im.width*H/im.height)), H), Image.LANCZOS)
    return Image.new('RGB',(round(H*0.6),H),(240,240,240))
A = ld(glob.glob(f'{TEST}/source/p-A/*.jpg')[0]); B = ld(glob.glob(f'{TEST}/source/p-B/*.jpg')[0])
rows = [
 ('REFERENCE', [A,B], ['A','B']),
 ('DeepGen 基干', [ld(f'{B1}/deepgen_base_{k}_cropped.png') for k in KEYS], KEYS),
 ('DeepGen + adapter  [refs=source]', [ld(f'{B2}/deepgen_adapter_{k}_raw.png') for k in KEYS], KEYS),
 ('DeepGen + adapter  [refs=both]', [ld(f'{B2}/deepgen_adapter_both_{k}_raw.png') for k in KEYS], KEYS),
 ('Qwen 基干', [ld(f'{B1}/qwen_base_{k}_raw.png') for k in KEYS], KEYS),
 ('Qwen + adapter  [refs=source]', [ld(f'{B2}/qwen_adapter_source_{k}_raw.png', f'{B2}/qwen_adapter_{k}_raw.png') for k in KEYS], KEYS),
 ('Qwen + adapter  [refs=both]', [ld(f'{B2}/qwen_adapter_both_{k}_raw.png') for k in KEYS], KEYS),
]
try:
    from PIL import ImageFont
    F1 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 17)
    F2 = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
except Exception:
    F1 = F2 = None
cw = max(max(sum(i.width+8 for i in ims) for _,ims,_ in rows)+24, 1100)
c = Image.new('RGB', (cw, sum(H+44 for _ in rows)+16), (255,255,255)); d = ImageDraw.Draw(c); y=8
for t, ims, lbs in rows:
    d.text((10,y), t, fill=(150,30,30), font=F1); y+=22; x=10
    for im, l in zip(ims, lbs):
        c.paste(im,(x,y)); d.rectangle([x,y,x+im.width,y+H], outline=(160,160,160))
        d.text((x+3,y+H+3), l, fill=(30,30,30), font=F2); x+=im.width+8
    y+=H+44
c.save(f'{TEST}/stage2_adapters/grid_full.png', optimize=True); print('wrote', c.size, 'rows', len(rows))
