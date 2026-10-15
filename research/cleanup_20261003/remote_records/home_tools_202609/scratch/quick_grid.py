import json, os
from PIL import Image, ImageDraw, ImageFont

TEST = '/home/shangguanrz/project/pic-edit/experiments/16-test'
B1 = f'{TEST}/stage1_baselines/images'
OUT = f'{TEST}/stage1_baselines/qwen_base_grid.png'
KEYS = [('zh_simple','中文','简易'),('zh_complex','中文','复杂'),
        ('en_simple','英文','简易'),('en_complex','英文','复杂')]
prompts = json.loads(open(f'{TEST}/prompts.json', encoding='utf-8').read())

H = 620
def ld(p):
    im = Image.open(p).convert('RGB')
    return im.resize((max(1,round(im.width*H/im.height)), H), Image.LANCZOS)

def font(sz, bold=False):
    p = ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
         else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    try:
        return ImageFont.truetype(p, sz)
    except Exception:
        return ImageFont.load_default()

F1, F2, F3 = font(20, True), font(14), font(12)
HEAD, PAD = 96, 14

ims = [(k, zh, cx, ld(f'{B1}/qwen_base_{k}_raw.png'), prompts[k]) for k, zh, cx in KEYS]
cw = sum(i[3].width + PAD for i in ims) + PAD
ch = HEAD + H + PAD
c = Image.new('RGB', (cw, ch), (255, 255, 255))
d = ImageDraw.Draw(c)
x = PAD
for k, zh, cx, im, pr in ims:
    d.text((x+2, 8), f'Qwen-Image-2.1  基干（无 adapter）', fill=(150,30,30), font=F1)
    d.text((x+2, 34), f'语言：{zh}    提示词：{cx}', fill=(20,20,20), font=F2)
    d.text((x+2, 54), '640x1088 · 40 步 · cfg 1.0 · seed 42', fill=(90,90,90), font=F3)
    d.text((x+2, 70), '参考图：[A, B] 多图 · 无几何条件 · KV cache', fill=(90,90,90), font=F3)
    c.paste(im, (x, HEAD))
    d.rectangle([x, HEAD, x+im.width, HEAD+H], outline=(150,150,150))
    x += im.width + PAD
c.save(OUT, optimize=True)
print('wrote', OUT, c.size)
