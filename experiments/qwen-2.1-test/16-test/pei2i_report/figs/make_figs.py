# -*- coding: utf-8 -*-
"""生成 PE-I2I 报告配图：流程示意图 + 重写过程可视化"""
import json, textwrap
from PIL import Image, ImageDraw, ImageFont

FZ, FZB = 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc'
FM, FMB = 'C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/consolab.ttf'
def f(sz, b=False, mono=False):
    try:
        return ImageFont.truetype((FMB if b else FM) if mono else (FZB if b else FZ), sz)
    except Exception:
        return ImageFont.load_default()

W = 1700
def canvas(h): return Image.new('RGB', (W, h), (252, 252, 253))

def box(d, xy, w, h, title, sub='', fill=(230,241,251), line=(24,95,165), tc=(4,44,83)):
    x, y = xy
    d.rounded_rectangle([x, y, x+w, y+h], radius=10, fill=fill, outline=line, width=2)
    if sub:
        d.text((x+w//2, y+h//2-13), title, font=f(22, True), fill=tc, anchor='mm')
        d.text((x+w//2, y+h//2+14), sub, font=f(17), fill=(70,80,95), anchor='mm')
    else:
        d.text((x+w//2, y+h//2), title, font=f(22, True), fill=tc, anchor='mm')

def arrow(d, p1, p2, col=(120,120,120), w=3):
    d.line([p1, p2], fill=col, width=w)
    import math
    ang = math.atan2(p2[1]-p1[1], p2[0]-p1[0]); L = 14
    for s in (-0.5, 0.5):
        d.line([p2, (p2[0]-L*math.cos(ang+s), p2[1]-L*math.sin(ang+s))], fill=col, width=w)

# ---------- Fig 1: 两段式流程 ----------
img = canvas(900); d = ImageDraw.Draw(img)
d.text((60, 34), '图 1　PE-I2I 自动重写的两段式流程', font=f(30, True), fill=(15,15,20))
d.text((60, 76), 'PE-I2I 是独立模型，跑在图像生成之前；基座 Qwen-Image-2.1 的管线与参数完全未改动',
       font=f(19), fill=(90,95,105))
y = 130
box(d, (60,  y), 400, 96, '阶段 0 · 输入', '图 A（身份来源）+ 图 B（姿态来源）+ 短指令')
box(d, (620, y), 460, 96, '阶段 1 · PE-I2I 重写', 'Qwen3.5-VL 9B 微调 · 视觉语言模型',
    fill=(238,237,254), line=(83,74,183), tc=(38,33,137))
arrow(d, (260, y+96), (260, y+150))
arrow(d, (850, y+96), (850, y+150))
# 输出 JSON
box(d, (60, y+150), 400, 96, 'rewritten_prompt', '本例中文 349 字 / 英文 2175 字符',
    fill=(225,245,238), line=(15,110,86), tc=(4,52,44))
box(d, (620, y+150), 460, 96, 'ratio_follow = "<image1>"', '官方对"输出继承谁构图"的判断',
    fill=(225,245,238), line=(15,110,86), tc=(4,52,44))
arrow(d, (260, y+246), (260, y+300))
box(d, (60, y+300), 400, 96, '阶段 2 · Qwen-Image-2.1 基座', '同一套管线 · 仅 prompt 字符串不同',
    fill=(238,237,254), line=(83,74,183), tc=(38,33,137))
arrow(d, (260, y+396), (260, y+450))
box(d, (60, y+450), 400, 76, '输出图像', '640×1088', fill=(234,243,222), line=(59,109,17), tc=(23,52,4))
# 右侧注解
d.rounded_rectangle([1130, y, 1640, y+546], radius=12, fill=(250,238,218), outline=(133,79,11), width=2)
d.text((1160, y+26), '关键区分', font=f(23, True), fill=(65,36,2))
for i, t in enumerate([
  'PE-I2I 做的是「文本生成」，',
  '不是出图。它不写入、不修改',
  'Qwen-Image-2.1 的任何权重。',
  '',
  '两次调用对比：',
  '  pipe(prompt=…, image=[A,B],',
  '       width=640, height=1088,',
  '       num_inference_steps=40,',
  '       true_cfg_scale=1.0, …)',
  '',
  '结构完全一致 —— 唯一的差异',
  '就是 prompt 那个字符串。',
  '',
  '所以它是纯粹的「前处理步骤」。',
]):
    d.text((1160, y+66+i*32), t, font=f(17, mono=(i in range(5,10))), fill=(80,55,20))
img.save('fig1_flow.png', optimize=True); print('fig1', img.size)

# ---------- Fig 2: 重写过程可视化 ----------
rows = json.load(open('D:/codeplus/pictureedit/.workbuddy/tmp/16test/pei2i/rewrites.json', encoding='utf-8'))
th = open('D:/codeplus/pictureedit/.workbuddy/tmp/16test/pei2i/rewrite_zh.txt', encoding='utf-8').read()
thinking = th[th.find('### thinking')+12: th.find('### answer')].strip()

img = canvas(1560); d = ImageDraw.Draw(img)
d.text((60, 34), '图 2　重写过程可视化（中文组）', font=f(30, True), fill=(15,15,20))
d.text((60, 76), '短指令 65 字 → 中间推理 8,086 字符 → 重写结果 349 字。关键：模型先「看写」两张图，再规划姿态迁移',
       font=f(19), fill=(90,95,105))
colw, gap, top = 500, 40, 140
xs = [60, 60+colw+gap, 60+2*(colw+gap)]
hdrs = [('① 输入 · 短指令', '65 字符', (250,238,218), (133,79,11), (65,36,2)),
        ('② 中间推理 · thinking', '8,086 字符（摘录）', (238,237,254), (83,74,183), (38,33,137)),
        ('③ 输出 · rewritten_prompt', '349 字符', (225,245,238), (15,110,86), (4,52,44))]
for x, (h1, h2, fl, ln, tc) in zip(xs, hdrs):
    d.rounded_rectangle([x, top, x+colw, top+62], radius=10, fill=fl, outline=ln, width=2)
    d.text((x+18, top+16), h1, font=f(21, True), fill=tc)
    d.text((x+18, top+40), h2, font=f(16), fill=(95,100,110))
zh = rows[0]['instruction']
body1 = textwrap.wrap(zh, 30)
zhout = rows[0]['rewritten_prompt']
body3 = textwrap.wrap(zhout, 30)[:14]
think_txt = ('The user\'s instruction is in Chinese, so the output should be in Chinese.\n\n'
             'Task: Change the pose of the person in <image1> to the pose of the person in '
             '<image2>, while strictly preserving the identity, appearance, clothing, and '
             'background of the person in <image1>.\n\n'
             'This is a pose/action transfer sub-type. Canvas = <image1>, source = <image2>.\n\n'
             'Let me analyze the images.\n\n'
             '<image1>: A man standing on a wet street... wearing a blue suit jacket (royal blue '
             'blazer), white dress shirt, dark blue striped tie, light gray trousers, black '
             'leather dress shoes. He wears glasses (rectangular frames), has short black hair '
             'slicked back, and a wristwatch on his left wrist...\n\n'
             '<image2>: A young man... wearing a black turtleneck sweater, white suspenders, '
             'gray checkered trousers... head turned to his left (viewer\'s right), looking at '
             'the basketball balanced on his left hand\'s fingertips...')
body2 = textwrap.wrap(think_txt, 40)[:22]
for x, body, mono in zip(xs, (body1, body2, body3), (False, True, False)):
    yy = top + 78
    for ln in body:
        d.text((x+18, yy), ln, font=f(16, mono=mono), fill=(45,50,60))
        yy += 27
note = {'0': '我的短指令里没有提眼镜、西装颜色、领带、手表 —— 这些全是模型从图里读出来的',
        '1': '模型明确说明：这是一次姿态迁移任务，画布取 <image1>',
        '2': '逐项列出后，重写里出现了原指令没有的「黑框眼镜」「深蓝条纹领带」「左手腕金属手表」'}
for i, x in enumerate(xs):
    d.text((x+18, top+78+len((body1, body2, body3)[i])*27+14), note[str(i)],
           font=f(15), fill=(170,60,30))
img.save('fig2_rewrite_process.png', optimize=True); print('fig2', img.size)
