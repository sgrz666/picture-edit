#!/usr/bin/env python3
"""Draw the DeepGen-V6 vs Qwen21 adapter architecture comparison."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 1000
im = Image.new('RGB', (W, H), (252, 252, 253))
d = ImageDraw.Draw(im)

FP = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
      '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
FPr = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']


def f(sz, bold=False):
    for p in (FP if bold else FPr):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


F_T, F_H, F_B, F_S = f(24, True), f(17, True), f(15), f(13)

C_SHARED = (228, 238, 250)
C_V6 = (232, 246, 232)
C_QW = (252, 236, 232)
C_EDGE = (110, 120, 135)
C_HI = (200, 60, 60)


def box(x, y, w, h, title, lines, fill, edge=C_EDGE, title_col=(20, 30, 45)):
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=edge, width=2)
    d.text((x + 10, y + 8), title, font=F_H, fill=title_col)
    for i, ln in enumerate(lines):
        d.text((x + 10, y + 32 + i * 19), ln, font=F_B, fill=(55, 60, 70))


def arrow(x0, y0, x1, y1, col=C_EDGE, label=None, dash=False):
    if dash:
        n = 14
        for i in range(n):
            if i % 2:
                continue
            t0, t1 = i / n, (i + 1) / n
            d.line([x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                    x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1], fill=col, width=2)
    else:
        d.line([x0, y0, x1, y1], fill=col, width=2)
    ax, ay = (x1 - x0), (y1 - y0)
    L = max((ax * ax + ay * ay) ** 0.5, 1)
    ux, uy = ax / L, ay / L
    d.polygon([(x1, y1), (x1 - 11 * ux + 5 * uy, y1 - 11 * uy - 5 * ux),
               (x1 - 11 * ux - 5 * uy, y1 - 11 * uy + 5 * ux)], fill=col)
    if label:
        d.text(((x0 + x1) / 2 + 6, (y0 + y1) / 2 - 9), label, font=F_S, fill=(90, 60, 60))


d.text((24, 18), 'DeepGen-V6  vs  Qwen-Image-2.1 adapter  —  interface migration', font=F_T,
       fill=(25, 30, 40))
d.text((24, 50), 'shared geometry front-end is reused verbatim; only the injection stage is re-implemented',
       font=F_B, fill=(110, 115, 125))

# ---------- shared front end ----------
d.text((24, 92), 'SHARED  (same classes, same V6 weights — loaded by load_v6_geometry)', font=F_H,
       fill=(40, 90, 150))
sy = 120
box(24, sy, 300, 88, 'SMPLXConditionInjector', ['normal / part_onehot / heatmap', 'smplx_global / mask  ->  bundle',
                                                '179 tensors transferred'], C_SHARED)
box(348, sy, 250, 88, 'SMPLXAdapterReasoner', ['cross-person reasoning', '148 tensors transferred'], C_SHARED)
box(622, sy, 300, 88, 'ReasonerControlBridge', ['geometry_scene [B,128,h,w]', '5 tensors transferred'], C_SHARED)
arrow(324, sy + 44, 346, sy + 44)
arrow(598, sy + 44, 620, sy + 44)

# ---------- fork ----------
fx, fy = 922, sy + 44
d.line([fx, fy, fx + 40, fy], fill=C_EDGE, width=2)
d.line([fx + 40, fy, fx + 40, 300], fill=C_EDGE, width=2)
d.line([fx + 40, fy, fx + 40, 640], fill=C_EDGE, width=2)
arrow(fx + 40, 300, 700, 300, label='geometry_scene')
arrow(fx + 40, 640, 700, 640, label='geometry_scene')

# ---------- V6 branch ----------
d.text((700, 176), 'A.  DeepGen V6  (SD3 MMDiT backbone, 24 layers, hidden 1536)', font=F_H,
       fill=(40, 120, 60))
box(700, 206, 380, 108, 'SharedRecurrentControlCore', [
    '1 shared DeepGen block  (transformer_blocks[0])',
    '3 branches x 6 stages:  geometry / interaction / detail',
    'each stage: LowRankStageAdapter(rank=64)',
    '            + CrossNorm + zero-init head'], C_V6)
box(1110, 206, 360, 108, 'DeepGenControlOutput', [
    '6 residual groups  [B, N, 1536]',
    '-> pipeline kwarg:',
    '   block_controlnet_hidden_states',
    '(native diffusers channel)'], C_V6)
arrow(1080, 260, 1108, 260)
box(1110, 334, 360, 96, 'injection scope', [
    'ALL image tokens',
    '6 stages cover all 24 blocks',
    'control core = a small DeepGen',
    'block, reusing backbone weights'], (255, 255, 255), edge=(40, 120, 60), title_col=(40, 120, 60))

# ---------- Qwen branch ----------
d.text((700, 516), 'B.  Qwen-Image-2.1  (single-stream, 32 layers, hidden 4096)', font=F_H,
       fill=(170, 70, 45))
box(700, 546, 380, 108, 'Qwen21BodyAdapter', [
    'geometry_in  Conv2d(128 -> width=512)',
    'hidden_in    Linear(4096 -> 512)',
    'time_in MLP(sigma)  + stage_embedding(6)',
    'fusion MLP  ->  6 x Linear(512 -> 4096)'], C_QW)
box(1110, 546, 360, 108, 'Qwen21ControlContext', [
    'register_forward_hook on',
    '  transformer_blocks[3,8,13,18,23,28]',
    'prefix = len(output) - target_tokens',
    'only output[:, prefix:] is modified'], C_QW)
arrow(1080, 600, 1108, 600)
box(1110, 674, 360, 118, 'injection scope', [
    'TARGET TOKENS ONLY',
    'reference-image prefix untouched',
    'target_tokens read at runtime from',
    '  img_shapes[0][-1]  (h*w)',
    '-> reference KV stays intact'], (255, 255, 255), edge=(170, 70, 45), title_col=(170, 70, 45))

# ---------- why ----------
d.rounded_rectangle([24, 300, 660, 500], radius=8, fill=(255, 250, 236), outline=(200, 160, 60), width=2)
d.text((40, 314), 'Why the injection scope MUST change', font=F_H, fill=(150, 100, 20))
for i, ln in enumerate([
    '',
    'SD3 / MMDiT  (V6):',
    '  hidden_states (image) and encoder_hidden_states (text)',
    '  are two SEPARATE inputs joined by attention inside a block.',
    '  A ControlNet residual on the image stream never touches text.',
    '  -> injecting on ALL image tokens is safe.',
    '',
    'Qwen-Image-2.1  (single stream):',
    '  reference latent and target latent are CONCATENATED into',
    '  ONE hidden_states sequence:  [source_latent ; noisy_target]',
    '  Injecting on all tokens would corrupt the reference KV,',
    '  and the appearance reference stops working.',
    '  -> must slice the tail and re-concatenate.',
]):
    d.text((40, 340 + i * 19), ln, font=F_B, fill=(70, 60, 40))

# ---------- loader ----------
d.rounded_rectangle([24, 516, 660, 700], radius=8, fill=(244, 246, 250), outline=C_EDGE, width=2)
d.text((40, 530), 'load_v6_geometry()  —  tri-filter transfer', font=F_H, fill=(40, 60, 90))
for i, ln in enumerate([
    '  keep name if:  prefix in {condition_injector.,',
    '                            reasoner., condition_bridge.}',
    '             AND  name exists in the new module',
    '             AND  shapes are EQUAL',
    '',
    'TRANSFERRED      condition_injector 179  reasoner 148',
    '                 condition_bridge 5',
    'AUTO-EXCLUDED    control_interface 135  (DeepGen-only)',
    '                 appearance_token_encoder / person_token_binder',
    '                 geometry_token_projection',
    '                 + geometry_in / hidden_in / zero_heads',
    '                 (1536 vs 4096 shape mismatch)',
]):
    d.text((40, 556 + i * 19), ln, font=F_B, fill=(50, 60, 75))

d.rounded_rectangle([24, 716, 1470, 786], radius=8, fill=(255, 244, 244), outline=C_HI, width=2)
d.text((40, 728), 'Leftover worth knowing about (harmless):', font=F_H, fill=(170, 50, 50))
for i, ln in enumerate([
    '  ReasonerControlBridge(token_dim=1536) is still hard-coded in qwen21 even though the Qwen hidden size is 4096.',
    '  Inspecting condition_bridge.py: token_dim is only stored as an attribute and never used to build any layer',
    '  (only geometry_projection / interaction_projection are built, from reasoning_dim -> geometry_channels).',
    '  That is why the 5 bridge tensors transfer cleanly despite the mismatch. Misleading, not a bug.',
]):
    d.text((40, 752 + i * 16), ln, font=F_S, fill=(120, 45, 45))

im.save('/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/adapter_arch_compare.png',
        optimize=True)
print('wrote adapter_arch_compare.png', im.size)
