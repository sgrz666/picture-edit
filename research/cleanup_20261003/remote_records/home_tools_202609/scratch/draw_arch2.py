#!/usr/bin/env python3
"""DeepGen-V6 vs Qwen21 adapter architecture — clean auto-sized layout."""
from PIL import Image, ImageDraw, ImageFont

W, H = 1620, 1360
im = Image.new('RGB', (W, H), (252, 252, 253))
d = ImageDraw.Draw(im)

FPr = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
FPb = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']


def fnt(sz, bold=False):
    for p in (FPb if bold else FPr):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            continue
    return ImageFont.load_default()


F_T, F_H, F_B, F_S = fnt(25, True), fnt(16, True), fnt(14), fnt(12.5)
LH, HEAD = 20, 30
EDGE = (120, 128, 140)


def box(x, y, w, title, lines, fill, edge=EDGE, tcol=(20, 30, 45), fs=F_B):
    h = HEAD + LH * len(lines) + 12
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill, outline=edge, width=2)
    d.text((x + 12, y + 7), title, font=F_H, fill=tcol)
    for i, ln in enumerate(lines):
        d.text((x + 12, y + HEAD + 4 + i * LH), ln, font=fs, fill=(52, 58, 68))
    return y + h


def arrow(x0, y0, x1, y1, col=EDGE, label=None):
    d.line([x0, y0, x1, y1], fill=col, width=2)
    ax, ay = x1 - x0, y1 - y0
    L = max((ax * ax + ay * ay) ** 0.5, 1)
    ux, uy = ax / L, ay / L
    d.polygon([(x1, y1), (x1 - 12 * ux + 5.5 * uy, y1 - 12 * uy - 5.5 * ux),
               (x1 - 12 * ux - 5.5 * uy, y1 - 12 * uy + 5.5 * ux)], fill=col)
    if label:
        d.text((x0 + 8, y0 - 17), label, font=F_S, fill=(95, 65, 65))


d.text((26, 16), 'DeepGen-V6  vs  Qwen-Image-2.1 adapter  —  how the interface was migrated', font=F_T,
       fill=(25, 30, 40))
d.text((26, 50), 'The geometry front-end is reused verbatim (same classes, same V6 weights). '
                 'Only the injection stage is re-implemented.', font=F_B, fill=(105, 112, 122))

# ---------------- shared row ----------------
d.text((26, 84), 'STAGE 1  ·  SHARED FRONT-END   (identical classes on both sides)', font=F_H,
       fill=(35, 85, 150))
y0 = 110
b1 = box(26, y0, 430, 'SMPLXConditionInjector', [
    'inputs: normal [B,3,H,W] / pose_heatmap [B,25,H,W]',
    '        part_onehot [B,14,H,W] / smplx_global [B,26]',
    '        human_mask [B,1,H,W] / task_id',
    'outputs: ConditionBundle',
    '>> 179 tensors transferred from V6'], (228, 238, 250))
arrow(456, y0 + 45, 486, y0 + 45)
b2 = box(486, y0, 340, 'SMPLXAdapterReasoner', [
    'cross-person reasoning over the bundle',
    'independent geometry / interaction paths',
    '>> 148 tensors transferred'], (228, 238, 250))
arrow(826, y0 + 45, 856, y0 + 45)
b3 = box(856, y0, 380, 'ReasonerControlBridge', [
    'geometry_projection  Conv2d(512 -> 128, 1)',
    'interaction_projection  Conv2d(512 -> 128, 1)',
    'outputs geometry_scene [B, 128, h, w]',
    '>> 5 tensors transferred'], (228, 238, 250))

fy = y0 + 45
d.line([1236, fy, 1290, fy], fill=EDGE, width=2)
d.line([1290, fy, 1290, 330], fill=EDGE, width=2)
d.line([1290, fy, 1290, 830], fill=EDGE, width=2)
arrow(1290, 330, 330, 330, label='geometry_scene [B,128,h,w]')
arrow(1290, 830, 330, 830, label='geometry_scene [B,128,h,w]')

# ---------------- V6 branch ----------------
d.text((26, 296), 'STAGE 2A  ·  DeepGen V6      SD3 MMDiT backbone · 24 blocks · hidden 1536',
       font=F_H, fill=(35, 115, 60))
ya = 322
ya = box(26, ya, 560, 'SharedRecurrentControlCore  (DeepGen-only)', [
    '1 SHARED DeepGen block  =  backbone.transformer_blocks[0]',
    '3 branches: geometry / interaction / detail',
    'each branch = 6 stages (num_stages=6)',
    '   per stage: LowRankStageAdapter(rank=64)',
    '              + CrossNorm',
    '              + zero-init Linear(hidden, hidden)',
    'so the control core is a small DeepGen block',
    'that reuses the backbone\u2019s own weights'], (232, 246, 232))
arrow(586, ya - 65, 620, ya - 65)
box(620, 322, 620, 'DeepGenControlOutput  ->  pipeline', [
    'block_controlnet_hidden_states =',
    '    6 residual groups  [B, N, 1536]',
    '',
    'injected via the NATIVE diffusers kwarg',
    '(ControlledDeepGenPipeline registers a',
    ' register_forward_pre_hook that sets',
    ' pipeline_kwargs["block_controlnet_',
    '                    hidden_states"])',
    '',
    'INJECTION SCOPE:  ALL image tokens',
    '6 stages cover all 24 blocks (~4 per stage)'], (232, 246, 232))

# ---------------- Qwen branch ----------------
d.text((26, 796), 'STAGE 2B  ·  Qwen-Image-2.1   single-stream · 32 blocks · hidden 4096',
       font=F_H, fill=(170, 70, 45))
yb = 822
yb = box(26, yb, 560, 'Qwen21BodyAdapter  (no control core)', [
    'geometry_in   Conv2d(128 -> width=512, 1)      <= geometry_scene',
    'hidden_in     Linear(4096 -> 512, bias=False)   <= current block hidden',
    'time_in       MLP(1 -> 512)                     <= sigma',
    'stage_embedding  Embedding(6, 512)              <= which of the 6 layers',
    'fusion        LayerNorm + Linear(512->1024) + SiLU + Linear(1024->512)',
    'zero_heads    6 x Linear(512 -> 4096),  zero-initialised',
    '',
    'LAYER_INDICES = (3, 8, 13, 18, 23, 28)   out of 32 blocks'], (252, 236, 232))
arrow(586, yb - 65, 620, yb - 65)
box(620, 822, 620, 'Qwen21ControlContext  ->  manual hooks', [
    'register_forward_pre_hook  reads img_shapes, computes',
    '     target_tokens = img_shapes[0][-1][1] * [-1][2]',
    'register_forward_hook on transformer_blocks[i] for i in LAYER_INDICES:',
    '     prefix = output.shape[1] - target_tokens',
    '     controlled = output[:, prefix:] + strength * delta',
    '     return torch.cat((output[:, :prefix], controlled), dim=1)',
    '',
    'INJECTION SCOPE:  TARGET TOKENS ONLY',
    'the reference-image prefix is never modified'], (252, 236, 232))

# ---------------- notes ----------------
ny = 1040
box(26, ny, 760, 'Why the injection scope MUST change', [
    'SD3 / MMDiT (V6):  hidden_states (image) and encoder_hidden_states (text)',
    'are two SEPARATE inputs joined by attention inside a block. A ControlNet',
    'residual added to the image stream never touches the text. Injecting on ALL',
    'image tokens is therefore safe.',
    'Qwen-Image-2.1 is SINGLE-STREAM: reference latent and target latent are',
    'CONCATENATED into one hidden_states sequence  [source_latent ; noisy_target].',
    'Injecting on all tokens would corrupt the reference KV and the appearance',
    'reference would stop working -> must slice the tail and re-concatenate.'], (255, 250, 236),
    edge=(200, 160, 60), tcol=(150, 100, 20))

box(806, ny, 788, 'load_v6_geometry()  —  tri-filter transfer', [
    'keep a tensor if   prefix in {condition_injector. , reasoner. ,',
    '                               condition_bridge.}',
    '              AND  the name exists in the new module',
    '              AND  the shapes are EQUAL',
    '',
    'TRANSFERRED       condition_injector 179 | reasoner 148 | bridge 5',
    'AUTO-EXCLUDED     control_interface 135  (DeepGen-only structure)',
    '                  appearance_token_encoder / person_token_binder',
    '                  geometry_token_projection',
    '                  geometry_in / hidden_in / zero_heads (1536 vs 4096)',
    '',
    'Leftover (harmless): bridge is still built with token_dim=1536 while Qwen',
    'hidden is 4096. condition_bridge.py only STORES token_dim and never uses it',
    'to build a layer -> shapes match, transfer succeeds. Misleading, not a bug.'],
    (244, 246, 250), tcol=(40, 60, 90))

out = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/adapter_arch_compare.png'
im.save(out, optimize=True)
print('wrote', out, im.size)
