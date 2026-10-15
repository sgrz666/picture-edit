"""Draw the Qwen-Image-2.1 pose-control architecture diagrams used in the walkthrough.

Outputs:
  qwen21_overview.png      - overall inference/training data flow
  qwen21_adapter_internals.png - adapter internals, full vs simplified comparison
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FROZEN = "#E8EEF7"
FROZEN_EDGE = "#4A6FA5"
TRAIN = "#FFF3DC"
TRAIN_EDGE = "#D99125"
COND = "#E5F3E1"
COND_EDGE = "#4E8B44"
FACE = "#FBE4EC"
FACE_EDGE = "#B0567F"
HAND = "#EDE4F7"
HAND_EDGE = "#7A4FA3"
OUT = "#E2F0F6"
OUT_EDGE = "#2E7C99"
DARK = "#22303F"


def box(ax, x, y, w, h, text, *, fc=FROZEN, ec=FROZEN_EDGE, fs=9.5, lw=1.4, align="center", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    weight = "bold" if bold else "normal"
    if align == "left":
        ax.text(x + 1.2, y + h - 1.4, text, ha="left", va="top", fontsize=fs,
                color=DARK, zorder=4, weight=weight, linespacing=1.45)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=DARK, zorder=4, weight=weight, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, *, color="#4A6FA5", lw=2.0, style="-|>", ls="-", shrink=2, zorder=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, linestyle=ls, shrinkA=shrink, shrinkB=shrink,
                                 zorder=zorder))


def label(ax, x, y, text, *, fs=8.2, color="#5A6B7C", ha="center", style="italic", weight="normal"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color, style=style, weight=weight,
            linespacing=1.35, zorder=5)


# ---------------------------------------------------------------- diagram 1
fig, ax = plt.subplots(figsize=(16.6, 9.4), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 97.2, "Qwen-Image-2.1 姿态编辑生图架构 —— 总体数据流",
        ha="center", va="center", fontsize=17, weight="bold", color=DARK)
ax.text(50, 93.4, "基座全程冻结，仅训练 SMPL-X 适配器；残差经 forward hook 注入，不改模型源码",
        ha="center", va="center", fontsize=10, color="#5A6B7C", style="italic")

# column A: inputs
box(ax, 1.5, 70, 15.5, 13, "源图像\n(源人物 RGB)", fc=COND, ec=COND_EDGE, fs=11, bold=True)
box(ax, 1.5, 41, 15.5, 17, "目标 SMPL-X\n姿态参数\n(双人/单人)", fc=COND, ec=COND_EDGE, fs=11, bold=True)
box(ax, 1.5, 12, 15.5, 11, "文本提示\n(固定指令)", fc=COND, ec=COND_EDGE, fs=11, bold=True)

# column B: encoders
box(ax, 24, 68.5, 19, 15,
    "VAE 编码器\n16ch → 2×2 patchify\n源 latent = 1024 token×64",
    fs=9.5, bold=True)
box(ax, 23.5, 38.5, 21, 20.5,
    "SMPL-X 条件渲染\n(nvdiffrast)\n· 法线图 (3ch)\n· 25 关键点热图\n· 14 类部位图\n· 接触图 / 全局 26D\n· 人体 mask",
    fc=COND, ec=COND_EDGE, align="left", fs=9)
box(ax, 24, 10.5, 19, 13,
    "Qwen2.5-VL 文本编码器\n(多模态，能读到源图)\n→ 文本嵌入 4096 维",
    fs=9.5, bold=True)

# column C: adapter
box(ax, 52, 33, 24, 37,
    "V6 SMPL-X 适配器\n（唯一可训练部分）\n\n条件注入器\n→ Reasoner 推理\n→ 控制桥 → 场景图\n→ 控制核 + Face/Hand\n   细节分支\n\n每个去噪步 σ 重新计算\n感知当前噪声与目标状态",
    fc=TRAIN, ec=TRAIN_EDGE, align="left", fs=9.5, bold=False)

# column D: DiT
box(ax, 82.5, 18.5, 15.5, 66, "", fc=FROZEN, ec=FROZEN_EDGE)
ax.text(90.25, 80.8, "Qwen-Image-2.1 DiT（冻结）", ha="center", fontsize=10.5, weight="bold", color=DARK)
ax.text(90.25, 77.6, "32 × Joint-Attention block\nhidden 4096", ha="center", fontsize=8.4, color="#4A6FA5")
ax.text(90.25, 74.2, "hidden_states = 拼接", ha="center", fontsize=8.0, color=DARK)
# token stream inside DiT
ax.add_patch(FancyBboxPatch((86.5, 57.5), 11.2, 14.5, boxstyle="square,pad=0", fc="#CFE0F0", ec="#9FB3C8", lw=1.0, zorder=4))
ax.text(92.1, 64.7, "源 latent 段\n参考 token 1024\n(原样保留，不加残差)", ha="center", va="center", fontsize=7.4, color="#2E5A85", zorder=5)
ax.add_patch(FancyBboxPatch((86.5, 39.5), 11.2, 16.5, boxstyle="square,pad=0", fc="#FBE8D3", ec="#C89A6B", lw=1.0, zorder=4))
ax.text(92.1, 47.7, "加噪目标段\n1024 token\n残差唯一作用区", ha="center", va="center", fontsize=7.4, color="#8A5A22", zorder=5)
for gy, gname in ((72.5, "G0"), (65.5, "G1"), (58.5, "G2"), (51.5, "G3"), (44.5, "G4"), (37.5, "G5")):
    ax.plot([83.4, 85.6], [gy, gy], color="#C2502B", lw=1.6, zorder=5)
    ax.text(82.9, gy, gname, ha="right", va="center", fontsize=7.6, color="#C2502B", weight="bold")
ax.text(90.25, 23.5, "blocks 0–31\n6 组残差注入点", ha="center", fontsize=8.2, color="#C2502B")

# bottom: noisy target / decode
box(ax, 52, 10.5, 24, 12,
    "加噪目标 latent  z_t\n训练: (1−σ)·z_tgt + σ·ε\n推理: 从纯噪声逐步去噪",
    fc=OUT, ec=OUT_EDGE, fs=9)
box(ax, 82.5, 3.5, 15.5, 11, "VAE 解码\n→ 输出图像", fc=OUT, ec=OUT_EDGE, fs=10, bold=True)

# arrows
arrow(ax, 17.2, 76.5, 23.8, 76.5)                     # src -> vae
arrow(ax, 17.2, 49.5, 23.3, 49.5)                     # smplx -> conditions
arrow(ax, 17.2, 17.5, 23.8, 17.5)                     # prompt -> vl
arrow(ax, 8.2, 69.8, 8.2, 26.2, ls="--", color="#7A93AC")   # src img down the left margin
arrow(ax, 8.2, 26.2, 25.5, 20.5, ls="--", color="#7A93AC")  # -> VL encoder top-left
label(ax, 11.5, 63.5, "源图喂给\nVL 编码器", fs=7.2)
arrow(ax, 43.2, 79.0, 80.8, 79.0, color="#2E5A85")    # source latent -> DiT ref segment (above adapter)
arrow(ax, 80.8, 79.0, 82.4, 75.5, color="#2E5A85")
label(ax, 63, 81.5, "源 latent 作为参考 token 拼接（Edit 式条件）", fs=7.6)
arrow(ax, 43.2, 19.5, 50.5, 27.5, color="#2E5A85")    # text embeds -> DiT (bent, 2 segments)
arrow(ax, 50.5, 27.5, 82.3, 31.5, color="#2E5A85")
label(ax, 58, 30.2, "文本嵌入（joint attention）", fs=7.6)
arrow(ax, 44.7, 49, 51.8, 49, color="#4E8B44", lw=2.4)  # conditions -> adapter
arrow(ax, 33.5, 68.3, 33.5, 59.2, ls="--", color="#7A93AC")
arrow(ax, 35.5, 59.2, 35.5, 68.3, ls="--", color="#7A93AC")
label(ax, 41.5, 63.6, "源 latent 亦进适配器\n(外观 token 绑定)", fs=7.2)
arrow(ax, 76.2, 51.5, 82.2, 51.5, color="#C2502B", lw=2.6)  # adapter -> DiT residuals
label(ax, 79.3, 55.8, "5 分支残差\n（仅加在目标段）", fs=7.4, color="#C2502B")
arrow(ax, 64, 22.7, 64, 32.8, color="#4E8B44")        # noisy target -> adapter (per-step state)
arrow(ax, 90.2, 18.3, 90.2, 15.0, color="#2E7C99")    # DiT -> decode
arrow(ax, 76.2, 14.5, 85.5, 17.8, color="#2E7C99")    # z_t -> DiT
ax.text(50, 2.0, "flow matching：模型预测速度场 v ≈ ε − z_tgt，沿 ODE 迭代 N 步（16-test 基线 40 步，验收常用 30 步）",
        ha="center", fontsize=8.4, color="#5A6B7C", style="italic")

fig.savefig(__file__.replace("draw_qwen21_arch.py", "qwen21_overview.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- diagram 2
fig, ax = plt.subplots(figsize=(17.2, 11.0), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.add_patch(FancyBboxPatch((0.5, 52.5), 99, 45, boxstyle="round,pad=0.3,rounding_size=1.0",
                            fc="#FFFDF6", ec="#D99125", lw=1.6, ls="--", zorder=1))
ax.add_patch(FancyBboxPatch((0.5, 2.0), 99, 46.5, boxstyle="round,pad=0.3,rounding_size=1.0",
                            fc="#F7FAFD", ec="#4A6FA5", lw=1.6, ls="--", zorder=1))
ax.text(50, 95.6, "完整版 qwen21_full —— 整套 V6 控制架构原样迁移，仅做 Qwen 边界桥",
        ha="center", fontsize=13.5, weight="bold", color="#8A5A22")
ax.text(50, 46.6, "简化版 qwen21 —— 轻量复用 V6 条件推理，直接在 Qwen 上建轻量残差头",
        ha="center", fontsize=13.5, weight="bold", color="#2E5A85")

# ---- full version row
box(ax, 2, 56, 18, 35,
    "① Qwen 边界桥（入口）\n\nlatent_bridge\n1×1 conv 64→64（恒等初始化）\n+ pixel_shuffle ×2\nQwen token 网格 → V6 双倍分辨率\n\ntext_bridge  4096 → ctx\npooled_bridge 4096 → 2048\n\n（内部保持 V6 的 1536 维\n表示完全不变）",
    fc=TRAIN, ec=TRAIN_EDGE, align="left", fs=8.2)
box(ax, 23, 56, 30, 35,
    "② 共享循环控制核 SharedRecurrentControlCore\n（SD3ControlNetModel 组件改装，内部 1536 维）\n\n3 条动态分支：geometry / interaction / detail\n每分支 6 个 stage，复用同一个 Joint Block（循环）\n每 stage：stage_emb → 共享 block\n→ LoRA 适配 + CrossNorm → 零初始化头\n→ 产出 1 组残差，共 6 组\n\n上下文：噪声目标 latent、σ、文本嵌入、\n外观 token（PersonTokenBinder 绑定 A/B）",
    fc=TRAIN, ec=TRAIN_EDGE, align="left", fs=8.2)
box(ax, 56, 56, 20, 35,
    "③ 精细分支\n\nFace v65：\nArcFace 512 + DINO patches\n+ 72 landmarks → Resampler\n+ Perceiver → 双人 2 个脸 ROI\n逆 warp 散射残差\n10/2 修复：按 512 训练网格计算，\n再平滑映射到实际尺寸并硬框限制\n\nHand v66：\n4 手 ROI × 21 关键点 + 45 维手姿\n+ mesh BPS；VTON 独立 KV 注意力",
    fc=FACE, ec=FACE_EDGE, align="left", fs=8.0)
box(ax, 79, 56, 19, 35,
    "④ 注入与强度调度\n\noutput_bridges\n5× Linear 1536→4096\n（继承头非零→桥零初始化，\n否则 xavier：只留一条零边界）\n\n32 个 block 全挂 hook：\nblock g ∈ 组 G(6g/32)，\n组残差只加到目标 token 段\n\n强度控制器按去噪进度调度：\ngeometry 1.0 / interaction 0.8\nface 仅 σ<0.65 激活\n纹理监督 σ<0.35",
    fc=TRAIN, ec=TRAIN_EDGE, align="left", fs=8.0)
arrow(ax, 20.3, 73.5, 22.7, 73.5, color="#D99125", lw=2.4)
arrow(ax, 53.3, 73.5, 55.7, 73.5, color="#D99125", lw=2.4)
arrow(ax, 76.3, 73.5, 78.7, 73.5, color="#D99125", lw=2.4)

# ---- simplified version row
box(ax, 2, 5, 31, 38,
    "① Body 适配器 Qwen21BodyAdapter\n\n复用 V6：条件注入器 + Reasoner + 控制桥\n→ geometry_scene (128ch) → 插值到目标网格\n→ Conv 128→512 展平为 token\n\n每个注入层（stage s，层索引 l）：\nstate = hidden_in(target) + geometry\n        + time_in(σ) + stage_embedding[s]\nΔ = zero_head[l]( fusion(state) )\n\n零初始化头 → 训练起点适配器开=关",
    fc=TRAIN, ec="#B8791E", align="left", fs=8.4)
box(ax, 36, 5, 27, 38,
    "② Face 适配器 Qwen21FaceAdapter\n\nArcFace 512 维身份向量\n+ 128² 源脸纹理 CNN（→512）\n→ code = identity + texture\n\n目标脸框（几何投影）→\nROI mask（脸外严格为 0）\n+ 归一化局部/全局坐标 → position\n\nΔ = zero_head(fusion(hidden + code\n      + position + time + layer_emb)) × mask\n\n不接触目标 RGB / 目标身份",
    fc=FACE, ec=FACE_EDGE, align="left", fs=8.4)
box(ax, 66, 5, 32, 38,
    "③ 注入与扩展\n\n同一组 hook，6 个注入层：\nblocks 3 / 8 / 13 / 18 / 23 / 28\n\n每层：target 段 += strength·Δ_body\n             += face_strength·Δ_face\n（层 ≥18 可按 body_face_scale\n 在脸区衰减 Body 残差）\n\nDual 扩展（Hi4D 双人线）：\n+ interaction 分支（接触场景）\n+ appearance 分支（PersonTokenBinder\n  把 A/B 源外观绑定到目标区域）",
    fc=OUT, ec=OUT_EDGE, align="left", fs=8.4)
arrow(ax, 33.4, 24, 35.6, 24, color="#8A5A22", lw=2.2)
arrow(ax, 63.4, 24, 65.6, 24, color="#2E7C99", lw=2.2)

# shared bottom note
ax.text(50, 1.6,
        "两版共用：hook 注入不改模型代码 · 只扰动目标 token 段（源参考段/文本段原样返回）· flow matching 速度损失（Body 人体区 3× 加权；Face 支持 global/ROI 两种）",
        ha="center", fontsize=8.6, color="#5A6B7C", style="italic")

fig.savefig(__file__.replace("draw_qwen21_arch.py", "qwen21_adapter_internals.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("done")
