"""Draw the adapter project layered-architecture diagram (docs/qwen21_architecture/)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DARK = "#22303F"
DATA, DATA_E = "#E5F3E1", "#4E8B44"        # offline assets
INJ, INJ_E = "#E8EEF7", "#4A6FA5"          # conditioning
REAS, REAS_E = "#E2F0F6", "#2E7C99"        # reasoning
BRIDGE, BRIDGE_E = "#EDE4F7", "#7A4FA3"    # bridge/binding
CORE, CORE_E = "#FFF3DC", "#D99125"        # control core
INJECT, INJECT_E = "#FBE4EC", "#B0567F"    # injection


def box(ax, x, y, w, h, text, *, fc, ec, fs=8.6, lw=1.3, bold=False, align="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    weight = "bold" if bold else "normal"
    if align == "left":
        ax.text(x + 1.1, y + h - 1.2, text, ha="left", va="top", fontsize=fs,
                color=DARK, zorder=4, weight=weight, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=DARK, zorder=4, weight=weight, linespacing=1.4)


def down(ax, x, y1, y2, *, color, lw=2.2):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=17,
                                 color=color, lw=lw, shrinkA=1, shrinkB=1, zorder=2))


def pill(ax, x, y, text, *, color, fs=7.6):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            zorder=5, weight="bold",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=color, lw=1.1))


fig, ax = plt.subplots(figsize=(17.0, 12.4), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

ax.text(50, 98.3, "Adapter 项目架构 —— 五层流水线，两个宿主，一条版本链",
        ha="center", fontsize=16.5, weight="bold", color=DARK)
ax.text(50, 95.6, "同一套 SMPL-X 控制架构：DeepGen-1.0 走原生残差接口，Qwen-Image-2.1 走边界桥 + hook 注入",
        ha="center", fontsize=9.5, color="#5A6B7C", style="italic")

LX, LW = 2, 63  # main column

# ---- Layer 0: offline data & condition assets
box(ax, LX, 86.5, LW, 7.2, "", fc=DATA, ec=DATA_E)
ax.text(LX + 1.2, 92.4, "第 0 层 · 数据与条件资产（离线，一次性预计算）", fontsize=9.8,
        weight="bold", color="#2F5B2A", ha="left", zorder=5)
box(ax, LX + 1, 87.2, 18, 4.9, "iPER / Hi4D / TikTok\n3DPW / CHAMP / SynBody", fc="white", ec=DATA_E, fs=8.2)
box(ax, LX + 20.5, 87.2, 20, 4.9, "SMPL-X 拟合 + nvdiffrast 渲染\n法线/热图/部位图/接触图/26D", fc="white", ec=DATA_E, fs=8.2)
box(ax, LX + 42, 87.2, 20, 4.9, "每样本缓存\nv6_conditions.pt / v65_face / v66_hand", fc="white", ec=DATA_E, fs=8.2)

down(ax, LX + 12, 86.3, 83.6, color=DATA_E)

# ---- Layer 1: condition injection
box(ax, LX, 74.2, LW, 9.2, "", fc=INJ, ec=INJ_E)
ax.text(LX + 1.2, 82.1, "第 1 层 · 条件注入层  condition_injector.py + conditions.py", fontsize=9.8,
        weight="bold", color="#2E5A85", ha="left", zorder=5)
box(ax, LX + 1, 74.9, 19.5, 5.8, "SpatialConditionEncoder\n空间条件 → 256ch 特征图", fc="white", ec=INJ_E, fs=8.2)
box(ax, LX + 22, 74.9, 19, 5.8, "GlobalConditionTokenEncoder\nbetas/旋转/平移/相机 → 4×256", fc="white", ec=INJ_E, fs=8.2)
box(ax, LX + 42.5, 74.9, 19.5, 5.8, "Contact / Task 编码器\n接触栅格+结构化关系+任务 token", fc="white", ec=INJ_E, fs=8.2)
down(ax, LX + 31, 74.0, 71.3, color=INJ_E)
pill(ax, LX + 40, 72.6, "ConditionBundle（强校验 dataclass，字段缺失/形状错误直接抛错）", color="#4A6FA5")

# ---- Layer 2: reasoning
box(ax, LX, 56.0, LW, 14.6, "", fc=REAS, ec=REAS_E)
ax.text(LX + 1.2, 69.3, "第 2 层 · 推理层  reasoning/（单人必然、双人按需，批内混合安全）", fontsize=9.8,
        weight="bold", color="#1F5F73", ha="left", zorder=5)
box(ax, LX + 1, 56.8, 19.5, 10.6,
    "PersonGeometryReasoner\n每人一路：空间特征\n+ 全局 token + 任务 token\n+ role embedding 区分 A/B",
    fc="white", ec=REAS_E, fs=8.2, align="left")
box(ax, LX + 22, 56.8, 19, 10.6,
    "双人路径（index_select 分组）：\nBidirectionalCrossPerson\nReasoner 相对几何注意力\nContactConditionReasoner\n接触推理 low/high",
    fc="white", ec=REAS_E, fs=8.2, align="left")
box(ax, LX + 42.5, 56.8, 19.5, 10.6,
    "DualFeatureFusion\n→ InternalControlState：\ngeometry low/high\ninteraction low/high\nperson tokens A/B + 掩码",
    fc="white", ec=REAS_E, fs=8.2, align="left")
down(ax, LX + 12, 55.8, 53.1, color=REAS_E)
pill(ax, LX + 40, 54.3, "InternalControlState（几何 / 交互两套独立表征）", color="#2E7C99")

# ---- Layer 3: bridge + identity binding
box(ax, LX, 43.0, LW, 9.6, "", fc=BRIDGE, ec=BRIDGE_E)
ax.text(LX + 1.2, 51.3, "第 3 层 · 控制桥 + 身份绑定  condition_bridge.py + token_encoders.py", fontsize=9.8,
        weight="bold", color="#5A3A85", ha="left", zorder=5)
box(ax, LX + 1, 43.7, 19.5, 6.2,
    "ReasonerControlBridge\n高低分辨率融合 →\ngeometry / interaction scene (128ch)",
    fc="white", ec=BRIDGE_E, fs=8.2, align="left")
box(ax, LX + 22, 43.7, 19, 6.2,
    "AppearanceTokenEncoder\nIP-Adapter 式：masked 源 latent\n→ 8 个解耦身份 token / 人",
    fc="white", ec=BRIDGE_E, fs=8.2, align="left")
box(ax, LX + 42.5, 43.7, 19.5, 6.2,
    "PersonTokenBinder\n几何 token 与外观 token\n绑定到同一 source/role 槽位",
    fc="white", ec=BRIDGE_E, fs=8.2, align="left")
down(ax, LX + 31, 42.8, 40.1, color=BRIDGE_E)
pill(ax, LX + 40, 38.8, "geometry/interaction 场景图 + 已绑定的 adapter tokens", color="#7A4FA3")

# ---- Layer 4: dynamic control core
box(ax, LX, 22.4, LW, 15.8, "", fc=CORE, ec=CORE_E)
ax.text(LX + 1.2, 36.9, "第 4 层 · 动态控制核  control_core.py + face/ + hand/（每个去噪步重新运行）", fontsize=9.8,
        weight="bold", color="#8A5A22", ha="left", zorder=5)
box(ax, LX + 1, 23.2, 29, 12.4,
    "SharedRecurrentControlCore\n3 条分支 × 6 个 stage，复用同一个\nJoint Block（SD3ControlNetModel 组件）：\nstage_emb → 共享 block → LoRA 适配\n→ CrossNorm → 零初始化头 → 1 组残差\n产出 6 组 × {geometry, interaction, detail}",
    fc="white", ec=CORE_E, fs=8.2, align="left")
box(ax, LX + 31.5, 28.6, 14.5, 7.0,
    "FaceControlAdapter (v65)\nArcFace+DINO+landmarks\n脸 ROI 散射",
    fc="white", ec="#B0567F", fs=8.0, align="left")
box(ax, LX + 47.5, 28.6, 14.5, 7.0,
    "HandControlAdapter (v66)\n4 手 ROI + BPS\nVTON 独立 KV",
    fc="white", ec="#7A4FA3", fs=8.0, align="left")
box(ax, LX + 31.5, 23.2, 30.5, 4.6,
    "ControlStrengthController：按去噪进度调度各分支强度窗口",
    fc="white", ec=CORE_E, fs=8.2)
down(ax, LX + 31, 22.2, 19.5, color=CORE_E)
pill(ax, LX + 40, 18.1, "6 组 × 5 分支目标 token 残差（未缩放）", color="#D99125")

# ---- Layer 5: injection interface / hosts
box(ax, LX, 2.0, LW, 15.6, "", fc=INJECT, ec=INJECT_E)
ax.text(LX + 1.2, 16.3, "第 5 层 · 注入接口 → 两个宿主  interface/ + controlled_pipeline / qwen_adapter", fontsize=9.8,
        weight="bold", color="#7A2F52", ha="left", zorder=5)
box(ax, LX + 1, 2.8, 29, 12.0,
    "DeepGenControlInterface\nbuild_raw_residuals / 对齐 / 缩放\n\n宿主 A：DeepGen-1.0（24 blocks）\nblock_controlnet_hidden_states\n原生 6 组残差接口",
    fc="white", ec=INJECT_E, fs=8.2, align="left")
box(ax, LX + 31.5, 2.8, 30.5, 12.0,
    "宿主 B：Qwen-Image-2.1（32 blocks）\n入口 latent_bridge + pixel_shuffle×2\ntext/pooled bridge；出口 5×output_bridge\n32 block 全挂 hook，组残差只加目标段\n简化版：6 层轻量残差头（qwen21/）",
    fc="white", ec=INJECT_E, fs=8.2, align="left")

# ---- right column: version chain
box(ax, 68, 50, 30, 46.5, "", fc="white", ec="#8A5A22", lw=1.5)
ax.text(83, 94.6, "版本链（checkpoint.py 事务化加载）", ha="center", fontsize=10.2, weight="bold", color="#8A5A22")
chain = (
    (7.4, "v6.3 统一架构\ngeometry/interaction 双分支 + 动态控制核\niPER 单人训练（冻结交互分支）"),
    (5.6, "v6.4 face/hand detail 分支\nlegacy detail 条件 + 六输出头"),
    (6.8, "v6.5 独立 Face 适配器\nArcFace/DINO/landmark 契约\nschema v1 · face ROI 散射"),
    (6.8, "v6.6 独立 Hand 适配器\n独立 KV · 深度感知重叠混合"),
    (8.8, "迁移：qwen21_full 整体搬运 V6\n边界桥适配 · build/load_v63..v66\nfreeze_for_*_training 冻结协议"),
)
top = 93.2
bottoms = []
for h, item in chain:
    y = top - h
    box(ax, 69.5, y, 27, h, item, fc="#FFFDF6", ec="#C89A6B", fs=7.9, align="left")
    bottoms.append(y)
    top = y - 1.4
for i in range(len(bottoms) - 1):
    down(ax, 83, bottoms[i], bottoms[i + 1] + chain[i + 1][0], color="#C89A6B", lw=1.8)

box(ax, 68, 2.0, 30, 45.5, "", fc="white", ec="#4A6FA5", lw=1.5)
ax.text(83, 45.9, "横切保障", ha="center", fontsize=10.2, weight="bold", color="#2E5A85")
box(ax, 69.5, 36.8, 27, 7.6,
    "事务化 checkpoint：preflight 全量校验后才改\n状态；拒收旧 optimizer 状态\nfrozen_digest 哈希验证冻结参数",
    fc="#F7FAFD", ec="#4A6FA5", fs=7.7, align="left")
box(ax, 69.5, 28.2, 27, 7.6,
    "契约测试（283 用例，CPU 可跑）：条件契约\n/ 注入契约 / 训练梯度流 / 评测一致性",
    fc="#F7FAFD", ec="#4A6FA5", fs=7.7, align="left")
box(ax, 69.5, 19.6, 27, 7.6,
    "训练脚本家族：train_iper_v6_native / v65_face\n/ v66_hand / qwen21_body·face·dual·full\n/ tiktok_v65 / hi4d_dual",
    fc="#F7FAFD", ec="#4A6FA5", fs=7.7, align="left")
box(ax, 69.5, 11.0, 27, 7.6,
    "评估：DWPose PCK / SSIM·PSNR / ArcFace\n身份 / LPIPS / landmark；四臂对照 +\nON/OFF/Perturbed 三臂协议",
    fc="#F7FAFD", ec="#4A6FA5", fs=7.7, align="left")
box(ax, 69.5, 2.4, 27, 7.6,
    "目录：src/pose_control/{v6, qwen21,\nqwen21_full}；共享 v6 条件推理，\nqwen 线只做边界适配",
    fc="#F7FAFD", ec="#4A6FA5", fs=7.7, align="left")

fig.savefig(__file__.replace("draw_adapter_arch.py", "adapter_architecture.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("done")
