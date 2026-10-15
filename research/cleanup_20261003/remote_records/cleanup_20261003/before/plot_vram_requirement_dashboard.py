# -*- coding: utf-8 -*-
"""
生成《DeepGen 1.0 姿态编辑实验显存需求全景看板》高清可视化图表
包含：
1. 训练模式显存实测与消费级/专业级显卡承载线 (16G/24G/48G/84G)
2. 全生命周期三大阶段（缓存生成、训练、推理）与建议容量预算
3. 显存优化瀑布分解（无检查点在线 -> 缓存 -> 检查点，降幅 61.2%）
4. 5B DeepGen vs 20B Qwen-Edit 跨量级显存对比
"""

import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(16, 11), dpi=250, facecolor='#F8FAFC')
fig.suptitle('DeepGen 1.0 (5B) 姿态编辑实验显存需求全景分析看板', fontsize=18, fontweight='bold', color='#1A365D', y=0.98)

# ----------------- 子图 1：四种训练模式与硬件承载线 -----------------
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor('#FFFFFF')
resolutions = ['512×512', '768×768', '1024×1024']
x = np.arange(len(resolutions))
width = 0.18

c_cache_ckpt = [10.51, 13.40, 18.01]
c_cache_nockpt = [15.05, 23.24, 37.34]
c_online_ckpt = [18.75, 21.82, 26.28]
c_online_nockpt = [23.61, 32.11, 46.41]

r1 = ax1.bar(x - 1.5*width, c_cache_ckpt, width, label='【推荐】缓存条件 + 开启梯度检查点', color='#2B6CB0', zorder=3)
r2 = ax1.bar(x - 0.5*width, c_cache_nockpt, width, label='缓存条件 + 关闭检查点', color='#38B2AC', zorder=3)
r3 = ax1.bar(x + 0.5*width, c_online_ckpt, width, label='在线编码 + 开启梯度检查点', color='#DD6B20', zorder=3)
r4 = ax1.bar(x + 1.5*width, c_online_nockpt, width, label='在线编码 + 关闭检查点 (高危)', color='#E53E3E', zorder=3)

# 硬件门槛基准线
ax1.axhline(16, color='#718096', linestyle='--', linewidth=1.2, alpha=0.85, label='16GB 门槛 (RTX 4080 / 4060Ti)')
ax1.axhline(24, color='#3182CE', linestyle='--', linewidth=1.5, alpha=0.9, label='24GB 门槛 (RTX 3090 / 4090)')
ax1.axhline(48, color='#805AD5', linestyle='--', linewidth=1.2, alpha=0.85, label='48GB 门槛 (RTX A6000)')
ax1.axhline(83, color='#E53E3E', linestyle='-', linewidth=1.5, alpha=0.7, label='83GB 本机实测物理上限 (RTX 6000D)')

ax1.set_ylabel('NVML 整卡峰值显存 (GiB)', fontsize=11, fontweight='bold', color='#1A365D')
ax1.set_title('(a) 训练模式显存实测与显卡承载门槛', fontsize=12, fontweight='bold', color='#1A365D', pad=8)
ax1.set_xticks(x)
ax1.set_xticklabels(resolutions, fontsize=10.5, fontweight='bold')
ax1.set_ylim(0, 90)
ax1.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E0')
ax1.legend(loc='upper left', fontsize=7.8, framealpha=0.92, facecolor='#FFFFFF')

for rects in [r1, r2, r3, r4]:
    for rect in rects:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5, color='#2D3748')

# ----------------- 子图 2：全生命周期阶段与建议容量 -----------------
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor('#FFFFFF')

w2 = 0.22
x2 = np.arange(len(resolutions))

s_cache = [14.89, 15.68, 16.74]
s_train = [10.57, 13.40, 18.01]
s_infer = [15.96, 17.42, 19.35]
budget = [18.0, 20.0, 22.0]

b1 = ax2.bar(x2 - w2, s_cache, w2, label='阶段 1：条件缓存抽取生成', color='#4FD1C5', zorder=3)
b2 = ax2.bar(x2, s_train, w2, label='阶段 2：推荐稳态训练峰值', color='#2B6CB0', zorder=3)
b3 = ax2.bar(x2 + w2, s_infer, w2, label='阶段 3：独立适配器推理峰值', color='#805AD5', zorder=3)

# 建议容量连线与散点
ax2.plot(x2, budget, color='#E53E3E', marker='s', markersize=8, linewidth=2.2, label='建议预算容量 ceil(峰值+余量)', zorder=4)

ax2.set_ylabel('显存占用 (GiB)', fontsize=11, fontweight='bold', color='#1A365D')
ax2.set_title('(b) 全实验生命周期显存占用与建议容量预算', fontsize=12, fontweight='bold', color='#1A365D', pad=8)
ax2.set_xticks(x2)
ax2.set_xticklabels(resolutions, fontsize=10.5, fontweight='bold')
ax2.set_ylim(0, 30)
ax2.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E0')
ax2.legend(loc='upper left', fontsize=8.2, framealpha=0.92, facecolor='#FFFFFF')

for i, b in enumerate(budget):
    ax2.annotate(f'建议: {int(b)}G', (x2[i], b), xytext=(0, 6), textcoords='offset points',
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E53E3E')

for rects in [b1, b2, b3]:
    for rect in rects:
        h = rect.get_height()
        ax2.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7.5, color='#4A5568')

# ----------------- 子图 3：显存优化策略降幅瀑布图 (1024 分辨率) -----------------
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor('#FFFFFF')

stages = ['基线:在线无检查点', '启用条件缓存\n(卸载VLM)', '启用梯度检查点\n(重算激活)', '最终推荐稳态']
vals = [46.41, 37.34, 18.01, 18.01]
colors = ['#E53E3E', '#DD6B20', '#38B2AC', '#2B6CB0']

bars3 = ax3.bar(stages, vals, width=0.5, color=colors, zorder=3)
ax3.set_ylabel('NVML 整卡显存 (GiB)', fontsize=11, fontweight='bold', color='#1A365D')
ax3.set_title('(c) 1024×1024 关键显存优化手段压减收益 (降幅 61.2%)', fontsize=12, fontweight='bold', color='#1A365D', pad=8)
ax3.set_ylim(0, 55)
ax3.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E0')

# 标注下降箭头与降幅
ax3.annotate('立减 9.07 GiB (-19.5%)\n释放 15.5GB VLM常驻', xy=(0.5, 41.5), xytext=(0.5, 47),
             ha='center', fontsize=8.5, color='#C53030', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#C53030', lw=1.5))

ax3.annotate('大减 19.33 GiB (-51.8%)\n压缩 8192序列激活', xy=(1.5, 27.5), xytext=(1.5, 33),
             ha='center', fontsize=8.5, color='#2C7A7B', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2C7A7B', lw=1.5))

for bar in bars3:
    y = bar.get_height()
    ax3.annotate(f'{y:.2f} GiB', xy=(bar.get_x() + bar.get_width()/2, y),
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold', color='#1A365D')

# ----------------- 子图 4：DeepGen 1.0 (5B) vs Qwen-Edit (20B) 对比 -----------------
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor('#FFFFFF')

dimensions = ['静态纯权重', '单图原生推理', '缓存微调训练', '在线微调训练']
deepgen_vals = [5.09 * 2 * 0.93, 15.96, 18.01, 26.28] # DeepGen 在 1024 下各指标
qwen20b_vals = [53.74, 63.0, 57.0, 78.0] # Qwen 20B 在 1024 下测算均值

x4 = np.arange(len(dimensions))
w4 = 0.32

b_dg = ax4.bar(x4 - w4/2, deepgen_vals, w4, label='DeepGen 1.0 (5B) [当前主推]', color='#2B6CB0', zorder=3)
b_qw = ax4.bar(x4 + w4/2, qwen20b_vals, w4, label='Qwen-Image-Edit (20B) [质量上界对照]', color='#E53E3E', zorder=3)

ax4.axhline(83, color='#C53030', linestyle='--', linewidth=1.5, alpha=0.8, label='RTX 6000D 可见上限 (83 GiB)')
ax4.axhline(24, color='#3182CE', linestyle=':', linewidth=1.2, alpha=0.8, label='消费级单卡旗舰 (24 GiB)')

ax4.set_ylabel('显存占用 (GiB)', fontsize=11, fontweight='bold', color='#1A365D')
ax4.set_title('(d) 5B 本案 vs 20B 官方大模型显存体量对比 (1024×1024)', fontsize=12, fontweight='bold', color='#1A365D', pad=8)
ax4.set_xticks(x4)
ax4.set_xticklabels(dimensions, fontsize=9.5, fontweight='bold')
ax4.set_ylim(0, 95)
ax4.grid(axis='y', linestyle=':', alpha=0.5, color='#CBD5E0')
ax4.legend(loc='upper left', fontsize=8.2, framealpha=0.92, facecolor='#FFFFFF')

for rects in [b_dg, b_qw]:
    for rect in rects:
        h = rect.get_height()
        ax4.annotate(f'{h:.1f}G',
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, color='#2D3748')

plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.07, wspace=0.22, hspace=0.28)

out_1 = 'd:/codeplus/pictureedit/results/deepgen_vram_requirement_dashboard.png'
out_2 = 'd:/codeplus/pictureedit/demooutput/deepgen_vram_requirement_dashboard.png'

plt.savefig(out_1)
plt.savefig(out_2)
plt.close()

print(f"VRAM Dashboard generated successfully:\n  1. {out_1}\n  2. {out_2}")
