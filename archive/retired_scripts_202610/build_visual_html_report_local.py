import json
from pathlib import Path

out_dir = Path("D:/codeplus/pictureedit/单样本测试")
with open(out_dir / "overfit_summary.json", "r", encoding="utf-8") as f:
    data = json.load(f)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset='utf-8'>
<title>iPER V6.4 Detail Branch Single-Sample Overfitting Report (024_6)</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 30px; background: #0f172a; color: #f8fafc; }}
h1, h2, h3 {{ color: #38bdf8; }}
.card {{ background: #1e293b; padding: 20px; border-radius: 8px; margin-bottom: 24px; border: 1px solid #334155; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #334155; }}
th {{ background: #0f172a; color: #94a3b8; }}
.badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
.pass {{ background: #065f46; color: #34d399; }}
.fail {{ background: #7f1d1d; color: #f87171; }}
.img-row {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }}
.img-card {{ background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #334155; text-align: center; }}
.img-card img {{ max-width: 600px; height: auto; border-radius: 4px; display: block; margin: 8px 0; }}
</style>
</head>
<body>
<h1>iPER V6.4 Detail Branch 单样本过拟合实验报告 (024_6)</h1>
<div class='card'>
  <h2>实验概况</h2>
  <p><b>Appearance:</b> 024_6 | <b>Source:</b> source_motion_f000375 | <b>Target:</b> target_f001492 | <b>分辨率:</b> 512×512</p>
  <p><b>训练步数:</b> 1500 步 | <b>总耗时:</b> {data['total_training_time_s']:.1f}s ({data['total_training_time_s']/60:.2f} mins) | <b>平均单步耗时:</b> {data['mean_step_time_ms']:.1f}ms | <b>峰值显存:</b> {data['peak_vram_gib']:.2f} GiB</p>
  <p><b>冻结参数 SHA-256 校验:</b> <code>{data['pre_sha256']}</code> (前后完全一致: {data['pre_sha256'] == data['post_sha256']})</p>
</div>

<div class='card'>
  <h2>验收准则矩阵 (Acceptance Criteria Matrix)</h2>
  <table>
    <tr><th>准则描述</th><th>状态</th><th>实测指标 / 详情</th></tr>
"""

for c in data['criteria']:
    cls = 'pass' if c['passed'] else 'fail'
    status = 'PASS' if c['passed'] else 'FAIL'
    html += f"    <tr><td>{c['desc']}</td><td><span class='badge {cls}'>{status}</span></td><td>{c['detail']}</td></tr>\n"

html += """  </table>
</div>

<div class='card'>
  <h2>生成效果演进 (4-Seed 核心指标变化)</h2>
  <table>
    <tr><th>Step</th><th>ΔFace SSIM (vs OFF)</th><th>ΔFace PSNR (dB)</th><th>ΔHands SSIM</th><th>Hands PCK@0.1</th><th>Outside PSNR Drop (dB)</th></tr>
"""

for ev in data['eval_history']:
    html += f"    <tr><td><b>Step {ev['step']}</b></td><td>{ev['median_face_ssim_diff']:+.4f}</td><td>{ev['median_face_psnr_diff']:+.2f}</td><td>{ev['median_hands_ssim_diff']:+.4f}</td><td>{ev['median_hands_pck_on']:.3f}</td><td>{ev['median_outside_psnr_drop']:.2f}</td></tr>\n"

html += """  </table>
</div>

<div class='card'>
  <h2>视觉对比条 (Seed 42 演进)</h2>
  <p>包含 [Source | Target GT | Detail OFF (V6.3) | Detail Correct (V6.4) | Detail Perturbed] 及脸、左手、右手特写裁剪</p>
  <div class='img-row'>
"""

for step in [0, 250, 500, 1000, 1500]:
    html += f"""    <div class='img-card'>
      <h4>Step {step:04d} (Seed 42)</h4>
      <a href='previews/step_{step:04d}_seed_42.png' target='_blank'><img src='previews/step_{step:04d}_seed_42.png' style='max-width:560px;'></a>
    </div>\n"""

html += """  </div>
</div>

<div class='card'>
  <h2>条件敏感性泛化诊断目标 (Step 1500)</h2>
  <div class='img-row'>
"""

for tgt in ['target_f001016', 'target_f000568', 'target_f001333']:
    html += f"""    <div class='img-card'>
      <h4>Diagnostic Target: {tgt}</h4>
      <a href='previews/step_1500_diag_{tgt}.png' target='_blank'><img src='previews/step_1500_diag_{tgt}.png' style='max-width:280px;'></a>
    </div>\n"""

html += """  </div>
</div>

</body>
</html>"""

with open(out_dir / 'index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print("✓ Successfully generated index.html in 单样本测试")
