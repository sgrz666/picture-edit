# TikTok PE-I2I 2底座 × 4提示词协议 (8组) 联合消融实验运行手册

**实验编号:** `pei2i_body_face_8arm_20261009`  
**基线代码 SHA:** `4fc7457957175ae0c7f02e120119aae86681f072`  
**方案规范:** `docs/superpowers/plans/2026-10-10-pei2i-body-face-eight-arm-code-plan.md`  

---

## 1. 实验目标与八组架构

本实验评估在固定 TikTok Body+Face 联合架构、固定步长（每组 1024 次新优化器更新，单样本批次，无梯度累积）以及统一 J256 初始权重条件下，两套基干底座（纯 Qwen 2.1 vs 冻结官方两图 BFSHead LoRA）与四种提示词协议（P0 默认、PG 仅数值几何、PB 粗粒度外观、PBG 外观+几何全提示）的交叉效应。

### 核心约束与输入隔离
1. **源图 A RGB 为唯一生成输入:** 目标图 B RGB 仅用于离线人工评测与真值对比，绝对禁止送入 Qwen 图像生成器或扩散潜在空间。
2. **PB / PBG 边界:** 目标 B 信息仅通过离线 Prompt Engineer (PE) 视觉语言模型重写为粗粒度文本描述，以及离线提取 Body25 / DWPose 数值坐标。
3. **原生分辨率评测:** 全程保持原图长宽比，在原尺寸画布上进行反补边 (unpad) 审计与指标计算 (`per_sample_native`)。
4. **冻结门禁:** LoRA 参数在训练期间保持 `requires_grad=False`；Hand 与 Depth 分支保持关闭。

### 八组配置速查表

| Arm 标识 | 基干底座 | 提示词协议 | 加载 LoRA | Body 分支 | Face 分支 | Hand 分支 | Depth 分支 |
|---|---|---|---|---|---|---|---|
| `qwen_p0` | Qwen 2.1 | P0 (固定默认) | 无 | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `qwen_pg` | Qwen 2.1 | PG (数值几何) | 无 | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `qwen_pb` | Qwen 2.1 | PB (外观描述) | 无 | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `qwen_pbg` | Qwen 2.1 | PBG (全信息) | 无 | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `head_p0` | BFSHead | P0 (固定默认) | BFSHead (强度1.0) | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `head_pg` | BFSHead | PG (数值几何) | BFSHead (强度1.0) | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `head_pb` | BFSHead | PB (外观描述) | BFSHead (强度1.0) | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |
| `head_pbg` | BFSHead | PBG (全信息) | BFSHead (强度1.0) | 激活 (lr=5e-6) | 激活 (lr=5e-6) | 关闭 | 关闭 |

---

## 2. 基础数据哈希锁定

- **共同 J256 初始化适配器权重 SHA256:** `c2cf6ba9f61b0c0a96ae8a287959223ea4e17631fd2f4834710156942c7aa3ac`
- **训练集名单 SHA256:** `da21ea9d95be09c735d487f58d927d7da5e82500c0556f082e66d03cf1e0586e` (512 对)
- **验证集名单 SHA256:** `35c24ab77e13c6396979a4891bca726f55e5b3ee28e679ecabf7ddb22a075306` (24 对)
- **跨人物诊断集名单:** 6 对
- **独立测试集 test24 名单:** 24 对 (严格门禁锁定，直到 8 组训练全部达 1024 步后方可生成提示词并推理)

---

## 3. CLI 工具使用说明

### 3.1 主控运行脚本: `scripts/run/run_qwen21_pei2i_study.py`

```text
usage: run_qwen21_pei2i_study.py [-h] --run RUN
                                 --phase {lock,rewrite-qa,rewrite,faces,cache,selfcheck,preflight,initial,train,finaltest,report,export}
                                 [--splits SPLITS [SPLITS ...]]
                                 [--arms ARMS [ARMS ...]]
                                 [--max-updates MAX_UPDATES]
                                 [--eval-every EVAL_EVERY] [--resume]
                                 [--dry-run]

PE-I2I study master runner

options:
  -h, --help            show this help message and exit
  --run RUN             Experiment root
  --phase {lock,rewrite-qa,rewrite,faces,cache,selfcheck,preflight,initial,train,finaltest,report,export}
                        Execution phase
  --splits SPLITS [SPLITS ...]
  --arms ARMS [ARMS ...]
  --max-updates MAX_UPDATES
  --eval-every EVAL_EVERY
  --resume
  --dry-run
```

### 3.2 中文离线报告生成器: `scripts/reports/build_qwen21_pei2i_study_report.py`

```text
usage: build_qwen21_pei2i_study_report.py [-h] --run RUN [--output OUTPUT]
                                          [--thumb-side THUMB_SIDE]

Build Chinese HTML report for 8-arm study

options:
  -h, --help            show this help message and exit
  --run RUN             Experiment run root
  --output OUTPUT       Output directory (default: <run>/report)
  --thumb-side THUMB_SIDE
                        Thumbnail max dimension
```

---

## 4. 逐步执行指令 (Linux/GPU 环境)

在 GPU 服务器（如 `sg`）上执行完整研究流程：

### 第一步：锁定输入，准备开发样本
```bash
export STUDY_PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
export STUDY_RUN=/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009

$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase lock
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase rewrite-qa
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase report
```

### 第二步：正式 PE 文本重写、Face 资产与缓存预计算
```bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase rewrite \
  --splits train val diagnostic
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase faces \
  --splits train val diagnostic test
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase cache \
  --splits train val diagnostic
```

### 第三步：自检门禁与八组梯度检查
```bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase selfcheck
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase preflight
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase initial --dry-run
```

### 第四步：保留各组 Step 0 基线输出
```bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase initial
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase report
```

### 第五步：逐 128 步断点保存，完成 1024 优化器更新
```bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase train \
  --max-updates 1024 --eval-every 128 --resume
```

### 第六步：8 组全部完成后解锁 test24 并完成最终评测
```bash
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase rewrite --splits test
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase cache --splits test
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase finaltest
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase report
$STUDY_PY scripts/run/run_qwen21_pei2i_study.py --run "$STUDY_RUN" --phase export
```

### 第七步：下载至本地并离线查看中文报告
在本地 Windows PowerShell 执行：
```powershell
$studyLocal = "D:\codeplus\pictureedit\research\pei2i_body_face_8arm_20261009"
New-Item -ItemType Directory -Path $studyLocal -Force
scp sg:/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/export/results.tar.gz "$studyLocal/results.tar.gz"
tar -xzf "$studyLocal/results.tar.gz" -C "$studyLocal"
Start-Process "$studyLocal/report/index.html"
```

---

## 5. 测试验证记录

本套代码已通过完整的自动化测试覆盖：

1. `tests/test_qwen21_prompt_study_protocol.py` (8 passed)
2. `tests/test_qwen21_prompt_study_pairs.py` (5 passed)
3. `tests/test_qwen21_prompt_study_geometry.py` (4 passed)
4. `tests/test_qwen21_prompt_study_prompts.py` (4 passed)
5. `tests/test_qwen21_prompt_study_faces.py` (2 passed)
6. `tests/test_qwen21_prompt_study_cache.py` (2 passed)
7. `tests/test_qwen21_prompt_study_training.py` (2 passed)
8. `tests/test_qwen21_prompt_study_eval.py` (2 passed)
9. `tests/test_qwen21_prompt_study_runner.py` (4 passed, 3696 total jobs verified)
10. `tests/test_qwen21_prompt_study_selfcheck.py` (2 passed)
11. `tests/test_qwen21_prompt_study_report.py` (2 passed)
12. `tests/test_qwen21_prompt_study_export.py` (2 passed)
13. `tests/test_qwen21_native_resolution.py` (14 passed)
14. `tests/test_qwen21_joint_pilot_resume.py` (29 passed)
15. `tests/test_qwen21_joint_pilot_visuals.py` (33 passed)
16. `tests/test_qwen21_joint_pilot_summary.py` (12 passed)
