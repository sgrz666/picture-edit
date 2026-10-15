# experiments/qwen21_iper_body/full_20k/ —— Qwen21 身体适配器的 20,000 步正式训练（全项目权威权重产出地）

> 路径：`…/qwen21_iper_body/full_20k`　|　本级：叶子目录，4 文件，1.6 GB　|　数据源：`run_manifest.json` / `train_metrics.jsonl`（17,000 行），读取 2026-10-12

## 一、在干什么
Qwen-Image-2.1 单流 7B 身体适配器在 iPER `dev_train_pairs` 上的完整 20,000 步微调，产出本项目下游反复引用的两个检查点。谱系上由 `pilot_3k_acc4_resumed/checkpoint_last.pt` 续训而来，初始化用 `iper_v6_native_20k_v1/checkpoint_last.pt` 提供的 DeepGen V6 几何条件先验。

## 二、文件与实证
| 文件 | 大小 | 实证内容 |
| :--- | :--- | :--- |
| `checkpoint_best.pt` | 771.9 MB | 训练全过程中验证最优的适配器权重（**下游双人线/16-test 复用此路径**） |
| `checkpoint_last.pt` | 771.9 MB | step 20000 收口权重（**四臂评测脚本加载此路径**） |
| `run_manifest.json` | 2.0 KB | `max_steps=20000, accumulation=2, micro_batch_size=2, save_every=500, val_count=10, seed=42`；`cache=/home/shangguanrz/nihaochen/qwen21_iper_cache`；`model_revision=790c926…`；`git_head=b1a00c2…`；torch 2.8.0/cu12.8/RTX 6000D |
| `train_metrics.jsonl` | 4.5 MB | 17,000 行逐步记录；末行 step 20000：loss 0.2886、val_loss 0.3465、elapsed 48,617s（≈13.5h）、peak 40.68 GiB |

## 三、注意事项
- **当前权威身体权重**：`full_20k/checkpoint_best.pt`（复用初始化）与 `checkpoint_last.pt`（评测基线）。
- `cache` 指向他人目录 `nihaochen/qwen21_iper_cache`，是训练/评测共享的不可删除潜变量缓存。
- 两个 `.pt` 各 771.9 MB，为大张量，默认不入 Git；根分区 96% 占用，清理前先按冷归档规范处理，切勿 `rm -rf`。
