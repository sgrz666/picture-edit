# Hi4D 测试集控制图排查

测试清单含 606 条配对，清单 SHA-256 为 `5d45bd564e4ce7db7b8dbf95d4f288755abd33234312bad75da5bdb9836e9453`。控制缓存已覆盖 606/606 条，缺失 0 条。

首次预计算在索引 573（第 574 条）失败，后续 33 条都属于 `pair02/backhug02/cam64`。复核源 SMPL 网格与 `target_geometry_npz` 中渲染的 A/B 语义轮廓发现：该相机序列的部位网格输出在垂直方向倒置。索引 573 两人的原始覆盖率为 44.7% 和 12.5%；仅将部位渲染图上下翻转后，覆盖率分别为 96.6% 和 97.3%。无须平移、缩放或调换 A/B 槽位。

转换器现在只有在翻转覆盖率至少 80%，且比原方向高 25 个百分点时才采用翻转，否则保留原结果并继续原有覆盖率检查。这个判断只读同一目标几何包的 SMPL 渲染和语义区域，不读取目标 RGB 或评测真值掩码。索引 573 的 A/B 部位图在语义区域内覆盖率均为 100%，控制张量全部有限，评测样本以 `load_target=False` 读取时没有 `tgt_image`。

训练在修复前已启动；训练启动时转换器代码 SHA-256 是 `622bbd9d1954b74373d712709c31d826622eb859ce7f2ae1ad6d7b949f89845b`，快照保存在 `code_snapshot/hi4d_qwen21_dataset_at_train_start.py`。测试控制缓存使用修复后 SHA-256 `fcc47094e1a8f3d5326e90596487d71560697ab18b906dd16ccb1cadac56d126`。训练/验证已缓存控制图未因本次修复重算；该变化只影响尚未生成控制图的测试相机序列。

复核命令（在项目根目录）：

```bash
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -m pytest -q tests/test_hi4d_geometry_registration.py
/home/shangguanrz/nihaochen/envs/qwen21/bin/python -c 'from src.data.hi4d_qwen21_dataset import Hi4DQwen21Dataset; d=Hi4DQwen21Dataset("datasets/Hi4D_pilot_v1/final_assets_v4_2/manifests/main_pairs_test.jsonl", "/home/shangguanrz/nihaochen/qwen21_hi4d_controls", load_target=False); print(len(d), sum(not d.control_path(r).exists() for r in d.rows))'
```
