# data/ —— 调试数据目录（不纳入 Git）

本目录存放**合成调试样本**，约 129 MB，仅用于打通训练/推理链路，不纳入仓库，也不是真实训练数据。

## 内容

```
data/debug_samples/
├── images/            # 16 对 512×512 合成图像：sample_000~015 的 _src.png / _tgt.png（共 32 张）
├── controls/          # 对应的模拟结构控制图（由 src/controls/generator.py 生成）
└── pairs.json         # 样本清单（动作对、提示词、是否双人）
```

动作对包括站立→抬右手、交叉双臂→张开、并排→握手、分开→拥抱、站立→坐下、招手→比耶等，由 `create_debug_samples.py` 中的动作列表循环生成。

## 如何重新生成

```bash
python create_debug_samples.py
```

> 注意：脚本内 `base_dir` 写死为服务器路径 `/home/shangguanrz/project/pic-edit/data/debug_samples`，在其他机器运行前需改为本机路径。生成依赖 `src.controls.generator`（Camera、ControlSignalGenerator）以及 opencv-python、Pillow、numpy。

## 说明

这些图像是用 Pillow 程序化绘制的**合成人物示意图形**，不是真实照片；控制图也是人工几何构造，与照片不存在真实对齐关系。因此它们只能用于验证接口、梯度链路和显存开销，**不能作为姿态迁移效果的证据**。真实效果实验需要另行准备身份明确、源图/目标图/目标控制一致的配对数据。
