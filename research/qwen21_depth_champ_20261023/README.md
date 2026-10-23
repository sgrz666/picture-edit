# Qwen 2.1 Depth / CHAMP 编码与融合研究

## 目的

在现有Full→Native路线中确定Depth的输入语义、独立编码方式和512维条件融合接口。重点核对CHAMP真实实现，而不是据名称判断本仓库的改编版已经具备相同能力。

## 里面有什么

- [研究报告](REPORT.zh-CN.md)：原版CHAMP、当前wrapper、真实Depth读取差异，以及分步改造建议。模型结构均为提案，本次没有改训练代码。
- `remote_depth_audit_v1.json`：服务器6份iPER、6份CHAMP示例、3份TikTok样本的数值审计；官方Depth权重的SHA256、所有key/shape和源文件hash。
- `vendored_weight_compatibility.json`：官方权重与本地wrapper的14个匹配CNN tensor、2个输出shape冲突、8个缺失mixer tensor。
- `depth_contract_probes_v1.json`：CPU验证mask-aware汇聚、两种零初始化梯度情形、灰度RGB核转换。
- `sources/`：18份固定commit的官方文本源码，保留CHAMP LICENSE；仅供审阅，不运行其中的训练、安装或Blender脚本。
- `upstream_sources.json` / `additional_sources.json`：来源URL、字节数、SHA256。
- `evidence.json` / `remote_source_hashes_after.json`：本地/服务器版本、审计边界和研究期间2个源文件的并行变更记录；结束时重新确认了Depth相关语义。
- `scripts/`：当前研究自己的抓取与审计脚本。不是生产模型实现。

## 关键结论与使用

先读报告第1、4、5节确定实现边界，再按第6节做对照。CHAMP的公开前向只在最后1/8特征执行一次空间attention，最终320通道特征相加后注入U-Net。我们保留Full frontend，另增Depth+valid分支，投影到Native已有512空间。

iPER抽查PNG与raw Z相关系数为−0.999847至−0.998480，现loader的NPY fallback方向相反。官方Depth文件约10.7MB；其active CNN+attention在移除旧输出head后有1,533,136个元素，适合作为warm-start对照，不代表已经迁移成功。

本地复查来源哈希（在仓库根目录执行，不需要GPU）：

```powershell
python -c "import json,hashlib,pathlib; r=pathlib.Path('research/qwen21_depth_champ_20261023'); m=json.loads((r/'upstream_sources.json').read_text()); assert all(hashlib.sha256((r/f['path']).read_bytes()).hexdigest()==f['sha256'] for f in m['files']); print('16 pinned sources verified')"
```

服务器重跑数据审计时，从仓库根目录投递脚本到新的研究目录；`--output`必须选择尚不存在的文件名。下例使用已投递脚本和已有权重缓存，不改资产、checkpoint或环境包：

```bash
ssh sg
/home/shangguanrz/nihaochen/envs/qwen21/bin/python /data/shangguanrz/pic-edit-research/qwen21_depth_champ_20261023/scripts/audit_remote_depth.py \
  --project /home/shangguanrz/project/pic-edit \
  --output /data/shangguanrz/pic-edit-research/qwen21_depth_champ_20261023/remote_depth_audit_recheck_v1.json \
  --weight-cache /data/shangguanrz/pic-edit-research/qwen21_depth_champ_20261023/official_weights
```

CPU机制探针也可本地运行：

```powershell
$depthProbeName = 'depth_contract_probe_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'
python research/qwen21_depth_champ_20261023/scripts/probe_depth_contracts.py --output (Join-Path $env:TEMP $depthProbeName)
```

不传`--official-weight`时跳过第一层权重转换，只核对mask与梯度。服务器的`depth_contract_probes_v1.json`包含实际官方权重转换结果：最大误差2.98e-7。重跑探针与资产审计均不生成图片；assert成功是机制验证，不是图像效果验收。

本地wrapper兼容性复查（使用审计中保存的官方tensor shape，不下载权重）：

```powershell
$depthCompatName = 'depth_compat_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.json'
python research/qwen21_depth_champ_20261023/scripts/audit_vendored_compatibility.py --output (Join-Path $env:TEMP $depthCompatName)
```

## 与其他工作的关系

延续 `../qwen21_full_native_geometry_20261022/` 和现有 `codex/full-native-geometry` 实现。研究读取了服务器当前canvas契约，后续Depth必须遵守同一个target canvas；本次没有覆盖其正在进行的loader/Native改动。原Full继续作为固定对照，未加载Union/Pro网络。
