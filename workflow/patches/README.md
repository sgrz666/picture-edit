# `workflow/patches` —— 对未入库 vendor 依赖的冻结副本

`workflow/` 的几何阶段依赖 `add/tools/SMPLest-X`，而 `/add/` 被仓库根 `.gitignore` 忽略、git 跟踪 0 条。也就是说流水线跑的那份估计器代码本身不在版本控制里。这里保存它相对上游的本地改动，让依赖可复现、可追责。

- 上游：`https://github.com/MotrixLab/SMPLest-X.git`
- 基线 commit：`fdebd887a317f9004b435c57812d1a8936295360`（`Update README.md`）
- 抓取时间：2026-10-12（node01）

| 文件 | 内容 |
| --- | --- |
| `SMPLest-X_local.diff` | 该 checkout 相对基线的全部本地改动：`main/inference.py`（+191/-2）、`scripts/inference.sh`、`utils/visualization_utils.py` |
| `SMPLest-X_run_smplestx_params_only.py` | 上游没有、本地新增的入口脚本；`smplx_worker.py` 直接调用它，`create_estimator_sandbox` 会把它复制进每个 run 的沙箱 |

## workflow 真正依赖的那一处

`workflow/config.json` 的 `person_detection_confidence` 是靠环境变量 `SMPLESTX_DET_CONF` 传进估计器的，读取它的是 `main/inference.py` 里新增的这段（见 `SMPLest-X_local.diff`）：

```python
det_conf_override = os.environ.get("SMPLESTX_DET_CONF")
if det_conf_override is not None:
    cfg.inference.detection.conf = float(det_conf_override)
```

上游没有这个 hook。如果 vendor 目录被重新同步成上游版本，检测阈值会静默回到估计器自己的默认值，而 workflow 仍然在 `config.json` 和 `B_camera.json` 里宣称用的是设定值。因此 `bash run.sh --check` 会显式断言它存在，缺失即报错（输出 `ESTIMATOR_CONF_HOOK_OK` 表示通过）。

## 恢复方式

```bash
cd /home/shangguanrz/project/pic-edit/add/tools/SMPLest-X
git checkout fdebd887a317f9004b435c57812d1a8936295360
git apply ../../../workflow/patches/SMPLest-X_local.diff
cp ../../../workflow/patches/SMPLest-X_run_smplestx_params_only.py scripts/
cd ../../../workflow && bash run.sh --check
```

这里的副本是证据快照，不是构建输入：流水线读的始终是 `add/tools/SMPLest-X` 里的那份，改那边不会自动同步到这里，反之亦然。
