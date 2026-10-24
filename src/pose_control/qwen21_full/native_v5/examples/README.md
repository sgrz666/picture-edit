# V5配置与job示例

本目录文件只作填写模板，没有执行。把`REPLACE`替换为真实路径／SHA／pair_id，所有新产物使用`/data/shangguanrz/`新版本目录。

`promote.example.json`引用已校验V4 manifest并为每个pair显式声明Detail／监督不可用；需要可用资产时按上一级README的available格式填写。`config.example.json`固定生产尺寸与默认分流。`evaluators.example.json`声明本地冻结评估器与SHA；`acceptance.example.json`仅供后续真实资产验收。

`quality.example.json`、`export_repair.example.json`、`import_repair.example.json`分别传给同名子命令。修复结果和可信support使用已有无pickle NPZ字典序列化器，键／维度见`schema.json`；结果只能有2D条件，不能写Hand45、BPS或Camera-Z。这些job都未运行。

多参考job沿用Source-only结构：`role=source_only`、独立`modalities:{arcface,face_dino,hand_dino}`、稳定`source_person_ids[2]`、`references[1..3]`，每份含`rgb:{path,sha256}`、自己的Canvas、`person_ids`、`geometry_sha256`和`people`（Face／Hand ROI、独立有效性、可选ArcFace五点）。R>1必须`explicit_multi_reference=true`；模型路径、文件SHA和device显式声明。使用V5的`references`子命令保存该声明，不向Qwen添加参考图。
