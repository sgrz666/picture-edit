# V4.1 Hybrid job与schema示例

这些JSON只用于填写配置，均未执行。`config.example.json`保存未验收工程默认值；`appearance.example.json`示例只请求Face DINO；`detector.example.json`只给补检ONNX路径；`references.example.jsonl`说明手工收集已提取引用的格式。`schema.json`描述Manifest来源字段及主要固定数组尺寸，运行时代码另执行dtype、SHA、人物和Canvas检查。

示例中的`REPLACE`必须换成真实路径／SHA。完整流程见上两级`NATIVE_FULL_V41_HYBRID.md`。本目录不会搜索或下载模型，也不会从文件缺失推断可用性。
