# HOME 历史工具归档

62份审计确认未变、无活动命令和当前源码引用的旧工具，按 scratch / historical_shell 分组；移动前后SHA-256一致，原HOME路径为兼容链接。没有运行这些工具。4个正式runner继续在HOME，实现分类入口在 scripts/run/home。

具体源/目标/哈希见 operations.json。历史脚本的参数与数据时点不代表当前训练规范。需要回退位置时，核对source确为指向记录target的链接且target哈希与记录一致，再用原文件替换该链接；不要覆盖已出现的新文件。
