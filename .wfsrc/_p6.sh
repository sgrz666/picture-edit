set -u
cd /home/shangguanrz/project/pic-edit/workflow || exit 1
QW=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
echo '=== 缓存复用（不带 --force 应 CACHE_HIT）==='
$QW evaluate2.py --run wp1_accept_hard02 --no-lpips 2>&1 | tail -2
echo '=== 成对比较：fitgeo vs promptv2 ==='
$QW scorecard.py --baseline testset_hard05_fitgeo --candidate testset_hard05_promptv2 2>&1 | tail -20
echo '=== 仓级测试收集未被新文件破坏 ==='
cd .. && $QW -m pytest tests/ -q --collect-only 2>&1 | tail -3
