PY=/home/shangguanrz/nihaochen/envs/qwen21/bin/python
ls -l $PY 2>/dev/null || echo NO_QWEN21
$PY -c "import torch,numpy;print('torch',torch.__version__,'np',numpy.__version__)" 2>&1 | tail -2
ls /home/shangguanrz/project/pic-edit/add/envs 2>/dev/null
