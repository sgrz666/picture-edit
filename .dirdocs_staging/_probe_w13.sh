cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== deepgen git diff stat + README diff ==="
git -C deepgen diff --stat | tail -5
git -C deepgen diff -- README.md | head -40
echo "=== deepgen ls-files count by dir ==="
git -C deepgen ls-files | awk -F/ '{print $1}' | sort | uniq -c
echo "=== deepgen/src tree ==="
find deepgen/src -maxdepth 2 -type f -name '*.py' -printf '%s %p\n' | sort -k2 | head -50
echo "=== deepgen configs tree ==="
find deepgen/configs -maxdepth 3 -type f ! -name 'README.md' ! -name '目录说明.md' -printf '%s %p\n' | head -30
echo "=== DeepGen-RL contents ==="
ls -la deepgen/DeepGen-RL/; cat deepgen/DeepGen-RL/README.md
echo "=== scripts/process_data, evaluation ==="
ls deepgen/scripts/process_data deepgen/scripts/evaluation
echo "=== patch applied? grep in pipeline ==="
grep -c 'block_controlnet_hidden_states' models/DeepGen-1.0-diffusers/deepgen_pipeline.py models/DeepGen-1.0-diffusers/deepgen_pipeline.py.bak.pre-control
git apply --check patches/deepgen_pose_control.patch 2>&1 | head -5
echo "=== project refs to deepgen/ (upstream copy) ==="
grep -rIn --include='*.py' --include='*.sh' --include='*.yaml' -e 'deepgen/src' -e 'deepgen\.src' -e 'from deepgen' -e 'sys.path.*deepgen' -e 'deepgen/scripts' src scripts configs workflow tests 2>/dev/null | head -20
echo "=== deepgen/README.md (modified) ==="
cat deepgen/README.md
