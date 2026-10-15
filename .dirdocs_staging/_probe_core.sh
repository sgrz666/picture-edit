cd /home/shangguanrz/project/pic-edit
for d in src src/adapter src/controls src/data src/evaluation src/evaluation/dwpose_onnx src/integration src/pose_control src/pose_control/qwen21 src/pose_control/qwen21_full src/pose_control/qwen21_full/detail src/pose_control/qwen21_full/face src/pose_control/qwen21_full/hand src/pose_control/qwen21_full/interface src/pose_control/qwen21_full/prompt_study src/pose_control/qwen21_full/reasoning src/pose_control/v6 src/pose_control/v6/detail src/pose_control/v6/face src/pose_control/v6/interface src/pose_control/v6/reasoning tests configs patches vram_lab; do
  files=$(find "$d" -maxdepth 1 -type f | wc -l)
  dirs=$(find "$d" -maxdepth 1 -type d ! -name . ! -name __pycache__ | wc -l)
  subtree=$(find "$d" -type f ! -path "*__pycache__*" ! -name "目录说明.md" | wc -l)
  sz=$(du -sh --si "$d" 2>/dev/null | cut -f1)
  doc=""; if [ -f "$d/目录说明.md" ]; then doc="gen-doc"; fi
  echo "$d | files=$files subdirs=$dirs subtree_files=$subtree size=$sz $doc"
done
