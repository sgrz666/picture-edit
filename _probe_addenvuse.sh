R=/home/shangguanrz/project/pic-edit
echo "== textual usage of each env name (excluding add/envs subtree and gen_folder_docs) =="
grep -rn --include='*.sh' --include='*.py' --include='*.md' --include='*.json' --include='*.yaml' --include='*.toml' \
  -e 'deepgen_pose' -e 'smplestx_bw' -e 'envs/multihmr' -e 'sapiens2_hf' -e 'envs/smpl_clean' \
  "$R" 2>/dev/null | grep -v '/add/envs/' | grep -v gen_folder_docs | grep -v '/archive/' | grep -v '/research/qwen21_dual_upgrade' | head -60
