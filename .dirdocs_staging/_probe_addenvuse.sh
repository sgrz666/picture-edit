R=/home/shangguanrz/project/pic-edit
echo "== textual usage of env names across repo (excluding add/envs subtree) =="
grep -rn -E 'deepgen_pose|smplestx_bw|envs/multihmr|sapiens2_hf|envs/smpl_clean' "$R" --include='*.sh' --include='*.py' --include='*.md' --include='*.json' --include='*.yaml' 2>/dev/null | grep -v '/add/envs/' | grep -v gen_folder_docs | grep -v '/archive/' | grep -v '/research/' | head -60
