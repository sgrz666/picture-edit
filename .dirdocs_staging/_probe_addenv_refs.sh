R=/home/shangguanrz/project/pic-edit
echo "== references to add/envs in repo scripts =="
grep -rn --include='*.sh' --include='*.py' --include='*.md' -l 'add/envs' "$R" 2>/dev/null | grep -v '/add/envs/' | head -30
echo
echo "== actual lines (first 40) =="
grep -rn --include='*.sh' --include='*.py' 'add/envs' "$R" 2>/dev/null | grep -v '/add/envs/' | head -40
