cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== full deepgen src listing (tracked files) ==="
git -C deepgen ls-files src scripts configs
echo "=== project refs to deepgen module names ==="
grep -rIn --include='*.py' -e 'custom_runner' -e 'CustomRunner' -e 'collate_functions' -e 'custom_adamw' -e 'deepgen' src scripts configs workflow tests 2>/dev/null | grep -vi 'deepgen_pipeline\|DeepGen-1.0-diffusers' | head -30
