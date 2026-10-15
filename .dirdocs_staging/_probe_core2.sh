cd /home/shangguanrz/project/pic-edit
echo ==Q21FULL; ls src/pose_control/qwen21_full/
echo ==PROMPT_STUDY_GLOB; find src -maxdepth 2 -name "prompt*" 
echo ==CONFIGS; ls configs/
echo ==TESTS; ls tests/ | head -70
echo ==SRC_README_FIRSTLINE; head -3 src/adapter/README.md src/pose_control/qwen21/README.md vram_lab/目录说明.md 2>/dev/null | head -20
