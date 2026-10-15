R=/home/shangguanrz/project/pic-edit
echo "== find builder by log banner"
grep -rln "FINAL 512 ASSET BUILDER" $R --include='*.py' --include='*.sh' 2>/dev/null | grep -v datasets/ | head
echo "== find sampled v4 tag"
grep -rln "iper-sampled-v4-6src64tgt" $R --include='*.py' --include='*.sh' --include='*.md' 2>/dev/null | grep -v '/datasets/' | head
echo "== root-level scripts dir listing (if any)"
ls $R | head -40
echo "== any build scripts under add/tools or workflow"
find $R/add $R/workflow -maxdepth 2 -name '*iper*' 2>/dev/null | head -20
echo "== git log for iper asset builder"
cd $R && git log --oneline -12 -- '*iper*512*' 2>/dev/null | head -20
echo "== scripts mentioning split_iper_train_val"
grep -rln "split_iper_train_val\|prepare_qwen21_full_data" $R/scripts $R/docs 2>/dev/null | head -10
echo "== docs mentioning iper assets"
grep -rln "iper_assets_512_nvdiffrast" $R/docs $R/workflow $R/reports 2>/dev/null | head -15
