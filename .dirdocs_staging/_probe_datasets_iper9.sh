R=/home/shangguanrz/project/pic-edit
echo "== scripts/data"; ls $R/scripts/data
echo "== scripts/audit"; ls $R/scripts/audit
echo "== scripts/run"; ls $R/scripts/run | head -40
echo "== grep iper builders"
grep -rln "iper_assets_512_nvdiffrast\|iper_sampled_6src64tgt" $R/scripts --include='*.py' --include='*.sh' | head -20
echo "== asset_registry"
ls -la $R/configs/asset_registry.json 2>/dev/null && head -c 2500 $R/configs/asset_registry.json
