R=/home/shangguanrz/project/pic-edit/add/outputs
echo "###### v1 manifest head"; head -c 900 "$R/3dpw_assets_v1/frame_assets_manifest.json"; echo
echo "###### v1 per-frame json sample"; head -c 1200 "$R/3dpw_assets_v1/test/downtown_bus_00/003977.json"; echo
echo "###### v1 keys"; python3 -c "import json;d=json.load(open('$R/3dpw_assets_v1/test/downtown_bus_00/003977.json'));print(sorted(d.keys()))" 2>/dev/null
echo "###### v2_1 manifest head"; head -c 600 "$R/3dpw_assets_v2_1/frame_assets_manifest.json"; echo
echo "###### v2_1 sequence_assets subdir sample"; ls "$R/3dpw_assets_v2_1/sequence_assets/test/downtown_bus_00" | head -8
echo "###### v2_1 sequence_assets counts"; for d in "$R/3dpw_assets_v2_1/sequence_assets"/*/; do printf '%s ' "$d"; find "$d" -type f | wc -l; done
echo "###### all_v1 subdir list + counts"; for d in "$R/3dpw_assets_all_v1"/*/; do printf '%s ' "$d"; find "$d" -type f | wc -l; done
echo "###### all_v1 small json files"; find "$R/3dpw_assets_all_v1" -maxdepth 1 -name '*.json' -printf '%s %f\n' | sort -n | head -20
