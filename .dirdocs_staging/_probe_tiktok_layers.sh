set -u
R=/data/shangguanrz/pic-edit-datasets/TikTokDataset
echo "=== V2 manifests & qc ==="
ls -la $R/TikTok_v2/manifests
echo "--- qc tree ---"
find $R/TikTok_v2/qc -maxdepth 2 -printf '%y %s %p\n' | head -40
echo "=== V3 manifests & qc & cache listing ==="
ls -la $R/TikTok_v3/manifests
find $R/TikTok_v3/qc -maxdepth 2 -printf '%y %s %p\n' | head -60
ls $R/TikTok_v3/qwen21_face_feature_cache | head -10
ls $R/TikTok_v3/qwen21_face_feature_cache | wc -l
echo "=== v3 stats.json ==="
cat $R/TikTok_v3/stats.json
echo "=== dwpose_v1 sample listing ==="
ls $R/TikTok_v3/dwpose_v1 | head -8
ls $R/TikTok_v3/dwpose_v1 | wc -l
echo "--- one sequence dir depth2 ---"
d=$(ls -d $R/TikTok_v3/dwpose_v1/*/ | head -1); echo "$d"; find "$d" -maxdepth 2 -printf '%y %s %p\n' | head -25
echo "=== 3d_assets_native: sequences count & one full depth2 sample ==="
find $R/TikTok_3d_assets_native -maxdepth 1 -type d | wc -l
find $R/TikTok_3d_assets_native/00001 -maxdepth 2 -printf '%y %s %p\n' | head -60
echo "--- a 3-subdir seq ---"
find $R/TikTok_3d_assets_native/00003 -maxdepth 2 -printf '%y %s %p\n' | head -25
echo "=== du apparent vs real for 3d_assets_native sample & v3 ==="
du -sh --si $R/TikTok_3d_assets_native/00001
du -sh --apparent-size --si $R/TikTok_3d_assets_native/00001
du -sh --si $R/TikTok_v3/dwpose_v1 | cat
du -sh --apparent-size --si $R/TikTok_v3/dwpose_v1 | cat
du -sh --si $R/TikTok_v3/qwen21_face_feature_cache | cat
du -sh --apparent-size --si $R/TikTok_v3/qwen21_face_feature_cache | cat
echo "=== showcase_single sample depth2 ==="
find $R/showcase_single/sample_01_00001_0014_to_0074 -maxdepth 2 -printf '%y %s %p\n' | head -40
du -sh --si $R/showcase_single/sample_01_00001_0014_to_0074
echo "=== model/model tree ==="
find $R/model -maxdepth 3 -printf '%y %s %p\n' | head -40
echo "=== .complete tree ==="
find $R/.complete -maxdepth 4 -printf '%y %s %p\n' | head -20
echo "=== splits contents ==="
cat $R/splits/split_summary.json
echo; head -5 $R/splits/train_sequences.txt; echo ...; cat $R/splits/test_sequences.txt $R/splits/val_sequences.txt
echo "=== log tail ==="
tail -n 30 $R/logs/build_tiktok_3d_assets_native_full.out
