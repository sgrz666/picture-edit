cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== models/README.md (existing!) ==="
cat models/README.md
echo "=== DWPose ==="
ls -la models/DWPose/
echo "=== DeepGen top ==="
ls -la models/DeepGen-1.0-diffusers/
echo "=== DeepGen du ==="
du -sh --si models/DeepGen-1.0-diffusers/* 2>/dev/null
echo "=== antelopev2 ==="
ls -la models/antelopev2/
echo "=== antelopev2 nested ==="
ls -la models/antelopev2/antelopev2/ 2>/dev/null
echo "=== face_eval ==="
ls -la models/face_eval/
echo "=== smpl ==="
ls -laR models/smpl/ | head -60
