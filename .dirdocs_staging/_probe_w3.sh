cd /home/shangguanrz/project/pic-edit || exit 1
for f in models/DWPose/README.md models/antelopev2/README.md models/face_eval/README.md models/smpl/README.md models/smpl/smpl/README.md models/DeepGen-1.0-diffusers/*/README.md; do
  echo "##### $f"; cat "$f" 2>/dev/null | head -40
done
echo "=== git log models ==="
git log --oneline -5 -- models/ 2>/dev/null
echo "=== git status models ==="
git status --short models/ | head -20
echo "=== git ls-files models ==="
git ls-files models/ | head -30
