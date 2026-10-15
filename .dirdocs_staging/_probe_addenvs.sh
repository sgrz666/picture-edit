R=/home/shangguanrz/project/pic-edit/add/envs
for d in "$R"/*/; do
  echo "===== $d"
  du -sh --si "$d" 2>/dev/null
  ls "$d" | head -20 | tr '\n' ' '; echo
  echo "-- bin/python:"; ls -la "$d/bin/python"* 2>/dev/null | head -5
  echo "-- conda history head:"; head -2 "$d/conda-meta/history" 2>/dev/null
  if [ -f "$d/pyvenv.cfg" ]; then echo "-- pyvenv.cfg:"; cat "$d/pyvenv.cfg"; fi
  echo "-- site-packages:"; ls -d "$d"/lib/python*/site-packages 2>/dev/null | head -3
  echo "-- torch version:"; head -6 "$d"/lib/python*/site-packages/torch/version.py 2>/dev/null
  echo "-- conda-meta json count:"; ls "$d"/conda-meta/*.json 2>/dev/null | wc -l
  echo "-- site-packages pkg count:"; ls "$d"/lib/python*/site-packages 2>/dev/null | wc -l
  echo "-- big weights:"; find "$d" -maxdepth 4 \( -name '*.pt' -o -name '*.safetensors' -o -name '*.pth' -o -name '*.onnx' -o -name '*.ckpt' \) -printf '%s %p\n' 2>/dev/null | sort -rn | head -6
  echo "-- top-level file count:"; find "$d" -maxdepth 1 -type f | wc -l
done
