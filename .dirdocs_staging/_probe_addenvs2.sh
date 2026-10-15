R=/home/shangguanrz/project/pic-edit/add/envs
for d in "$R"/*/; do
  echo "===== $d"
  du -sh --si "$d" 2>/dev/null
  echo "TOP:"; ls "$d" | head -20 | tr '\n' ' '; echo
  echo "BINPYTHON:"; ls -la "$d/bin/python"* 2>/dev/null | head -5
  echo "CONDAHIST:"; head -2 "$d/conda-meta/history" 2>/dev/null
  if [ -f "$d/pyvenv.cfg" ]; then echo "PYVENVCFG:"; cat "$d/pyvenv.cfg"; fi
  echo "SITEPKGS:"; ls -d "$d"/lib/python*/site-packages 2>/dev/null | head -3
  echo "TORCHVER:"; head -6 "$d"/lib/python*/site-packages/torch/version.py 2>/dev/null
  echo "CONDAMETA_N:"; ls "$d"/conda-meta/*.json 2>/dev/null | wc -l
  echo "WEIGHTS:"; find "$d" -maxdepth 3 -type f \( -name '*.pt' -o -name '*.safetensors' -o -name '*.pth' -o -name '*.onnx' \) -printf '%s %p\n' 2>/dev/null | sort -rn | head -6
  echo "NFILES:"; find "$d" -type f 2>/dev/null | wc -l
done
