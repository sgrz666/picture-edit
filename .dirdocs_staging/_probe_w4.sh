cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== antelopev2/dinov2-giant ls ==="
ls -la models/antelopev2/dinov2-giant/
du -sh --si models/antelopev2/dinov2-giant/*
echo "=== dinov2 config head ==="
head -c 1200 models/antelopev2/dinov2-giant/config.json 2>/dev/null
echo "=== antelopev2 .cache ==="
find models/antelopev2/.cache -maxdepth 3 -printf '%y %s %p\n' | head -20
echo "=== DeepGen model_index.json ==="
cat models/DeepGen-1.0-diffusers/model_index.json
echo "=== DeepGen prompt_template.json ==="
cat models/DeepGen-1.0-diffusers/prompt_template.json
echo "=== transformer config ==="
cat models/DeepGen-1.0-diffusers/transformer/config.json
echo "=== connector config ==="
cat models/DeepGen-1.0-diffusers/connector/config.json
echo "=== scheduler config ==="
cat models/DeepGen-1.0-diffusers/scheduler/scheduler_config.json
echo "=== vae config ==="
cat models/DeepGen-1.0-diffusers/vae/config.json
echo "=== vlm files ==="
ls -la models/DeepGen-1.0-diffusers/vlm/
echo "=== DeepGen .cache ==="
find models/DeepGen-1.0-diffusers/.cache -maxdepth 3 -printf '%y %s %p\n' | head
