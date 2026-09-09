import os
import sys
import time
from PIL import Image, ImageDraw
import torch

model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
if model_path not in sys.path:
  sys.path.insert(0, model_path)

from deepgen_pipeline import DeepGenPipeline


def create_dummy_source_image() -> Image.Image:
  """Creates a sample source image for editing testing."""
  img = Image.new("RGB", (512, 512), color=(240, 240, 245))
  draw = ImageDraw.Draw(img)
  draw.ellipse([230, 80, 282, 132], fill=(220, 180, 150))  # Head
  draw.rectangle([220, 132, 292, 300], fill=(50, 100, 200))  # Torso (blue shirt)
  draw.rectangle([210, 300, 250, 460], fill=(40, 40, 60))  # Left leg
  draw.rectangle([262, 300, 302, 460], fill=(40, 40, 60))  # Right leg
  draw.line([(220, 150), (170, 250)], fill=(50, 100, 200), width=16)  # Left arm
  draw.line(
      [(292, 150), (342, 250)], fill=(50, 100, 200), width=16
  )  # Right arm
  return img


def run_native_deepgen_test():
  print("=== STARTING NATIVE DEEPGEN IMAGE EDIT TEST ===", flush=True)
  output_dir = "/home/shangguanrz/project/pic-edit/demooutput"
  os.makedirs(output_dir, exist_ok=True)

  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  t0 = time.time()

  print(f"Loading pipeline from {model_path} with BF16...", flush=True)
  pipe = DeepGenPipeline.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
  )
  pipe.to("cuda")

  load_time = time.time() - t0
  load_alloc = torch.cuda.max_memory_allocated() / (1024**3)
  load_reserved = torch.cuda.max_memory_reserved() / (1024**3)
  print(f"✓ Pipeline loaded in {load_time:.2f}s", flush=True)
  print(
      f"✓ VRAM after load: Allocated={load_alloc:.2f} GiB,"
      f" Reserved={load_reserved:.2f} GiB\n",
      flush=True,
  )

  # Create source image
  src_img = create_dummy_source_image()
  src_path = os.path.join(output_dir, "native_input_src.png")
  src_img.save(src_path)
  print(f"✓ Input test image saved to {src_path}", flush=True)

  prompt = "Change the person pose to raising both hands above the head."
  print(
      f"Running inference with prompt: '{prompt}' (512x512, 20 steps)...",
      flush=True,
  )
  torch.cuda.reset_peak_memory_stats()
  t_infer_start = time.time()

  with torch.inference_mode():
    output = pipe(
        prompt=prompt,
        image=src_img,
        height=512,
        width=512,
        num_inference_steps=20,
        guidance_scale=4.0,
        seed=42,
    )

  infer_time = time.time() - t_infer_start
  infer_alloc = torch.cuda.max_memory_allocated() / (1024**3)
  infer_reserved = torch.cuda.max_memory_reserved() / (1024**3)

  out_img = output.images[0]
  out_path = os.path.join(output_dir, "native_test_result.png")
  out_img.save(out_path)

  print(f"✓ Inference completed in {infer_time:.2f}s", flush=True)
  print(f"✓ Result saved to {out_path}", flush=True)
  print(
      f"✓ Peak VRAM during inference: Allocated={infer_alloc:.2f} GiB,"
      f" Reserved={infer_reserved:.2f} GiB\n",
      flush=True,
  )
  print("*** NATIVE DEEPGEN TEST PASSED SUCCESSFULLY! ***", flush=True)


if __name__ == "__main__":
  run_native_deepgen_test()
