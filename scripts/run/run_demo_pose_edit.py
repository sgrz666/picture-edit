import os
import sys

model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
sys.path.insert(0, model_path)

import time
import torch
from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline

# Configuration
input_dir = "/home/shangguanrz/project/pic-edit/inputs"
output_dir = "/home/shangguanrz/project/pic-edit/demooutput"
os.makedirs(output_dir, exist_ok=True)

neg_prompt = (
    "blurry, low quality, low resolution, distorted, deformed, broken content, "
    "missing parts, damaged details, artifacts, glitch, noise, extra fingers, "
    "missing fingers, mutated hands, bad composition, wrong proportion, unfinished."
)

def make_comparison_grid(src_img: Image.Image, dst_img: Image.Image, title_src="Original Input", title_dst="Edited Pose") -> Image.Image:
    """Concatenate source and edited images side-by-side with nice labels."""
    src_resized = src_img.resize((512, 512))
    dst_resized = dst_img.resize((512, 512))
    grid = Image.new("RGB", (1024, 512 + 40), color=(25, 25, 25))
    grid.paste(src_resized, (0, 40))
    grid.paste(dst_resized, (512, 40))
    draw = ImageDraw.Draw(grid)
    draw.text((20, 12), f"[Input] {title_src}", fill=(220, 220, 220))
    draw.text((532, 12), f"[DeepGen 1.0 Edit] {title_dst}", fill=(50, 255, 150))
    return grid

print("==========================================================", flush=True)
print("  DeepGen 1.0 人物图像姿态编辑（单人 & 双人交互）Demo", flush=True)
print("==========================================================", flush=True)

# 1. Load pipeline
print(f"\n[1/4] Loading DeepGen-1.0-diffusers pipeline from: {model_path}...", flush=True)
t_load_start = time.time()
pipe = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
print(f"Pipeline loaded in {time.time() - t_load_start:.1f}s!", flush=True)
print(f"Allocated VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB", flush=True)

single_img_path = os.path.join(input_dir, "person1.jpg")
dual_img_path = os.path.join(input_dir, "dual_person.jpg")

# 2. Test Case 1: 单人图像姿态编辑
if os.path.exists(single_img_path):
    print("\n[2/4] Testing Single-Person Pose Editing...", flush=True)
    src_single = Image.open(single_img_path).convert("RGB")
    
    # Prompt A: Celebration pose
    prompt_single_a = "Change the person's pose to raise both hands high in celebration, looking joyful with a smile, natural hands, preserving the exact same person identity and clothing style."
    print(f"  Prompt A: '{prompt_single_a}'", flush=True)
    t0 = time.time()
    out_single_a = pipe(
        prompt=prompt_single_a,
        image=src_single,
        negative_prompt=neg_prompt,
        height=512, width=512,
        num_inference_steps=35,
        guidance_scale=4.5,
        seed=1024,
    ).images[0]
    print(f"  Finished in {time.time() - t0:.2f}s!", flush=True)
    out_path_a = os.path.join(output_dir, "demo_single_pose_celebration.png")
    out_single_a.save(out_path_a)
    grid_a = make_comparison_grid(src_single, out_single_a, "Single Person", "Pose: Raise Both Hands in Celebration")
    grid_a.save(os.path.join(output_dir, "comparison_single_pose_celebration.png"))
    print(f"  Saved comparison to: {os.path.join(output_dir, 'comparison_single_pose_celebration.png')}", flush=True)

    # Prompt B: Waving hand
    prompt_single_b = "Change the person's pose to wave one hand in friendly greeting, natural standing posture, maintaining face identity and clothes consistent."
    print(f"  Prompt B: '{prompt_single_b}'", flush=True)
    t0 = time.time()
    out_single_b = pipe(
        prompt=prompt_single_b,
        image=src_single,
        negative_prompt=neg_prompt,
        height=512, width=512,
        num_inference_steps=35,
        guidance_scale=4.5,
        seed=2048,
    ).images[0]
    print(f"  Finished in {time.time() - t0:.2f}s!", flush=True)
    out_path_b = os.path.join(output_dir, "demo_single_pose_wave.png")
    out_single_b.save(out_path_b)
    grid_b = make_comparison_grid(src_single, out_single_b, "Single Person", "Pose: Wave One Hand")
    grid_b.save(os.path.join(output_dir, "comparison_single_pose_wave.png"))
    print(f"  Saved comparison to: {os.path.join(output_dir, 'comparison_single_pose_wave.png')}", flush=True)

# 3. Test Case 2: 双人交互姿态编辑 (Shake Hands & Embrace)
if os.path.exists(dual_img_path):
    print("\n[3/4] Testing Dual-Person Interactive Pose Editing...", flush=True)
    src_dual = Image.open(dual_img_path).convert("RGB")
    
    # Dual Prompt A: Shake Hands
    prompt_dual_a = "Modify the two people to shake hands warmly in friendly greeting, body postures facing each other slightly, natural hands and arms without distortion, maintaining identical identities, clothing and background."
    print(f"  Prompt A: '{prompt_dual_a}'", flush=True)
    t0 = time.time()
    out_dual_a = pipe(
        prompt=prompt_dual_a,
        image=src_dual,
        negative_prompt=neg_prompt,
        height=512, width=512,
        num_inference_steps=35,
        guidance_scale=4.5,
        seed=42,
    ).images[0]
    print(f"  Finished in {time.time() - t0:.2f}s!", flush=True)
    out_dual_path_a = os.path.join(output_dir, "demo_dual_pose_shake_hands.png")
    out_dual_a.save(out_dual_path_a)
    grid_dual_a = make_comparison_grid(src_dual, out_dual_a, "Dual Person", "Interaction: Shaking Hands")
    grid_dual_a.save(os.path.join(output_dir, "comparison_dual_pose_shake_hands.png"))
    print(f"  Saved comparison to: {os.path.join(output_dir, 'comparison_dual_pose_shake_hands.png')}", flush=True)

    # Dual Prompt B: Put arm on shoulder / hug
    prompt_dual_b = "Change the two people's interaction so that the person on the left puts an arm around the right person's shoulder in a warm friendly embrace, natural contact, preserving identity and clothing."
    print(f"  Prompt B: '{prompt_dual_b}'", flush=True)
    t0 = time.time()
    out_dual_b = pipe(
        prompt=prompt_dual_b,
        image=src_dual,
        negative_prompt=neg_prompt,
        height=512, width=512,
        num_inference_steps=35,
        guidance_scale=4.5,
        seed=888,
    ).images[0]
    print(f"  Finished in {time.time() - t0:.2f}s!", flush=True)
    out_dual_path_b = os.path.join(output_dir, "demo_dual_pose_shoulder.png")
    out_dual_b.save(out_dual_path_b)
    grid_dual_b = make_comparison_grid(src_dual, out_dual_b, "Dual Person", "Interaction: Arm Around Shoulder")
    grid_dual_b.save(os.path.join(output_dir, "comparison_dual_pose_shoulder.png"))
    print(f"  Saved comparison to: {os.path.join(output_dir, 'comparison_dual_pose_shoulder.png')}", flush=True)

# 4. Test Case 3: 坐姿编辑
print("\n[4/4] Testing Additional Pose Case (Sitting)...", flush=True)
if os.path.exists(single_img_path):
    prompt_single_c = "Change the person's pose to sit down comfortably on a wooden chair, hands resting on knees, maintaining the same identity and winter outfit."
    t0 = time.time()
    out_single_c = pipe(
        prompt=prompt_single_c,
        image=src_single,
        negative_prompt=neg_prompt,
        height=512, width=512,
        num_inference_steps=35,
        guidance_scale=4.5,
        seed=777,
    ).images[0]
    print(f"  Finished in {time.time() - t0:.2f}s!", flush=True)
    grid_c = make_comparison_grid(src_single, out_single_c, "Single Person", "Pose: Sitting Comfortably")
    grid_c.save(os.path.join(output_dir, "comparison_single_pose_sitting.png"))

print("\n==========================================================", flush=True)
print("  All Demo Tasks Finished Successfully!", flush=True)
print("==========================================================", flush=True)