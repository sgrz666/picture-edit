import json
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.controls.generator import Camera, ControlSignalGenerator


def generate_debug_dataset(num_pairs: int = 16):
  base_dir = "/home/shangguanrz/project/pic-edit/data/debug_samples"
  images_dir = os.path.join(base_dir, "images")
  controls_dir = os.path.join(base_dir, "controls")
  os.makedirs(images_dir, exist_ok=True)
  os.makedirs(controls_dir, exist_ok=True)

  cam = Camera(
      fx=600.0,
      fy=600.0,
      cx=256.0,
      cy=256.0,
      image_size=(512, 512),
      T=np.array([0.0, 0.0, 2.8], dtype=np.float32),
  )
  generator = ControlSignalGenerator(target_size=(512, 512))

  manifest = []

  # Actions list
  actions = [
      ("standing_straight", "raise_right_arm", "A person raising right arm"),
      ("standing_straight", "raise_both_arms", "A person raising both arms"),
      ("arms_crossed", "open_arms", "A person opening both arms wide"),
      (
          "standing_side_by_side",
          "two_person_handshake",
          "Two persons shaking hands",
      ),
      ("two_person_apart", "two_person_hug", "Two persons hugging"),
      (
          "standing_side_by_side",
          "holding_hands",
          "Two persons holding hands together",
      ),
      ("standing", "sitting_down", "A person sitting down"),
      ("waving", "standing_peace", "A person showing peace sign"),
  ]

  for idx in range(num_pairs):
    action_info = actions[idx % len(actions)]
    src_action, tgt_action, prompt_text = action_info
    sample_id = f"sample_{idx:03d}"

    is_two_person = "two_person" in tgt_action or "holding_hands" in tgt_action

    # 1. Generate Synthetic Source Image
    src_img = Image.new("RGB", (512, 512), color=(245, 245, 248))
    draw = ImageDraw.Draw(src_img)

    if not is_two_person:
      # Single person
      draw.ellipse([230, 80, 282, 132], fill=(225, 185, 155))  # Head
      draw.rectangle(
          [220, 132, 292, 290], fill=(60, 120, 210)
      )  # Blue shirt
      draw.rectangle([215, 290, 250, 460], fill=(45, 45, 65))  # Left leg
      draw.rectangle([262, 290, 297, 460], fill=(45, 45, 65))  # Right leg
      draw.line([(220, 150), (180, 270)], fill=(60, 120, 210), width=16)
      draw.line([(292, 150), (332, 270)], fill=(60, 120, 210), width=16)
    else:
      # Two persons
      # Person A (left)
      draw.ellipse([160, 90, 205, 135], fill=(225, 185, 155))
      draw.rectangle(
          [150, 135, 215, 290], fill=(210, 80, 80)
      )  # Red shirt
      draw.rectangle([145, 290, 175, 460], fill=(45, 45, 65))
      draw.rectangle([185, 290, 215, 460], fill=(45, 45, 65))
      draw.line([(150, 150), (120, 270)], fill=(210, 80, 80), width=14)
      draw.line([(215, 150), (250, 230)], fill=(210, 80, 80), width=14)

      # Person B (right)
      draw.ellipse([305, 90, 350, 135], fill=(225, 185, 155))
      draw.rectangle(
          [295, 135, 360, 290], fill=(60, 120, 210)
      )  # Blue shirt
      draw.rectangle([290, 290, 320, 460], fill=(45, 45, 65))
      draw.rectangle([330, 290, 360, 460], fill=(45, 45, 65))
      draw.line([(295, 150), (260, 230)], fill=(60, 120, 210), width=14)
      draw.line([(360, 150), (390, 270)], fill=(60, 120, 210), width=14)

    src_filename = f"{sample_id}_src.png"
    src_path = os.path.join(images_dir, src_filename)
    src_img.save(src_path)

    # 2. Generate Synthetic Target Image (Edited Pose)
    tgt_img = Image.new("RGB", (512, 512), color=(245, 245, 248))
    draw_tgt = ImageDraw.Draw(tgt_img)
    if not is_two_person:
      draw_tgt.ellipse([230, 80, 282, 132], fill=(225, 185, 155))
      draw_tgt.rectangle([220, 132, 292, 290], fill=(60, 120, 210))
      draw_tgt.rectangle([215, 290, 250, 460], fill=(45, 45, 65))
      draw_tgt.rectangle([262, 290, 297, 460], fill=(45, 45, 65))
      # Arms in target pose (raising)
      draw_tgt.line([(220, 150), (170, 70)], fill=(60, 120, 210), width=16)
      draw_tgt.line([(292, 150), (342, 70)], fill=(60, 120, 210), width=16)
    else:
      # Two persons interacting (hands touching in center)
      draw_tgt.ellipse([160, 90, 205, 135], fill=(225, 185, 155))
      draw_tgt.rectangle([150, 135, 215, 290], fill=(210, 80, 80))
      draw_tgt.rectangle([145, 290, 175, 460], fill=(45, 45, 65))
      draw_tgt.rectangle([185, 290, 215, 460], fill=(45, 45, 65))
      draw_tgt.line([(150, 150), (120, 270)], fill=(210, 80, 80), width=14)
      draw_tgt.line(
          [(215, 150), (256, 210)], fill=(210, 80, 80), width=14
      )  # right hand extends to center

      draw_tgt.ellipse([305, 90, 350, 135], fill=(225, 185, 155))
      draw_tgt.rectangle([295, 135, 360, 290], fill=(60, 120, 210))
      draw_tgt.rectangle([290, 290, 320, 460], fill=(45, 45, 65))
      draw_tgt.rectangle([330, 290, 360, 460], fill=(45, 45, 65))
      draw_tgt.line(
          [(295, 150), (256, 210)], fill=(60, 120, 210), width=14
      )  # left hand meets in center
      draw_tgt.line([(360, 150), (390, 270)], fill=(60, 120, 210), width=14)
      # Touch circle
      draw_tgt.ellipse([248, 202, 264, 218], fill=(225, 185, 155))

    tgt_filename = f"{sample_id}_tgt.png"
    tgt_path = os.path.join(images_dir, tgt_filename)
    tgt_img.save(tgt_path)

    # 3. Generate Matching 8-Channel Control Map
    from test_control_signals import create_synthetic_smplx_human

    if is_two_person:
      p0 = create_synthetic_smplx_human(-0.15, 0.0)
      p1 = create_synthetic_smplx_human(0.15, 0.0)
      ctrl_res = generator.generate_all_control_maps([p0, p1], cam)
    else:
      p0 = create_synthetic_smplx_human(0.0, 0.0)
      ctrl_res = generator.generate_all_control_maps([p0], cam)

    ctrl_filename = f"{sample_id}_control8ch.npy"
    ctrl_path = os.path.join(controls_dir, ctrl_filename)
    np.save(ctrl_path, ctrl_res["stacked_control_8ch"])

    manifest.append({
        "sample_id": sample_id,
        "src_image": src_filename,
        "tgt_image": tgt_filename,
        "control_tensor": ctrl_filename,
        "prompt": prompt_text,
        "is_two_person": is_two_person,
        "has_contact": len(ctrl_res["contact_table"]) > 0,
        "contact_records": ctrl_res["contact_table"],
    })

  manifest_path = os.path.join(base_dir, "pairs.json")
  with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

  print(
      f"✓ Staged {len(manifest)} debug sample pairs into {base_dir}", flush=True
  )
  print(f"✓ Manifest written to {manifest_path}", flush=True)


if __name__ == "__main__":
  generate_debug_dataset(16)
