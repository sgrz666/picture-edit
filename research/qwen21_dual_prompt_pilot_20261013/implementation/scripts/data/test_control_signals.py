import os
import cv2
import numpy as np
import torch

from src.controls.generator import Camera, ControlSignalGenerator


def create_synthetic_smplx_human(center_x: float, center_z: float):
  """Creates a synthetic 3D human skeleton and pointcloud mesh for testing."""
  # 24 joint 3D positions in meters
  joints = np.array(
      [
          [0.0, 0.0, 0.0],  # 0: pelvis
          [-0.1, -0.1, 0.0],  # 1: left hip
          [0.1, -0.1, 0.0],  # 2: right hip
          [0.0, 0.2, 0.0],  # 3: spine1
          [-0.1, -0.5, 0.0],  # 4: left knee
          [0.1, -0.5, 0.0],  # 5: right knee
          [0.0, 0.4, 0.0],  # 6: spine2
          [-0.1, -0.9, 0.0],  # 7: left ankle
          [0.1, -0.9, 0.0],  # 8: right ankle
          [0.0, 0.6, 0.0],  # 9: spine3
          [-0.1, -0.95, 0.1],  # 10: left foot
          [0.1, -0.95, 0.1],  # 11: right foot
          [0.0, 0.7, 0.0],  # 12: neck
          [-0.15, 0.65, 0.0],  # 13: left collar
          [0.15, 0.65, 0.0],  # 14: right collar
          [0.0, 0.85, 0.0],  # 15: head
          [-0.25, 0.6, 0.0],  # 16: left shoulder
          [0.25, 0.6, 0.0],  # 17: right shoulder
          [-0.35, 0.35, 0.0],  # 18: left elbow
          [0.35, 0.35, 0.0],  # 19: right elbow
          [-0.4, 0.1, 0.0],  # 20: left wrist
          [0.4, 0.1, 0.0],  # 21: right wrist
          [0.0, 0.75, 0.05],  # 22: jaw
          [-0.05, 0.82, 0.08],  # 23: left eye
      ],
      dtype=np.float32,
  )

  # Shift center
  joints[:, 0] += center_x
  joints[:, 2] += center_z

  # Invert Y so up is positive in world coords (Y down in camera coords)
  joints[:, 1] = -joints[:, 1]

  # Generate synthetic surface vertices around joints (cylinder-like body volume)
  verts_list = []
  norms_list = []
  num_pts_per_bone = 60
  for u, v in [
      (0, 1),
      (0, 2),
      (0, 3),
      (1, 4),
      (2, 5),
      (4, 7),
      (5, 8),
      (3, 6),
      (6, 9),
      (9, 12),
      (12, 15),
      (9, 13),
      (9, 14),
      (13, 16),
      (14, 17),
      (16, 18),
      (17, 19),
      (18, 20),
      (19, 21),
  ]:
    p1 = joints[u]
    p2 = joints[v]
    alphas = np.linspace(0, 1, num_pts_per_bone)[:, None]
    line_pts = (1 - alphas) * p1 + alphas * p2
    # Add radial noise around bone
    theta = np.linspace(0, 2 * np.pi, num_pts_per_bone)
    radius = 0.08 if u in (0, 3, 6, 9) else 0.04
    offsets = np.stack(
        [
            radius * np.cos(theta),
            radius * np.sin(theta),
            np.zeros(num_pts_per_bone),
        ],
        axis=-1,
    )
    surface_pts = line_pts + offsets
    verts_list.append(surface_pts)

    # Normals pointing outward
    norms = offsets.copy()
    norms[:, 2] = 0.1
    norms = norms / (np.linalg.norm(norms, axis=-1, keepdims=True) + 1e-6)
    norms_list.append(norms)

  vertices = np.concatenate(verts_list, axis=0).astype(np.float32)
  normals = np.concatenate(norms_list, axis=0).astype(np.float32)

  return {"joints_3d": joints, "vertices": vertices, "normals": normals}


def test_control_signal_generation():
  print("=== TESTING CONTROL SIGNAL GENERATION ===")
  os.makedirs("demooutput", exist_ok=True)

  cam = Camera(
      fx=600.0,
      fy=600.0,
      cx=256.0,
      cy=256.0,
      image_size=(512, 512),
      T=np.array([0.0, 0.0, 2.8], dtype=np.float32),
  )
  generator = ControlSignalGenerator(target_size=(512, 512))

  # Person 0 (left) and Person 1 (right) with mutual hand contact
  p0 = create_synthetic_smplx_human(center_x=-0.2, center_z=0.0)
  p1 = create_synthetic_smplx_human(center_x=0.2, center_z=0.0)

  # Simulate physical hand contact: move p0's right hand close to p1's left hand
  contact_pt = np.array([0.0, 0.0, 2.8], dtype=np.float32)
  p0["vertices"][:20] = (
      contact_pt
      - np.array([0.0, 0.0, 2.8])
      + np.random.randn(20, 3) * 0.01
      + np.array([-0.01, 0.0, 0.0])
  )
  p1["vertices"][:20] = (
      contact_pt
      - np.array([0.0, 0.0, 2.8])
      + np.random.randn(20, 3) * 0.01
      + np.array([0.01, 0.0, 0.0])
  )

  results = generator.generate_all_control_maps([p0, p1], cam)

  # Check dimensions and ranges
  assert results["skeleton_rgb"].shape == (
      512,
      512,
      3,
  ), f"Bad skeleton shape: {results['skeleton_rgb'].shape}"
  assert results["depth_map"].shape == (
      512,
      512,
      1,
  ), f"Bad depth shape: {results['depth_map'].shape}"
  assert results["normal_rgb"].shape == (
      512,
      512,
      3,
  ), f"Bad normal shape: {results['normal_rgb'].shape}"
  assert results["contact_heatmap"].shape == (
      512,
      512,
      1,
  ), f"Bad contact shape: {results['contact_heatmap'].shape}"
  assert results["stacked_control_8ch"].shape == (
      512,
      512,
      8,
  ), f"Bad stacked 8ch shape: {results['stacked_control_8ch'].shape}"

  # Save images for verification
  cv2.imwrite("demooutput/control_skeleton.png", results["skeleton_rgb"])
  depth_vis = (results["depth_map"] * 255).astype(np.uint8)
  cv2.imwrite(
      "demooutput/control_depth.png", cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
  )
  cv2.imwrite("demooutput/control_normal.png", results["normal_rgb"])
  contact_vis = (results["contact_heatmap"] * 255).astype(np.uint8)
  cv2.imwrite(
      "demooutput/control_contact.png",
      cv2.applyColorMap(contact_vis, cv2.COLORMAP_HOT),
  )

  print("✓ Skeleton RGB saved: demooutput/control_skeleton.png")
  print("✓ Depth map saved: demooutput/control_depth.png")
  print("✓ Normal map saved: demooutput/control_normal.png")
  print("✓ Contact heatmap saved: demooutput/control_contact.png")
  print(f"✓ Contact records table: {results['contact_table']}")
  print(
      f"✓ Stacked 8ch control tensor min={results['stacked_control_8ch'].min():.3f},"
      f" max={results['stacked_control_8ch'].max():.3f}"
  )
  print("\n*** CONTROL SIGNAL GENERATOR VERIFICATION PASSED! ***")


if __name__ == "__main__":
  test_control_signal_generation()
