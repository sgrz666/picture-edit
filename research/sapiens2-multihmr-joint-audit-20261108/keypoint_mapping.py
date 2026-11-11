"""Keypoint Mapping and Kinematic Masking Module for Sapiens 2 x MultiHMR Joint Optimization.

Provides:
- Exact dictionary mapping Goliath 308 keypoints to SMPL-X 144 joints (GOLIATH_TO_SMPLX).
- Bijective 68-point facial landmark dictionary (FACE68_GOLIATH_TO_SMPLX).
- Benchmark evaluation joint index lists (GOLIATH_EVAL_IDS, SMPLX_EVAL_IDS).
- Adaptive confidence and occlusion weighting function (compute_adaptive_weights).
- 63-dimensional arm-chain parameter isolation mask (get_arm_chain_mask).
"""

from typing import Dict, List, Tuple
import torch

# ---------------------------------------------------------------------------
# 1. 12 Hi4D Benchmark Evaluation Joints Alignment
# ---------------------------------------------------------------------------
GOLIATH_EVAL_IDS: List[int] = [5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]
SMPLX_EVAL_IDS: List[int] = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]

# ---------------------------------------------------------------------------
# 2. Bijective 68-Point Facial Landmark Mapping (Goliath 70-307 -> SMPL-X 76-143)
# ---------------------------------------------------------------------------
FACE68_GOLIATH_TO_SMPLX: Dict[int, int] = {
    # Right eyebrow (5): SMPL-X 76..80
    80: 76,  # end_of_r_eyebrow -> right_eye_brow1 (outer)
    84: 77,  # upper_midpoint_3_of_r_eyebrow -> right_eye_brow2
    83: 78,  # upper_midpoint_2_of_r_eyebrow -> right_eye_brow3
    81: 79,  # upper_midpoint_1_of_r_eyebrow -> right_eye_brow4
    78: 80,  # upper_startpoint_of_r_eyebrow -> right_eye_brow5 (inner)
    # Left eyebrow (5): SMPL-X 81..85
    87: 81,  # upper_startpoint_of_l_eyebrow -> left_eye_brow5 (inner)
    90: 82,  # upper_midpoint_1_of_l_eyebrow -> left_eye_brow4
    92: 83,  # upper_midpoint_2_of_l_eyebrow -> left_eye_brow3
    93: 84,  # upper_midpoint_3_of_l_eyebrow -> left_eye_brow2
    89: 85,  # end_of_l_eyebrow -> left_eye_brow1 (outer)
    # Nose bridge (4): SMPL-X 86..89
    71: 86,  # center_of_nose_root -> nose1
    73: 87,  # midpoint_1_of_nose_bridge -> nose2
    74: 88,  # midpoint_2_of_nose_bridge -> nose3
    72: 89,  # tip_of_nose_bridge -> nose4
    # Lower nose / nostrils (5): SMPL-X 90..94
    180: 90,  # r_outer_corner_of_nose -> right_nose_2
    182: 91,  # inner_corner_of_r_nostril -> right_nose_1
    178: 92,  # tip_of_nose -> nose_middle
    185: 93,  # inner_corner_of_l_nostril -> left_nose_1
    181: 94,  # l_outer_corner_of_nose -> left_nose_2
    # Right eye (6): SMPL-X 95..100
    121: 95,  # r_outer_end_of_upper_lash_line -> right_eye1
    124: 96,  # r_midpoint_2_of_upper_lash_line -> right_eye2
    123: 97,  # r_midpoint_1_of_upper_lash_line -> right_eye3
    120: 98,  # r_inner_end_of_upper_lash_line -> right_eye4
    166: 99,  # r_midpoint_1_of_lower_lash_line -> right_eye5
    165: 100,  # r_midpoint_2_of_lower_lash_line -> right_eye6
    # Left eye (6): SMPL-X 101..106
    96: 101,  # l_inner_end_of_upper_lash_line -> left_eye4
    100: 102,  # l_midpoint_1_of_upper_lash_line -> left_eye3
    99: 103,  # l_midpoint_2_of_upper_lash_line -> left_eye2
    97: 104,  # l_outer_end_of_upper_lash_line -> left_eye1
    147: 105,  # l_midpoint_2_of_lower_lash_line -> left_eye6
    148: 106,  # l_midpoint_1_of_lower_lash_line -> left_eye5
    # Outer mouth contour (12): SMPL-X 107..118
    188: 107,  # r_outer_corner_of_mouth -> right_mouth_1
    192: 108,  # midpoint_1_of_upper_outer_lip -> right_mouth_2
    193: 109,  # midpoint_2_of_upper_outer_lip -> right_mouth_3
    190: 110,  # center_of_cupid_bow -> mouth_top
    196: 111,  # midpoint_3_of_upper_outer_lip -> left_mouth_3
    197: 112,  # midpoint_4_of_upper_outer_lip -> left_mouth_2
    189: 113,  # l_outer_corner_of_mouth -> left_mouth_1
    201: 114,  # midpoint_4_of_lower_outer_lip -> left_mouth_5
    200: 115,  # midpoint_3_of_lower_outer_lip -> left_mouth_4
    191: 116,  # center_of_lower_outer_lip -> mouth_bottom
    195: 117,  # midpoint_2_of_lower_outer_lip -> right_mouth_4
    194: 118,  # midpoint_1_of_lower_outer_lip -> right_mouth_5
    # Inner lip contour (8): SMPL-X 119..126
    204: 119,  # r_inner_corner_of_mouth -> right_lip_1
    208: 120,  # midpoint_1_of_upper_inner_lip -> right_lip_2
    206: 121,  # center_of_upper_inner_lip -> lip_top
    209: 122,  # midpoint_2_of_upper_inner_lip -> left_lip_2
    205: 123,  # l_inner_corner_of_mouth -> left_lip_1
    211: 124,  # midpoint_2_of_lower_inner_lip -> left_lip_3
    207: 125,  # center_of_lower_inner_lip -> lip_bottom
    210: 126,  # midpoint_1_of_lower_inner_lip -> right_lip_3
    # Jawline / Face contour (17): SMPL-X 127..143
    269: 127,  # r_bottom_connecting_point_of_ear_lobe -> right_contour_1
    267: 128,  # r_tip_of_ear_lobe -> right_contour_2
    262: 129,  # r_bottom_point_of_tragus -> right_contour_3
    263: 130,  # r_protruding_point_of_tragus -> right_contour_4
    261: 131,  # r_highest_point_of_antitragus -> right_contour_5
    248: 132,  # r_start_of_antihelix -> right_contour_6
    250: 133,  # r_midpoint_1_of_antihelix -> right_contour_7
    76: 134,  # center_of_labiomental_groove -> right_contour_8
    77: 135,  # tip_of_chin -> contour_middle
    70: 136,  # center_of_glabella -> left_contour_8
    224: 137,  # l_midpoint_1_of_antihelix -> left_contour_7
    222: 138,  # l_start_of_antihelix -> left_contour_6
    235: 139,  # l_highest_point_of_antitragus -> left_contour_5
    237: 140,  # l_protruding_point_of_tragus -> left_contour_4
    236: 141,  # l_bottom_point_of_tragus -> left_contour_3
    241: 142,  # l_tip_of_ear_lobe -> left_contour_2
    243: 143,  # l_bottom_connecting_point_of_ear_lobe -> left_contour_1
}

# ---------------------------------------------------------------------------
# 3. Complete Goliath 308 -> SMPL-X 144 Joint Mapping Dictionary
# ---------------------------------------------------------------------------
GOLIATH_TO_SMPLX: Dict[int, int] = {
    # Head / Face keypoints (5)
    0: 55,  # nose -> nose
    1: 57,  # left_eye -> left_eye
    2: 56,  # right_eye -> right_eye
    3: 59,  # left_ear -> left_ear
    4: 58,  # right_ear -> right_ear
    # Torso & Limbs (12)
    5: 16,  # left_shoulder -> left_shoulder
    6: 17,  # right_shoulder -> right_shoulder
    7: 18,  # left_elbow -> left_elbow
    8: 19,  # right_elbow -> right_elbow
    62: 20,  # left_wrist -> left_wrist
    41: 21,  # right_wrist -> right_wrist
    9: 1,  # left_hip -> left_hip
    10: 2,  # right_hip -> right_hip
    11: 4,  # left_knee -> left_knee
    12: 5,  # right_knee -> right_knee
    13: 7,  # left_ankle -> left_ankle
    14: 8,  # right_ankle -> right_ankle
    # Feet (6)
    15: 60,  # left_big_toe -> left_big_toe
    16: 61,  # left_small_toe -> left_small_toe
    17: 62,  # left_heel -> left_heel
    18: 63,  # right_big_toe -> right_big_toe
    19: 64,  # right_small_toe -> right_small_toe
    20: 65,  # right_heel -> right_heel
    # Right Hand: 15 finger joints (Goliath 21..40 -> SMPL-X 40..54)
    # Forefinger (Index)
    28: 40,  # right_forefinger_third_joint -> right_index1
    27: 41,  # right_forefinger2 -> right_index2
    26: 42,  # right_forefinger3 -> right_index3
    # Middle finger
    32: 43,  # right_middle_finger_third_joint -> right_middle1
    31: 44,  # right_middle_finger2 -> right_middle2
    30: 45,  # right_middle_finger3 -> right_middle3
    # Pinky finger
    40: 46,  # right_pinky_finger_third_joint -> right_pinky1
    39: 47,  # right_pinky_finger2 -> right_pinky2
    38: 48,  # right_pinky_finger3 -> right_pinky3
    # Ring finger
    36: 49,  # right_ring_finger_third_joint -> right_ring1
    35: 50,  # right_ring_finger2 -> right_ring2
    34: 51,  # right_ring_finger3 -> right_ring3
    # Thumb
    24: 52,  # right_thumb_third_joint -> right_thumb1
    23: 53,  # right_thumb2 -> right_thumb2
    22: 54,  # right_thumb3 -> right_thumb3
    # Left Hand: 15 finger joints (Goliath 42..61 -> SMPL-X 25..39)
    # Forefinger (Index)
    49: 25,  # left_forefinger_third_joint -> left_index1
    48: 26,  # left_forefinger2 -> left_index2
    47: 27,  # left_forefinger3 -> left_index3
    # Middle finger
    53: 28,  # left_middle_finger_third_joint -> left_middle1
    52: 29,  # left_middle_finger2 -> left_middle2
    51: 30,  # left_middle_finger3 -> left_middle3
    # Pinky finger
    61: 31,  # left_pinky_finger_third_joint -> left_pinky1
    60: 32,  # left_pinky_finger2 -> left_pinky2
    59: 33,  # left_pinky_finger3 -> left_pinky3
    # Ring finger
    57: 34,  # left_ring_finger_third_joint -> left_ring1
    56: 35,  # left_ring_finger2 -> left_ring2
    55: 36,  # left_ring_finger3 -> left_ring3
    # Thumb
    45: 37,  # left_thumb_third_joint -> left_thumb1
    44: 38,  # left_thumb2 -> left_thumb2
    43: 39,  # left_thumb3 -> left_thumb3
}

# Integrate 68 face contour points into global mapping
GOLIATH_TO_SMPLX.update(FACE68_GOLIATH_TO_SMPLX)


# ---------------------------------------------------------------------------
# 4. Adaptive Confidence & Occlusion Weighting
# ---------------------------------------------------------------------------
def compute_adaptive_weights(
    scores: torch.Tensor,
    region: str = "body",
    occlusion_state: str = "visible",
    conf_threshold: float = 0.3,
) -> torch.Tensor:
  """Computes adaptive loss weights based on Sapiens confidences, body region, and occlusion.

  Args:
      scores: Float tensor of confidence scores in [0.0, 1.0].
      region: Region string ('body', 'hand', 'face', 'foot').
      occlusion_state: Occlusion state string ('visible', 'partial', 'behind').
      conf_threshold: Cutoff threshold (default 0.3). Scores < threshold are
        zeroed out.

  Returns:
      Tensor of weights matching scores shape and device.
  """
  valid_states = {"visible": 1.0, "partial": 0.3, "behind": 0.0}
  if occlusion_state not in valid_states:
    raise ValueError(
        f"Invalid occlusion_state '{occlusion_state}'. Must be one of"
        f" {list(valid_states.keys())}"
    )

  region_scales = {
      "body": 1.0,
      "hand": 2.5,
      "face": 1.5,
      "foot": 1.2,
  }
  r_scale = region_scales.get(region, 1.0)
  occ_factor = valid_states[occlusion_state]

  # Base weight scaled by region and occlusion factor
  weights = scores.clone().float() * (r_scale * occ_factor)

  # Cutoff threshold filtering (< conf_threshold -> 0.0)
  weights = torch.where(
      scores >= conf_threshold, weights, torch.zeros_like(weights)
  )

  # Fast short-circuit for complete occlusion
  if occlusion_state == "behind":
    weights.zero_()

  return weights


# ---------------------------------------------------------------------------
# 5. Arm-Chain Masking (63-dim body pose parameter isolation)
# ---------------------------------------------------------------------------
def get_arm_chain_mask(side: str) -> torch.Tensor:
  """Generates a 63-dimensional binary mask isolating shoulder, elbow, and wrist rotations in the SMPL-X body pose vector.

  In SMPL-X, body_pose contains 21 joints (excluding root pelvis) x 3
  axis-angles = 63 dims:
  - Left arm: joints 15 (shoulder), 17 (elbow), 19 (wrist) -> indices [45:48],
  [51:54], [57:60]
  - Right arm: joints 16 (shoulder), 18 (elbow), 20 (wrist) -> indices [48:51],
  [54:57], [60:63]

  Args:
      side: Arm side ('left' or 'right').

  Returns:
      Float32 tensor of shape (63,) with 1.0 for the active arm chain and 0.0
      elsewhere.
  """
  if side not in ("left", "right"):
    raise ValueError(f"Invalid side '{side}'. Must be 'left' or 'right'.")

  mask = torch.zeros(63, dtype=torch.float32)
  joints = (15, 17, 19) if side == "left" else (16, 18, 20)
  for j in joints:
    mask[3 * j : 3 * j + 3] = 1.0

  return mask
