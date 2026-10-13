import _workflow_paths  # noqa: F401 - puts the workflow and repo roots on sys.path for unittest discovery
import ast
import json
from pathlib import Path
import unittest

import numpy as np

from src.pose_edit.geometry import (
    align_to_source_bbox,
    project_camera,
    render_poseguide,
    smplx137_to_dwpose134,
    valid_points,
)


class GeometryTests(unittest.TestCase):
    def test_project_camera_uses_pinhole_intrinsics_and_marks_invalid_depth(self):
        points = np.array([[2.0, 4.0, 2.0], [-1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [0.0, 0.0, -1.0]])
        actual = project_camera(points, (100.0, 200.0), (10.0, 20.0))
        expected = np.array([
            [2.0 / 2.0001 * 100 + 10, 4.0 / 2.0001 * 200 + 20],
            [-1.0 / 1.0001 * 100 + 10, 1.0 / 1.0001 * 200 + 20],
            [np.nan, np.nan],
            [np.nan, np.nan],
        ])
        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_projection_rejects_wrong_shapes(self):
        with self.assertRaises(ValueError):
            project_camera(np.zeros((2, 4)), (10, 10), (0, 0))

    def test_projection_rejects_invalid_camera_parameters(self):
        points = np.array([[1.0, 1.0, 1.0]])
        with self.assertRaises(ValueError):
            project_camera(points, (0.0, 10.0), (0, 0))
        with self.assertRaises(ValueError):
            project_camera(points, (10.0, 10.0), (np.nan, 0))

    def test_mapping_preserves_wrist_and_face_landmark_semantics(self):
        points = np.zeros((137, 3), dtype=float)
        points[:, 0] = np.arange(137)
        mapped = smplx137_to_dwpose134(points)
        self.assertEqual(mapped.shape, (134, 3))
        # DWPose body order starts with nose/neck and then right arm before left arm.
        self.assertEqual(mapped[:18, 0].tolist(), [24, 7, 9, 11, 13, 8, 10, 12, 2, 4, 6, 1, 3, 5, 23, 21, 22, 20])
        # Each hand starts at its corresponding body wrist and then keeps its own fingers.
        self.assertEqual(mapped[92:97, 0].tolist(), [12, 25, 26, 27, 28])
        self.assertEqual(mapped[113:118, 0].tolist(), [13, 45, 46, 47, 48])
        self.assertEqual(mapped[18:24, 0].tolist(), [14, 15, 16, 17, 18, 19])
        # Face order is contour, eyebrows, nose, eyes, mouth.
        self.assertEqual(mapped[24:41, 0].tolist(), list(range(120, 137)))
        self.assertEqual(mapped[41:51, 0].tolist(), list(range(69, 79)))
        self.assertEqual(mapped[51:60, 0].tolist(), list(range(79, 88)))
        self.assertEqual(mapped[60:72, 0].tolist(), list(range(88, 100)))
        self.assertEqual(mapped[72:92, 0].tolist(), list(range(100, 120)))

    def test_mapping_accepts_2d_and_rejects_other_joint_shapes(self):
        self.assertEqual(smplx137_to_dwpose134(np.zeros((137, 2))).shape, (134, 2))
        for points in (np.zeros((136, 3)), np.zeros((137, 4)), np.zeros((2, 137, 3))):
            with self.subTest(shape=points.shape), self.assertRaises(ValueError):
                smplx137_to_dwpose134(points)

    def test_align_bbox_uses_one_scale_and_preserves_angles_and_nan(self):
        points = np.array([[10.0, 20.0], [20.0, 20.0], [10.0, 30.0], [np.nan, 4.0]])
        aligned, metadata = align_to_source_bbox(points, (0, 0, 40, 20), (10, 10, 70, 70))
        self.assertAlmostEqual(metadata["scale"], 1.5)
        np.testing.assert_allclose(aligned[:3], [[25, 55], [40, 55], [25, 70]])
        self.assertTrue(np.isnan(aligned[3]).all())
        self.assertAlmostEqual(np.dot(aligned[1] - aligned[0], aligned[2] - aligned[0]), 0.0)

    def test_valid_points_requires_finite_in_canvas_coordinates(self):
        points = np.array([[0, 0], [9.9, 4.9], [10, 2], [2, 5], [-0.01, 2], [np.nan, 0], [0, np.inf]])
        np.testing.assert_array_equal(valid_points(points, 10, 5), [True, True, False, False, False, False, False])

    def test_render_does_not_draw_limb_through_missing_joint(self):
        points = np.full((134, 2), np.nan)
        points[1] = [10, 10]  # neck
        points[2] = [30, 10]  # right shoulder
        # The elbow is missing, so the renderer must not bridge shoulder to wrist.
        points[4] = [70, 10]  # right wrist
        image = render_poseguide(points, (80, 30))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (80, 30))
        self.assertEqual(image.getpixel((50, 10)), (0, 0, 0))
        self.assertNotEqual(image.getpixel((30, 10)), (0, 0, 0))

    def test_render_skips_segments_with_out_of_canvas_endpoint(self):
        points = np.full((134, 2), np.nan)
        points[1] = [10, 10]
        points[2] = [90, 10]
        image = render_poseguide(points, (80, 30))
        self.assertEqual(image.getpixel((50, 10)), (0, 0, 0))

    def test_render_connects_head_landmarks_and_ankles_to_feet(self):
        points = np.full((134, 2), np.nan)
        points[0] = [10, 10]  # nose
        points[14] = [30, 10]  # right eye
        points[15] = [50, 10]  # right ear
        points[13] = [10, 50]  # left ankle
        points[18] = [30, 50]  # left big toe
        points[20] = [50, 50]  # left heel
        points[10] = [10, 70]  # right ankle
        points[21] = [30, 70]  # right big toe
        points[23] = [50, 70]  # right heel
        image = render_poseguide(points, (70, 90))
        for pixel in ((20, 10), (40, 10), (20, 50), (20, 70)):
            with self.subTest(pixel=pixel):
                self.assertNotEqual(image.getpixel(pixel), (0, 0, 0))

    def test_render_uses_distinct_colors_for_hand_digits(self):
        points = np.full((134, 2), np.nan)
        points[92] = [20, 20]
        points[93] = [30, 20]  # thumb first joint
        points[97] = [30, 40]  # index first joint
        image = render_poseguide(points, (60, 60))
        self.assertNotEqual(image.getpixel((25, 20)), image.getpixel((25, 30)))


DWPose_BODY18 = ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow',
    'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
    'R_Eye', 'R_Ear', 'L_Eye', 'L_Ear')
DWPose_FEET6 = ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel')


def vendor_tuple(source, attribute):
    """Read a tuple literal out of the installed SMPLest-X tables without importing torch."""
    lines = source.splitlines()
    start = next(index for index, line in enumerate(lines)
        if line.strip().startswith(f'self.{attribute} ='))
    body = []
    for line in lines[start + 1:]:
        if line.strip() == ')':
            break
        body.append(line.split('#')[0])
    text = '\n'.join(body).split('*[')[0].strip().rstrip(',')
    return ast.literal_eval(text + ')')


class InstalledEstimatorLayoutTests(unittest.TestCase):
    """Bind the 137->134 table to the joint names it claims, not to a copy of itself.

    The mapping constant is a hand-transcribed table, so re-asserting the same numbers in a
    test proves nothing. These cases read the installed SMPLest-X ``joints_name`` and
    ``flip_pairs`` straight from its source file.
    """

    @classmethod
    def setUpClass(cls):
        workflow_root = Path(__file__).resolve().parents[1]
        config = json.loads((workflow_root / 'config.json').read_text(encoding='utf-8'))
        source_path = Path(config['smplest_root']) / 'human_models/human_models.py'
        if not source_path.is_file():
            raise unittest.SkipTest(f'installed SMPLest-X layout is not readable: {source_path}')
        source = source_path.read_text(encoding='utf-8', errors='replace')
        cls.names = vendor_tuple(source, 'joints_name')
        cls.flips = vendor_tuple(source, 'flip_pairs')

    def slots(self):
        points = np.zeros((137, 3), dtype=float)
        points[:, 0] = np.arange(137)
        return [int(value) for value in smplx137_to_dwpose134(points)[:, 0]]

    def test_body_slots_are_the_dwpose_18_names_in_order(self):
        self.assertEqual(tuple(self.names[index] for index in self.slots()[:18]), DWPose_BODY18)

    def test_foot_slots_carry_left_then_right_toe_and_heel_names(self):
        self.assertEqual(tuple(self.names[index] for index in self.slots()[18:24]), DWPose_FEET6)

    def test_each_hand_block_is_wrist_then_thumb_index_middle_ring_pinky(self):
        slots = self.slots()
        for side, block in (('L', slice(92, 113)), ('R', slice(113, 134))):
            digits = ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky')
            expected = (f'{side}_Wrist',) + tuple(f'{side}_{digit}_{phalanx}'
                for digit in digits for phalanx in range(1, 5))
            with self.subTest(side=side):
                self.assertEqual(tuple(self.names[index] for index in slots[block]), expected)

    def test_face_blocks_are_symmetric_contiguous_ranges_of_the_vendor_table(self):
        paired = {index: partner for group in self.flips for index, partner in
            ((group[0], group[1]), (group[1], group[0])) if isinstance(group, tuple)
            and len(group) == 2}
        slots = self.slots()[24:92]
        boundaries = [(0, 17), (17, 27), (27, 36), (36, 48), (48, 68)]
        self.assertEqual(len(slots), 68)
        self.assertEqual(len(set(slots)), 68)
        for start, end in boundaries:
            block = slots[start:end]
            block_set = set(block)
            with self.subTest(block=f'{start}:{end}'):
                self.assertEqual(sorted(block), list(range(block[0], block[-1] + 1)),
                    'each face block must stay one contiguous vendor range')
                self.assertTrue(all(65 <= index < 137 for index in block))
                # Centre-line landmarks legitimately have no partner; a partner in another
                # block would mean the two halves of one feature were split apart.
                crossed = [index for index in block
                    if index in paired and paired[index] not in block_set]
                self.assertEqual(crossed, [],
                    'a left/right landmark pair must stay inside one face block')

    def test_no_joint_is_dropped_or_invented_between_137_and_134(self):
        slots = self.slots()
        self.assertTrue(set(slots) <= set(range(137)))
        repeated = sorted({index for index in slots if slots.count(index) == 2})
        self.assertEqual(repeated,
            [self.names.index('L_Wrist'), self.names.index('R_Wrist')],
            'only the two wrists repeat, once in the body chain and once per hand')


class SubjectSelectionTests(unittest.TestCase):
    """A multi-person photo must be an explicit choice, never a silent guess."""

    def setUp(self):
        from src.pose_edit.geometry import choose_subject
        self.choose = choose_subject
        # Largest box first is NOT the detection order here, so ordering has to be derived.
        self.boxes = np.array([[86.0, 270.0, 350.0, 1278.0],
            [246.0, 67.0, 725.0, 1280.0],
            [671.0, 703.0, 975.0, 1279.0]])
        self.scores = np.array([0.85, 0.915, 0.892])

    def test_auto_accepts_the_only_person(self):
        self.assertEqual(self.choose(self.boxes[:1], self.scores[:1], 'auto'), 0)

    def test_auto_refuses_to_pick_between_several_people(self):
        with self.assertRaisesRegex(ValueError, '--subject'):
            self.choose(self.boxes, self.scores, 'auto')

    def test_largest_ignores_detection_order(self):
        self.assertEqual(self.choose(self.boxes, self.scores, 'largest'), 1)
        shuffled = np.array([self.boxes[2], self.boxes[0], self.boxes[1]])
        self.assertEqual(self.choose(shuffled, self.scores[[2, 0, 1]], 'largest'), 2)

    def test_index_walks_the_largest_first_ordering(self):
        # Areas: box1 581k > box0 266k > box2 175k, so the index walks 1, 0, 2.
        self.assertEqual(self.choose(self.boxes, self.scores, '0'), 1)
        self.assertEqual(self.choose(self.boxes, self.scores, '1'), 0)
        self.assertEqual(self.choose(self.boxes, self.scores, '2'), 2)
        with self.assertRaises(ValueError):
            self.choose(self.boxes, self.scores, '3')

    def test_bad_rules_and_boxes_are_rejected(self):
        for rule in ('biggest', '', '-1', None):
            with self.subTest(rule=rule), self.assertRaises(ValueError):
                self.choose(self.boxes, self.scores, rule)
        with self.assertRaises(ValueError):
            self.choose(np.zeros((0, 4)), np.zeros(0), 'auto')
        with self.assertRaises(ValueError):
            self.choose(np.array([[10.0, 10.0, 10.0, 40.0]]), np.array([0.9]), 'largest')


class TorsoFramingTests(unittest.TestCase):
    """Box-fit framing collapses when A and B are cropped with opposite aspect ratios."""

    def setUp(self):
        from src.pose_edit.geometry import TORSO_TO_PERSON_BOX_HEIGHT, align_to_source_bbox
        self.align = align_to_source_bbox
        self.constant = TORSO_TO_PERSON_BOX_HEIGHT
        # A DWPose-134 array whose neck is 100px above the hip centre, plus one missing joint.
        self.points = np.full((134, 2), np.nan)
        self.points[1] = [50.0, 100.0]     # neck
        self.points[8] = [40.0, 200.0]     # right hip
        self.points[11] = [60.0, 200.0]    # left hip
        self.points[0] = [50.0, 80.0]      # nose
        # B is a wide crop, A is a tall narrow one: the hard_05 configuration.
        self.bbox_B = [117.0, 428.0, 824.0, 1235.0]
        self.bbox_A = [100.0, 100.0, 300.0, 900.0]

    def test_torso_length_is_neck_to_hip_centre(self):
        from src.pose_edit.geometry import torso_length
        self.assertAlmostEqual(torso_length(self.points), 100.0)
        missing = np.full((134, 2), np.nan)
        self.assertTrue(np.isnan(torso_length(missing)))

    def test_person_framing_shrinks_the_pose_when_the_boxes_disagree(self):
        _, person = self.align(self.points, self.bbox_B, self.bbox_A, 'source_person')
        _, torso = self.align(self.points, self.bbox_B, self.bbox_A, 'source_torso')
        self.assertAlmostEqual(person['scale'], 200.0 / 707.0)
        self.assertAlmostEqual(torso['scale'], self.constant * 800.0 / 100.0)
        self.assertGreater(torso['scale'], person['scale'] * 5,
            'a wide B inside a narrow A must not be fitted by the width ratio')

    def test_torso_scale_is_derived_from_the_source_box_and_the_measured_torso(self):
        _, metadata = self.align(self.points, self.bbox_B, self.bbox_A, 'source_torso')
        self.assertEqual(metadata['framing'], 'source_torso')
        self.assertAlmostEqual(metadata['B_torso_px'], 100.0)
        self.assertAlmostEqual(metadata['A_box_height_px'], 800.0)
        self.assertAlmostEqual(metadata['scale'], self.constant * 800.0 / 100.0)

    def test_both_framings_keep_proportions_and_missing_joints(self):
        for framing in ('source_person', 'source_torso'):
            aligned, metadata = self.align(self.points, self.bbox_B, self.bbox_A, framing)
            with self.subTest(framing=framing):
                self.assertTrue(np.isnan(aligned[2]).all())
                scale = metadata['scale']
                neck, hips = aligned[1], np.mean(aligned[[8, 11]], axis=0)
                self.assertAlmostEqual(float(np.linalg.norm(aligned[0] - neck))
                    / float(np.linalg.norm(neck - hips)), 20.0 / 100.0, places=9)
                centre_a = np.array([200.0, 500.0])
                centre_b = np.array([(117.0 + 824.0) / 2, (428.0 + 1235.0) / 2])
                offset = centre_a - scale * centre_b
                np.testing.assert_allclose(aligned[1], self.points[1] * scale + offset)
                np.testing.assert_allclose(aligned[8], self.points[8] * scale + offset)

    def test_unknown_framing_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unknown pose framing'):
            self.align(self.points, self.bbox_B, self.bbox_A, 'source_head')

    def test_torso_framing_needs_a_neck_and_a_hip(self):
        blank = np.full((134, 2), np.nan)
        with self.assertRaisesRegex(ValueError, 'source_torso framing needs'):
            self.align(blank, self.bbox_B, self.bbox_A, 'source_torso')


if __name__ == "__main__":
    unittest.main()
