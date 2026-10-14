import importlib
import unittest

import _workflow_multiple_paths  # noqa: F401 - puts the workflow and repo roots on sys.path


class DualPromptContractTests(unittest.TestCase):
    """Four images in, and the generator may only be pointed at the ones it receives."""

    def setUp(self):
        self.module = importlib.import_module('dual_prompts')

    def test_with_b_keeps_all_four_reference_numbers(self):
        compiled = self.module.compile_prompt(
            'Move the image-left person of <image1> into the embrace shown by <image2>, '
            'following the joint arrangement of <image3> and <image4>.', 'with_b')
        self.assertIn('<image1>', compiled)
        self.assertIn('<image2>', compiled)
        self.assertIn('<image3>', compiled)
        self.assertIn('<image4>', compiled)
        self.assertIn('Do not add a third person', compiled)

    def test_skeleton_only_shifts_the_guides_down_and_forbids_the_target_photo(self):
        compiled = self.module.compile_prompt(
            'Turn the two people in <image1> to the hug in <image3> and <image4>.',
            'skeleton_only')
        self.assertIn('the hug in <image2> and <image3>', compiled)
        self.assertNotIn('<image4>', compiled)
        with self.assertRaises(ValueError):
            self.module.compile_prompt('Copy the pose of <image2>.', 'skeleton_only')

    def test_out_of_range_reference_is_rejected_on_both_arms(self):
        for arm in ('with_b', 'skeleton_only'):
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.module.compile_prompt('Follow <image5>.', arm)

    def test_instruction_names_both_people_and_the_source_ratio_contract(self):
        for arm in ('with_b', 'skeleton_only'):
            instruction = self.module.pe_instruction(arm)
            with self.subTest(arm=arm):
                self.assertIn('<image1>', instruction)
                self.assertIn('exactly two people', instruction)
                self.assertIn('ratio_follow="<image1>"', instruction)
                self.assertIn('image-left/image-right', instruction)

    def test_unknown_arm_and_policy_are_rejected(self):
        with self.assertRaises(ValueError):
            self.module.pe_instruction('both', 'source_only')
        with self.assertRaises(ValueError):
            self.module.pe_instruction('with_b', 'target_objects')
        with self.assertRaises(ValueError):
            self.module.compile_prompt('text', 'nope')

    def test_compiled_prompt_still_satisfies_the_generator_guard(self):
        for arm in ('with_b', 'skeleton_only'):
            compiled = self.module.compile_prompt('Keep <image1> and follow <image3>.', arm)
            with self.subTest(arm=arm):
                self.assertEqual(self.module.validate_generator_prompt(
                    compiled, allowed=self.module.generator_allowed(arm)), compiled)


if __name__ == '__main__':
    unittest.main()
