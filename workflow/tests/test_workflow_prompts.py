import importlib
import unittest


class PromptContractTests(unittest.TestCase):
    def setUp(self):
        try:
            self.prompts = importlib.import_module('pose_transfer.prompts')
        except ModuleNotFoundError:
            self.fail('workflow prompt compiler is missing')

    def test_parse_official_answer_after_thinking(self):
        result = self.prompts.parse_pe_answer('<think>ignore this {not JSON}</think>\n'
            '{"rewritten_prompt":"Raise the forearm.","wh_ratio":"","ratio_follow":"<image1>"}')
        self.assertEqual(result['rewritten_prompt'], 'Raise the forearm.')

    def test_three_to_two_image_roles_do_not_swap_target_and_skeleton(self):
        result = self.prompts.compile_prompt(
            'Keep the man in <image1>. Raise the image-right forearm with the palm upward; '
            'follow the joint arrangement in <image3>.', 'source_only')
        self.assertNotIn('<image3>', result)
        self.assertIn('joint arrangement in <image2>', result)
        self.assertIn('palm upward', result)
        self.assertIn('Do not add the basketball', result)

    def test_reference_to_absent_target_photo_is_rejected(self):
        for text in ('Copy the exact pose and facial expression from <image2>.',
                     'Raise the arm as in image 2 and follow <image3>.'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.prompts.compile_prompt(text, 'source_only')

    def test_source_objects_are_preserved_without_unconditional_empty_hand(self):
        result = self.prompts.compile_prompt('Retain the source cup in the raised hand.')
        self.assertIn('Preserve objects already in the source', result)
        self.assertNotIn('must remain empty', result)

    def test_unknown_image_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            self.prompts.compile_prompt('Use <image4>.', 'source_only')

    def test_ratio_cannot_follow_absent_target_photo(self):
        with self.assertRaises(ValueError):
            self.prompts.parse_pe_answer('{"rewritten_prompt":"Raise arm",'
                '"wh_ratio":"","ratio_follow":"<image2>"}')

    def test_instruction_assigns_target_appearance_and_objects_explicitly(self):
        instruction = self.prompts.pe_instruction('source_only')
        self.assertIn('<image3>', instruction)
        self.assertIn('basketball', instruction)
        self.assertIn('identity', instruction)


class GeneratorPromptGuardTests(unittest.TestCase):
    """The two-image contract also has to hold for a hand-supplied --prompt-file."""

    def setUp(self):
        self.prompts = importlib.import_module('pose_transfer.prompts')

    def test_source_and_guide_references_are_accepted(self):
        text = 'Lift the image-right arm of the person in <image1> to the joints in <image2>.'
        self.assertEqual(self.prompts.validate_generator_prompt(text), text)

    def test_reference_to_a_third_image_is_rejected_before_generation(self):
        for text in ('Copy the pose from <image3>.',
                     'Read the target photograph as image 3 and follow <image2>.'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.prompts.validate_generator_prompt(text)

    def test_prompt_naming_no_image_is_rejected(self):
        with self.assertRaises(ValueError):
            self.prompts.validate_generator_prompt('Raise both arms and turn the head.')

    def test_compiled_prompt_still_satisfies_the_guard(self):
        compiled = self.prompts.compile_prompt('Keep <image1> and follow the guide in <image3>.')
        self.assertEqual(self.prompts.validate_generator_prompt(compiled), compiled)


if __name__ == '__main__':
    unittest.main()
