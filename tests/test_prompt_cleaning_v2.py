"""Contract tests for scripts/data/clean_prompt_banks_v2.py.

Each rule gets a must-strip case AND a must-keep case, because a cleaner that only has
"must strip" examples gets widened until it eats the instruction itself. One test guards the
regression found while building v2: rewriting a viewer-side identity claim using a person
number taken from a *neighbouring* clause binds person 2 to person 1's skeleton.
"""
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'clean_prompt_banks_v2', ROOT / 'scripts' / 'data' / 'clean_prompt_banks_v2.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

TIK_FIELDS = C.BANKS['tiktok']['fields']
HI_FIELDS = C.BANKS['hi4d']['fields']
SW = C.SKELETON_WORD


def tiktok(text):
    row = {'pair_id': 'p', 'compiled_prompt': text, 'compiled_prompt_inference': text}
    return C.clean_one(row, C.BANKS['tiktok'], set())['compiled_prompt']


def hi4d(text):
    row = {'pair_id': 'p', 'compiled_prompt_training': text, 'compiled_prompt_inference': text}
    return C.clean_one(row, C.BANKS['hi4d'], set())['compiled_prompt_training']


# ---------------------------------------------------------------- R1 watermark
def test_watermark_clause_is_removed_but_the_pose_instruction_survives():
    out = tiktok('On the source image as the canvas, re-pose the person ' + SW + '. '
                 'Keep the framing unchanged, including the upper-left white "TikTok" logo text '
                 'and the white username text "@mikeilajduh".')
    assert 'TikTok' not in out and '@mikeilajduh' not in out and 'logo' not in out.lower()
    assert 're-pose the person' in out


def test_a_sentence_that_is_only_a_watermark_order_is_dropped():
    out = tiktok('Render the watermark "@user" exactly as in the source image. '
                 'Raise both arms to shoulder height as described by ' + SW + '.')
    assert '@user' not in out and 'watermark' not in out.lower()
    assert 'Raise both arms' in out


def test_held_object_with_text_on_it_is_not_a_watermark(  # must NOT over-strip
):
    out = tiktok('The person holds a book whose cover title is visible. ' + SW + ' sets the arm angles.')
    assert 'holds a book' in out


# ---------------------------------------------------------------- R2 appearance
def test_colour_enumeration_becomes_one_generic_preservation_clause():
    out = tiktok('Keep the woman\u2019s voluminous curly black afro hairstyle, septum nose ring, '
                 'orange long-sleeved bodysuit, gray drawstring sweatpants and the gray wall '
                 'completely unchanged. Bend the left elbow to waist height per ' + SW + '.')
    assert 'afro' not in out and 'sweatpants' not in out
    assert 'orange' not in out and 'gray' not in out
    assert 'Keep the person exactly as the source image shows them' in out
    assert 'Bend the left elbow' in out                      # pose content untouched
    assert out.count('Keep the person exactly as the source image shows them') == 1


def test_uncoloured_pose_wording_is_kept():
    out = tiktok('The sleeves stretch along the raised arms while ' + SW + ' sets the shoulder line.')
    assert 'sleeves stretch along the raised arms' in out


def test_cloth_dynamics_instruction_keeps_its_noun_after_decoding_the_colour():
    out = tiktok('The orange sleeves must stretch and fold naturally along the raised arms. '
                 + SW + ' defines the joint angles.')
    assert 'must stretch and fold naturally along the raised arms' in out
    assert 'orange' not in out


# ---------------------------------------------------------------- R3 GT references
def test_fifth_image_prohibition_and_positive_use_both_go():
    out = hi4d('Re-pose the two men to the combined dual-person skeleton. Do not copy clothing '
               'from the fifth image. Use the fifth image only as an interaction reference for '
               'torso contact. Person 1\u2019s isolated skeleton drives the arms.')
    assert 'fifth image' not in out.lower()
    assert 'combined dual-person skeleton' in out


def test_target_photograph_provenance_is_cut_and_the_claim_stays():
    out = tiktok('Transfer the facial expression and hand configuration from the unseen target '
                 'photograph and encoded in ' + SW + '.')
    assert 'target photograph' not in out and 'unseen' not in out
    assert 'facial expression' in out                          # content kept, provenance gone


def test_ground_truth_wording_never_survives():
    out = hi4d('Match the interaction read from the ground truth image, '
               'with the combined dual-person skeleton setting positions.')
    assert 'ground truth' not in out.lower()


# ---------------------------------------------------------------- R4 collapse
def test_skeleton_lead_in_collision_collapses_to_one_reference():
    out = tiktok('Transfer the pose encoded by the joint skeleton in ' + SW + ' to the person.')
    assert 'joint skeleton in ' + SW not in out
    assert out.count('joint arrangement') >= 1 and SW in out


def test_hi4d_role_word_self_collision_collapses():
    out = hi4d('Re-pose the two men to exactly match the combined dual-person skeleton in the '
               'combined dual-person skeleton, with person 1\u2019s isolated skeleton as guide.')
    assert 'combined dual-person skeleton in the combined dual-person skeleton' not in out


# ---------------------------------------------------------------- R5 side anchor
def test_person_number_and_side_are_decoupled_consistently():
    out = hi4d('The left man (person 1, reddish-brown hair, beard, dark gray T-shirt) must adopt '
               "the pose from person 1's isolated skeleton: torso upright. The right man "
               '(person 2, dark hair, navy T-shirt) must adopt the pose from person 2\u2019s '
               'isolated skeleton: torso leaning.')
    assert 'left man' not in out and 'right man' not in out
    assert 'isolated skeleton is person 1 must adopt the pose from person 1' in out
    assert 'isolated skeleton is person 2 must adopt the pose from person 2' in out
    assert 'T-shirt' not in out and 'hair' not in out


def test_rewrite_never_binds_a_person_to_another_person_s_skeleton():
    """Regression: the number must come from the attached parenthetical, not a neighbouring clause."""
    src = ('The left figure, wearing the navy shorts, must adopt the pose from person 2\u2019s '
           "isolated skeleton, while person 1's isolated skeleton drives the right figure.")
    out = hi4d(src)
    for m in re.finditer(r'isolated skeleton is person ([12])[^.]{0,120}?pose (?:from|of) '
                         r'person ([12])\u2019?\'?s isolated skeleton', out):
        assert m.group(1) == m.group(2), out
    assert 'person 2must' not in out and not re.search(r'person \d[a-z]', out)


# ---------------------------------------------------------------- loader invariants
def test_training_form_stays_loadable_for_both_banks():
    tik = tiktok('On the source image as the canvas, transfer only the pose encoded by the joint '
                 'skeleton in ' + SW + ', keeping the black jacket and gray trousers unchanged.')
    assert '<image' not in tik and SW in tik
    hi = hi4d('On the source image (canvas), re-pose the two men using the combined dual-person '
              "skeleton and person 1's isolated skeleton, keeping the red shirts unchanged.")
    assert '<image' not in hi
    assert "combined dual-person skeleton" in hi


def test_inference_form_keeps_only_downstream_image_numbers():
    row = {'pair_id': 'p', 'compiled_prompt': 'x ' + SW,
           'compiled_prompt_inference': 'Edit the single person in <image1> into the pose; '
                                        '<image2> is the skeleton and the gray wall stays put.'}
    out = C.clean_one(row, C.BANKS['tiktok'], set())['compiled_prompt_inference']
    assert '<image1>' in out and '<image2>' in out
    assert '<image3>' not in out and '<image5>' not in out


# ---------------------------------------------------------------- R6 repair
def test_on_image_text_claim_goes_whole_and_the_pose_survives():
    out = tiktok('Raise both arms to shoulder height per ' + SW + '. '
                 'Keep the banner with text "SALE TODAY" exactly as it is, positioned top left.')
    assert 'SALE' not in out and 'banner' not in out.lower() and '"' not in out
    assert 'Raise both arms to shoulder height' in out


def test_transcribed_garment_lettering_leaves_no_dangling_comma():
    out = tiktok('Keep the sweatpants carrying the text STOP LOO.. and the sneakers unchanged. '
                 'Bend the right knee per ' + SW + '.')
    assert 'STOP' not in out and ', ,' not in out and ',.' not in out and ' ,' not in out
    assert 'Bend the right knee' in out


def test_pe_reasoning_voice_is_dropped_not_reworded():
    out = hi4d('Re-pose the two men using the combined dual-person skeleton. '
               'No wait. Let me carefully re-read the actual skeleton poses. '
               'The left hip is forward and the right arm is extended.')
    assert 'no wait' not in out.lower() and 'let me' not in out.lower()
    assert 'left hip is forward' in out


def test_no_orphan_conjunction_dangling_prep_or_double_period():
    out = hi4d('Transfer only the pose and interaction from the skeletons and; never copy '
               'clothing from the combined dual-person skeleton. So the target interaction is '
               'described in words. Person 1\'s isolated skeleton sets the arm angles.... wait')
    assert ' and;' not in out and not re.search(r'\b(?:and|or|to|of|in)\s*[.,;]', out, re.I)
    assert '...' not in out and not re.search(r'(?:^|[.!?]\s+)(?:And|So|Or|With|and|so|or|with)\s', out)


def test_that_person_anaphora_is_not_left_orphaned():
    out = hi4d('Re-pose the two men using person 1\'s isolated skeleton as the precise joint '
               'guide for the left person and person 2\'s isolated skeleton as the exact joint '
               'guide for the right person, with the combined dual-person skeleton for spacing.')
    assert 'for that person' not in out
    assert 'as the precise joint guide ' in out or 'as the exact joint guide ' in out
    assert 'left person' not in out and 'right person' not in out
    assert "person 1's isolated skeleton" in out and "person 2's isolated skeleton" in out


def test_skeleton_contract_wording_is_not_mistaken_for_a_text_claim():  # must-keep
    out = tiktok('Do not render the skeleton lines or their colors in the output, per ' + SW + '.')
    assert 'skeleton lines or their colors' in out


def test_action_label_never_survives_in_the_instruction():
    """Found on a printed sheet: "...to understand the true interaction: talk12, torso contact" """
    out = hi4d('Re-pose the two men using the combined dual-person skeleton to resolve the '
               'interaction semantics (a face-to-face reaching confrontation, fight01) as the '
               'true guide for torso contact and hand placement.')
    assert 'fight01' not in out and '(,)' not in out and '  ' not in out
    assert 'face-to-face reaching confrontation' in out          # the useful words stay


def test_shade_words_count_as_appearance_claims():
    """The sheet for fight01_cam52_000015_000002 kept "the man in the dark T-shirt"."""
    out = hi4d('The left person (the man in the dark T-shirt and navy shorts) must adopt the pose '
               "from person 1's isolated skeleton: torso upright, arms lowered.")
    assert 'dark' not in out and 'navy' not in out and 'T-shirt' not in out
    assert 'torso upright' in out and 'arms lowered' in out


