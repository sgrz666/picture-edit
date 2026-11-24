"""SpatialReward-8B prompts and scalar fusion for Native pose editing."""

from dataclasses import dataclass
import hashlib
import json
import math

from src.data.canvas_contract import CanvasSpec
from ..native_v5.conditioning import ROLES, SOURCE_ROLES, verified


UPSTREAM = '0222484b6a30f865525e8dc3a90acf24654e096a'
SCORE_VERSION = 'spatialreward_native_v1'
SC_PROMPT = (
    'Evaluate the edit in the last image by comparing it with the earlier references and the instruction. '
    'Locate edited areas in the last image before judging them. Return one JSON object with edit_region '
    '(a list of objects with id, label and bbox_2d in 0–1000 image coordinates), brief reasoning, '
    'and score (exactly two numbers from 0 to 25). The first number is instruction following; '
    'the second is preservation of source content that the instruction did not ask to change. '
    'Higher is better. Use <|bbox_{id}|> to cite a localized region and <|global|> for the overall view.'
)
PQ_PROMPT = (
    'Rate the visual quality of this image. Return one JSON object with brief reasoning and score '
    '(exactly two numbers from 0 to 25). First rate naturalness, including human anatomy, perspective and lighting. '
    'Then rate freedom from visual artifacts, such as extra limbs, distortions, seams, watermarks or blur. '
    'Higher means more natural or fewer artifacts, respectively.'
)


@dataclass(frozen=True)
class SpatialTarget:
    roles: tuple
    references: tuple
    instruction: str

    @property
    def prompt_sha256(self):
        return hashlib.sha256(self.instruction.encode('utf8')).hexdigest()

    def record(self):
        return {'roles': self.roles, 'references': self.references, 'prompt_sha256': self.prompt_sha256}


def spatial_target(condition, reference_mode, *, slot_sources=None):
    profile = condition.get('profile')
    images = condition.get('images', ())
    if (profile not in ROLES or tuple(entry.get('role') for entry in images) != ROLES[profile]
            or condition.get('primary_source_index') != 0):
        raise ValueError('SpatialReward requires ordered Native references with Source A at image 0')
    if reference_mode not in ('rgb', 'guides'):
        raise ValueError('Unknown SpatialReward reference mode')
    entries = [entry for entry in images if reference_mode == 'guides' or entry['role'] in SOURCE_ROLES]
    roles = tuple(entry['role'] for entry in entries)
    references = []
    for entry in entries:
        path = verified(entry['file'])
        canvas = CanvasSpec.from_record(entry['canvas'])
        if (canvas.fingerprint() != entry['canvas_id'] or entry['space'] not in ('original', 'canvas')
                or entry['role'] not in SOURCE_ROLES and entry['canvas_id'] != condition['target_canvas_id']):
            raise ValueError('SpatialReward references differ from their bound Canvases')
        references.append({**entry['file'], 'path': str(path), 'crop': canvas.valid_rect if entry['space'] == 'canvas' else None})
    names = {
        'source': 'Source A', 'source_a': 'Source A', 'source_b': 'Source B', 'target_pose': 'target skeleton',
        'combined_pose': 'combined target skeleton', 'person_0_pose': 'target skeleton for slot 0',
        'person_1_pose': 'target skeleton for slot 1',
    }
    order = ', '.join(f'{i + 1} = {names[role]}' for i, role in enumerate(roles))
    count = 'one person' if profile == 'single_2ref' else 'two distinct people'
    guidance = ('Follow the supplied target skeletons.' if reference_mode == 'guides' else
                'Judge the visible change of pose. The exact target skeleton is not provided.')
    if profile == 'dual_source_5ref' and reference_mode == 'guides':
        if slot_sources is None:
            raise ValueError('Dual-source guided SpatialReward requires the Native slot-to-source binding')
        slots = tuple(slot_sources)
        if len(slots) != 2 or any(type(slot) is not int for slot in slots) or set(slots) != {0, 1}:
            raise ValueError('Invalid SpatialReward slot-to-source binding')
        guidance += f' Slot 0 belongs to Source {"AB"[slots[0]]}; slot 1 belongs to Source {"AB"[slots[1]]}.'
    elif profile == 'dual_4ref' and reference_mode == 'guides':
        evidence = (condition.get('person_order') or {}).get('evidence') or {}
        left, slot0 = evidence.get('left_actor'), (condition.get('person_order') or {}).get('slot0_identity')
        if left in ('A', 'B') and slot0 in ('A', 'B'):
            side = 'left' if left == slot0 else 'right'
            guidance += f' Slot 0 is the person on the {side} in Source A.'
    instruction = (f'Reference order: {order}. Image {len(roles) + 1} is the edit. '
                   f'Change the pose of {count}. {guidance} Preserve original identities, clothing, hairstyle and '
                   'body proportions. Preserve the scene and background from Source A. Keep natural anatomy and clean details.')
    return SpatialTarget(roles, tuple(references), instruction)


def _two_scores(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 2 or
            any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 25 for value in values)):
        raise ValueError('SpatialReward requires two finite numeric scores in [0, 25]')
    return list(values)


def parse_scores(text, *, stream='sc'):
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        raise ValueError('SpatialReward returned no JSON object')
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError('SpatialReward returned malformed JSON') from exc
    # SpatialReward-8B also emits explicit named PQ fields. Preserve their
    # semantics and apply the same numeric checks as for an ordered score list.
    if stream == 'pq' and 'score' not in payload:
        return _two_scores([payload.get('naturalness'), payload.get('artifacts')])
    return _two_scores(payload.get('score'))


def spatial_score(semantic, quality, mode):
    follow, consistency = _two_scores(semantic)
    naturalness, artifacts = _two_scores(quality)
    sc = (.6 * follow + .4 * consistency) / 25.
    pq = (.5 * naturalness + .5 * artifacts) / 25.
    if mode not in ('official', 'complementary'):
        raise ValueError('Unknown SpatialReward fusion mode')
    semantic_score = sc if mode == 'official' else consistency / 25.
    return {'instruction_following': follow, 'source_consistency': consistency,
            'naturalness': naturalness, 'artifact_free': artifacts, 'sc': sc, 'pq': pq,
            'reward': semantic_score**.8 * pq**.2}
