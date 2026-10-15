"""Validate the saved diagnosis evidence without SSH, GPU, or model loading."""
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / 'evidence' / name).read_text(encoding='utf-8-sig'))


audit = load('audit_evidence.json')
for remote, digest in audit['source_sha256'].items():
    snapshot = ROOT / 'remote_source' / Path(remote).name
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == digest, remote

parity = load('fresh_parity_512.json')
manifest = load('fresh_parity_manifest.json')
assert parity['passed'] and len(parity['cases']) == 2
assert manifest['optimizer_steps'] == 0 and manifest['inference_steps'] == 0
for case in parity['cases']:
    assert case['passed'] and case['resolution'] == 512 and case['sigma'] == .5
    assert case['all_models_frozen'] and case['parameter_gradients_absent']
    for comparison in case['comparisons'].values():
        assert comparison['passed'] and comparison['exactly_equal']
        assert comparison['max_abs'] == 0

for methods in audit['native_value_metrics'].values():
    for value in methods.values():
        assert value['n_images'] == 6 and value['pose_pck_002']['n'] == 6
for value in audit['pilot_scores'].values():
    assert value['n'] == 24 and value['config']['geometry_strength'] == 1
    assert value['config']['face_strength'] == 1
assert audit['pilot_training']['J']['aux_eligible_updates'] == 20

counts = load('checkpoint_parameter_counts.json')
assert sum(counts['groups'].values()) == counts['trainable_parameter_elements']
log = (ROOT / 'evidence/targeted_tests.log').read_text(encoding='utf-8-sig')
assert '26 passed' in log and 'FAILED ' not in log

for image in (ROOT / 'figures').glob('*.png'):
    data = image.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', image
    width, height = struct.unpack('>II', data[16:24])
    assert min(width, height) > 0 and data[-8:-4] == b'IEND', image
assert len(list((ROOT / 'figures').glob('*.png'))) == 2
print('PASS: source hashes, fresh two-case parity, metric cohorts, parameter counts, '
      '26-test log, and two PNG figures verified.')
