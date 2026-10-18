import sys
delivery_root=sys.argv[1]
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
read = lambda path: json.loads(path.read_text())
done = read(root / 'training_complete.json')
assert done['status'] == 'success' and done['production_training'] is True
assert done['completed_steps'] == done['requested_steps'] == 300
assert set(done['variants']) == {'short', 'structured'}
assert done['initialization_sha256_equal'] and done['backbone_unchanged']
assert done['backbone_sha256_before'] == done['backbone_sha256_after']
assert done['validation_selection'] == 'true_val_split_only'
assert done['test_data_read'] is False
for variant in ('short', 'structured'):
    branch = read(root / variant / 'training_complete.json')
    gate = read(root / variant / 'real_model_gates.json')
    assert branch['status'] == 'success' and branch['completed_steps'] == 300
    assert branch['validation_selection'] == 'true_val_split_only'
    assert gate['zero_init_exact'] and gate['gradients_none'] and gate['optimizer_steps'] == 0
    rows = [json.loads(line) for line in (root / variant / 'train_metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == list(range(1, 301))
    vals = [row for row in rows if 'validation_loss' in row]
    assert [row['step'] for row in vals] == list(range(50, 301, 50))
    best = min(vals, key=lambda row: row['validation_loss'])
    assert branch['best_step'] == best['step']
    assert branch['best_validation'] == best['validation_loss']
    checkpoint = root / variant / 'checkpoint_best.pt'
    sha = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    assert sha.hexdigest() == branch['best_checkpoint']['sha256']
    print(variant, 'best step', branch['best_step'], 'validation loss', branch['best_validation'])
print('Both training branches complete; best uses true validation only.')
sys.argv=[sys.argv[0],delivery_root+"/test_pair00_v1"]
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
read = lambda name: json.loads((root / name).read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
done, run, guard = read('evaluation_complete.json'), read('run.json'), read('inference_guard.json')
assert done['status'] == 'success' and done['generated_outputs'] == 12
for key, name in [('run_sha256', 'run.json'), ('inference_guard_sha256', 'inference_guard.json'),
                  ('comparison_board_sha256', 'comparison_board.png')]:
    assert done[key] == sha(root / name)
assert guard['weights_unchanged'] and guard['all_parameter_grads_none'] and guard['all_models_frozen_eval']
assert guard['optimizer_instances_before'] == guard['optimizer_instances_after'] == 0
assert all(count == 0 for count in guard['blocked_attempts'].values())
assert len(guard['generation_samples']) == 12
assert all(not item['grad_enabled'] and item['inference_mode'] for item in guard['generation_samples'])
assert all(not item['grad_enabled'] and item['inference_mode'] for item in guard['native_forward_samples'])
assert guard['target_supervision_opened_during_generation'] is False
assert run['target_supervision']['opened_after_all_12_generations']
assert run['target_supervision']['never_passed_to_generate']
for record in run['output_records']:
    sidecar = root / pathlib.Path(record['path']).name
    assert sha(sidecar) == record['sha256']
    assert sha(sidecar.with_suffix('.png')) == record['png_sha256']
print('12 images verified; no test backpropagation or weight update.')
