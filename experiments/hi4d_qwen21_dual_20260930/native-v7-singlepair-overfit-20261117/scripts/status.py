"""Read compact progress without loading models or printing per-parameter logs."""
from pathlib import Path
import argparse
import json


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    events = args.root / 'logs' / (args.name + '-events.jsonl')
    event = json.loads(events.read_text(encoding='utf8').splitlines()[-1])
    result = {'event': {key: event.get(key) for key in ('timestamp', 'status', 'stage', 'global_update')},
              'reports': {}, 'segments': []}
    for name in ('completed.json', 'failure.json'):
        path = args.root / 'report' / args.name / name
        if path.is_file():
            result['reports'][name] = read(path)
    configs = [read(path) for path in (args.root / 'configs' / args.name).glob('S*-ready.json')]
    for folder in sorted((args.root / 'output/training/segments').glob('*')):
        resolved = folder / 'resolved.json'
        if not resolved.is_file() or read(resolved)['config'] not in configs:
            continue
        metric, probe, auxiliary_updates, latest_auxiliary = None, None, 0, None
        with (folder / 'metrics.jsonl').open(encoding='utf8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)['record']
                except (json.JSONDecodeError, KeyError):
                    continue  # A concurrent append may be incomplete.
                if row.get('optimizer_updated'):
                    metric = {key: row[key] for key in ('global_update', 'stage_update', 'gradient_norm')}
                    metric['sigmas'] = [entry['sigma'] for entry in row['window']]
                    keys = ('velocity', 'person_fm', 'contact_fm', 'identity', 'identity_contrastive',
                            'eligible', 'contrastive_eligible', 'weighted_micro_objective')
                    metric['losses'] = [{key: parts[key] for key in keys if key in parts}
                                        for parts in row['losses']]
                    if any(parts.get('eligible', 0) for parts in row['losses']):
                        auxiliary_updates += 1
                        latest_auxiliary = metric
                elif row.get('kind') in ('same_scene_observation_loss', 'held_out_loss_probe'):
                    probe = row
        result['segments'].append({'path': str(folder), 'latest_optimizer': metric,
                                   'auxiliary_updates': auxiliary_updates, 'latest_auxiliary': latest_auxiliary,
                                   'loss_probe_present': probe is not None})
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
