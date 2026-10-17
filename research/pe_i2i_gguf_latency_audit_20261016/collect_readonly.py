"""Inspect existing PE/GGUF evidence; no inference, training or file writes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

# AutoTokenizer below is strictly CPU-side; do not let this audit claim a GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
project = Path('/home/shangguanrz/project/pic-edit')
experiment = Path('/data/shangguanrz/pic-edit-experiments/pe_i2i_gguf_selection_20261014')
runtime = Path('/data/shangguanrz/tools/llama-cpp/b11617-cuda12.8')
model_dir = Path('/data/shangguanrz/pic-edit-models/Qwen-Image-2.1-PE-I2I-GGUF')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, **kwargs):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60, **kwargs)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


config = read(project / 'workflow/config.json')
snapshot = {
    'schema': 'pe_i2i_gguf_latency_readonly_audit_v1',
    'observed_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    'project': str(project), 'experiment': str(experiment),
    'git_log': command(['git', '-C', str(project), 'log', '-5', '--oneline']),
    'configuration': config,
    'production_code': {}, 'experiment_code': {},
    'gguf_files': [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(model_dir.glob('*.gguf'))],
    'declared_weight_checksums': (model_dir / 'SHA256SUMS').read_text(),
    'gpu_model_inference_run_by_this_audit': False,
    'tokenizer': {'device': 'CPU', 'loaded': False},
}
for relative in ['workflow/pe_worker.py', 'workflow/run.py', 'workflow/config.json',
                 'workflow/pose_transfer/prompts.py', 'src/pose_edit/prompt_contract.py']:
    path = project / relative
    text = path.read_text(encoding='utf-8')
    snapshot['production_code'][relative] = {
        'sha256': sha(path), 'bytes': path.stat().st_size,
        'relevant_lines': [{'line': i, 'text': line} for i, line in enumerate(text.splitlines(), 1)
                           if any(key in line for key in ('from_pretrained', 'bfloat16', 'model.generate',
                               'select_generation_prompt', '--prompt-file', 'run_worker(', 'pe_model', 'PE_SAMPLING'))],
    }
for filename in ['pe_llama_batch.py', 'gate0_llama_smoke.py', 'gguf_probe.py', 'fetch_llama_runtime.sh', 'prompt_token_delta.py']:
    path = experiment / 'scripts' / filename
    snapshot['experiment_code'][filename] = {'sha256': sha(path), 'bytes': path.stat().st_size}

tokenizer = None
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['pe_model'], local_files_only=True)
    snapshot['tokenizer'].update(loaded=True, path=config['pe_model'], class_name=type(tokenizer).__name__)
except Exception as exc:
    snapshot['tokenizer']['error'] = f'{type(exc).__name__}: {exc}'

cases = []
for path in sorted((experiment / 'output/gate1/records').glob('*.json')):
    record = read(path)
    hf_dir = project / f'workflow/runs/test/{record["case"]}/prompt'
    hf = read(hf_dir / 'pe_result.json')
    row = {
        'case': record['case'], 'gguf_record': record, 'gguf_record_sha256': sha(path),
        'hf_result_path': str(hf_dir / 'pe_result.json'), 'hf_result_sha256': sha(hf_dir / 'pe_result.json'),
        'hf_seconds': hf['seconds'], 'hf_success_tokens': hf['output_tokens'],
        'hf_attempts': hf['attempts'],
        'hf_all_attempt_tokens': sum(a['output_tokens'] for a in hf['attempts']),
        'gguf_all_attempt_tokens': sum(a['completion_tokens'] for a in record['attempts']),
        'gguf_answer_only_records': [],
    }
    for attempt in record['attempts']:
        raw_path = Path(attempt['raw_path'])
        raw = raw_path.read_text(encoding='utf-8')
        item = {'attempt': attempt['attempt'], 'path': str(raw_path), 'sha256': sha(raw_path),
                'characters': len(raw), 'recorded_completion_tokens': attempt['completion_tokens'],
                'contains_think_end_tag': '</think>' in raw,
                'saved_only_message_content': True}
        if tokenizer is not None:
            item['saved_content_tokens_teacher_tokenizer'] = len(tokenizer.encode(raw, add_special_tokens=False))
            item['unrepresented_completion_tokens_estimate'] = (
                attempt['completion_tokens'] - item['saved_content_tokens_teacher_tokenizer'])
            item['estimate_caveat'] = 'Retokenized content is not an API reasoning-token counter; template/special-token accounting may differ.'
        row['gguf_answer_only_records'].append(item)
    if tokenizer is not None:
        row['hf_final_json_tokens'] = len(tokenizer.encode(json.dumps(hf['official_result'], ensure_ascii=False), add_special_tokens=False))
    cases.append(row)
snapshot['cases'] = cases
gguf_seconds = sum(row['gguf_record']['seconds_total'] for row in cases)
hf_seconds = sum(row['hf_seconds'] for row in cases)
gguf_tokens = sum(row['gguf_all_attempt_tokens'] for row in cases)
hf_tokens = sum(row['hf_all_attempt_tokens'] for row in cases)
snapshot['same_denominator_totals'] = {
    'gguf_request_seconds_excludes_server_loading': gguf_seconds,
    'hf_stage_seconds_includes_loading_and_retries': hf_seconds,
    'gguf_all_attempt_output_tokens': gguf_tokens, 'hf_all_attempt_output_tokens': hf_tokens,
    'gguf_tokens_per_request_second': gguf_tokens / gguf_seconds,
    'hf_tokens_per_stage_second': hf_tokens / hf_seconds,
    'gguf_relative_output_work': gguf_tokens / hf_tokens,
    'gguf_relative_latency': gguf_seconds / hf_seconds,
    'caveat': 'Observed different outputs under different backends/sampling/shared load; not a controlled quantization speedup measurement.',
}

text = (experiment / 'output/gate1/server.log').read_text(errors='replace')
timings = []
for task_id in re.findall(r'\| task (\d+) \| prompt eval time =', text):
    relevant = [line for line in text.splitlines() if f'| task {task_id} |' in line]
    joined = '\n'.join(relevant)
    p = re.search(r'prompt eval time =\s*([\d.]+) ms', joined)
    d = re.search(r'\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens', joined)
    total = re.search(r'total time =\s*([\d.]+) ms', joined)
    acceptance = re.search(r'draft acceptance = ([\d.]+).*mean len =\s*([\d.]+)', joined)
    timings.append({'task_id': int(task_id), 'prompt_processing_ms': float(p[1]),
                    'decode_ms': float(d[1]), 'decode_tokens': int(d[2]),
                    'server_total_ms': float(total[1]),
                    'mtp_acceptance': float(acceptance[1]) if acceptance else None,
                    'mtp_mean_len': float(acceptance[2]) if acceptance else None})
snapshot['server_timings'] = timings
snapshot['decode_fraction_of_server_total'] = sum(x['decode_ms'] for x in timings) / sum(x['server_total_ms'] for x in timings)
snapshot['server_log_header'] = text.splitlines()[:16]
snapshot['gguf_script_cpu_vision_flag'] = '--no-mmproj-offload' in (experiment / 'scripts/pe_llama_batch.py').read_text()
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = ':'.join([str(runtime / 'llama-b11617'),
                                 str(runtime / 'cudart-llama-b11617-bin-ubuntu-cuda-12.8-x64')])
help_result = command([str(runtime / 'llama-b11617/llama-server'), '--help'], env=env)
help_text = help_result['stdout'] + help_result['stderr']
snapshot['installed_runtime_help'] = {
    'returncode': help_result['returncode'],
    'reasoning_budget_available': '--reasoning-budget N' in help_text,
    'reasoning_budget_help_lines': [line for line in help_text.splitlines() if 'reasoning-budget' in line],
    'projector_offload_help_lines': [line for line in help_text.splitlines() if 'mmproj-offload' in line],
    'full_help_sha256': hashlib.sha256(help_text.encode()).hexdigest(),
}
processes = command(['ps', '-eo', 'pid,etime,pcpu,pmem,args'])
snapshot['current_relevant_processes'] = [line for line in processes['stdout'].splitlines()
    if any(key in line for key in ('pe_rewrite_tiktok.py run', 'llama-server -m ', 'pe_llama_batch.py --'))]
snapshot['source_manifests'] = []
for path in sorted(experiment.rglob('*')):
    if path.is_file() and path.suffix in ('.py', '.sh', '.log', '.json', '.txt') and '__pycache__' not in path.parts:
        snapshot['source_manifests'].append({'path': str(path), 'relative_to_experiment': path.relative_to(experiment).as_posix(),
                                             'bytes': path.stat().st_size, 'sha256': sha(path)})
print(json.dumps(snapshot, ensure_ascii=False, indent=2))
