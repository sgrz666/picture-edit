"""Persistent, local SpatialReward-8B worker with a JSON-lines protocol."""

import argparse
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread

from PIL import Image

from ..native_v5.conditioning import verified
from ..native_v62.training.io import file_sha
from .spatial_reward import PQ_PROMPT, SC_PROMPT, SCORE_VERSION, UPSTREAM, parse_scores, spatial_score


def ordered_results(rows, count, mode):
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError('SpatialReward returned an incomplete candidate group')
    by_id = {}
    for row in rows:
        candidate = row.get('id') if isinstance(row, dict) else None
        if type(candidate) is not int or not 0 <= candidate < count or candidate in by_id:
            raise ValueError('SpatialReward returned an invalid or duplicate candidate ID')
        by_id[candidate] = {'id': candidate, **spatial_score(row.get('sc'), row.get('pq'), mode)}
    return [by_id[i] for i in range(count)]


def model_identity(model_path):
    root = Path(model_path).resolve()
    index = json.loads((root / 'model.safetensors.index.json').read_text(encoding='utf8'))
    shards = set(index['weight_map'].values())
    if not shards or any(Path(name).name != name or not name.endswith('.safetensors') for name in shards):
        raise ValueError('SpatialReward requires a complete local safetensors index')
    required = {'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'preprocessor_config.json'} | shards
    assets = {p.name for p in root.iterdir() if p.is_file() and p.suffix in ('.json', '.jinja', '.txt')}
    files = {name: file_sha(root / name) for name in sorted(required | assets)}
    return {'model_path': str(root), 'files': files, 'upstream_code': UPSTREAM, 'score_version': SCORE_VERSION,
            'template_sha256': sha256((SC_PROMPT + '\n' + PQ_PROMPT).encode('utf8')).hexdigest()}


class SpatialWorker:
    def __init__(self, model, *, python=None, device='cpu', mode='official', references='rgb',
                 batch_size=2, max_pixels=262144, max_new_tokens=512, timeout=900):
        if min(batch_size, max_pixels, max_new_tokens, timeout) < 1:
            raise ValueError('SpatialReward worker limits must be positive')
        self.model = Path(model).resolve()
        self.python = str(Path(python or sys.executable).absolute())
        self.device, self.mode, self.references = device, mode, references
        self.batch_size, self.max_pixels, self.max_new_tokens, self.timeout = batch_size, max_pixels, max_new_tokens, timeout

    def __enter__(self):
        repo = Path(__file__).resolve().parents[4]
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        self.process = subprocess.Popen([self.python, '-u', '-m', __name__, '--model', str(self.model),
                                         '--device', self.device, '--batch-size', str(self.batch_size),
                                         '--max-pixels', str(self.max_pixels), '--max-new-tokens', str(self.max_new_tokens)],
                                        cwd=repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.replies = Queue()
        self.reader = Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.identity = {**self._response('ready')['identity'], 'mode': self.mode, 'references': self.references}
        except Exception:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _read(self):
        for line in self.process.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def _response(self, status):
        try:
            line = self.replies.get(timeout=self.timeout)
        except Empty as exc:
            self.process.kill()
            raise RuntimeError(f'SpatialReward worker timed out waiting for {status}') from exc
        if line is None:
            raise RuntimeError('SpatialReward worker exited before replying; inspect worker stderr')
        result = json.loads(line)
        if result.get('status') != status:
            raise RuntimeError(f"SpatialReward worker failed: {result.get('error', 'unexpected reply')}")
        return result

    def score_group(self, target, candidates):
        files = [{'id': i, 'path': str(Path(path).resolve()), 'sha256': file_sha(path)} for i, path in enumerate(candidates)]
        if not files:
            raise ValueError('SpatialReward requires a nonempty candidate group')
        request = {'instruction': target.instruction, 'references': target.references, 'candidates': files}
        try:
            self.process.stdin.write((json.dumps(request, allow_nan=False) + '\n').encode('utf8'))
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError('SpatialReward worker stopped before receiving the candidate group') from exc
        return ordered_results(self._response('ok')['results'], len(files), self.mode)

    def __exit__(self, *_):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.reader.join(timeout=2)
        self.process.stdout.close()


def _image(ref):
    from .rewards import rendered_rgb

    with Image.open(verified(ref)) as image:
        if ref.get('crop'):
            image = image.crop(ref['crop'])
        return rendered_rgb(image)


class _Model:
    def __init__(self, args):
        import torch
        import transformers
        from qwen_vl_utils import process_vision_info

        self.identity = model_identity(args.model)
        self.device = torch.device(args.device)
        self.processor = transformers.AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        self.processor.tokenizer.padding_side = 'left'
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={'': str(self.device)}, local_files_only=True,
            use_safetensors=True, trust_remote_code=False).eval().requires_grad_(False)
        self.vision_info = process_vision_info
        self.args, self.torch = args, torch
        self.identity.update(python=sys.version, executable=sys.executable, environment=sys.prefix, torch=str(torch.__version__),
                             transformers=transformers.__version__, device=str(self.device), dtype='bfloat16',
                             qwen_vl_utils=metadata.version('qwen-vl-utils'),
                             batch_size=args.batch_size, max_pixels=args.max_pixels, max_new_tokens=args.max_new_tokens)

    def _predict(self, images, prompt, *, stream):
        messages = [[{'role': 'user', 'content': [
            *({'type': 'image', 'image': image, 'max_pixels': self.args.max_pixels} for image in group),
            {'type': 'text', 'text': prompt}]}] for group in images]
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        inputs, _ = self.vision_info(messages)
        batch = self.processor(text=texts, images=inputs, padding=True, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            generated = self.model.generate(**batch, do_sample=False, max_new_tokens=self.args.max_new_tokens)
        text = self.processor.batch_decode(generated[:, batch.input_ids.shape[1]:], skip_special_tokens=True)
        if len(text) != len(images):
            raise ValueError('SpatialReward model returned an incomplete batch')
        return [parse_scores(out, stream=stream) for out in text]

    def score(self, request):
        refs = [_image(entry) for entry in request['references']]
        items = request['candidates']
        results = []
        for begin in range(0, len(items), self.args.batch_size):
            chunk = items[begin:begin + self.args.batch_size]
            edited = [_image(entry) for entry in chunk]
            sc = self._predict([refs + [image] for image in edited], SC_PROMPT + '\n' + request['instruction'], stream='sc')
            pq = self._predict([[image] for image in edited], PQ_PROMPT, stream='pq')
            results.extend({'id': row['id'], 'sc': a, 'pq': b} for row, a, b in zip(chunk, sc, pq, strict=True))
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-pixels', type=int, default=262144)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    with os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf8') as reply:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        def send(record):
            print(json.dumps(record, allow_nan=False), file=reply, flush=True)

        try:
            model = _Model(args)
            send({'status': 'ready', 'identity': model.identity})
            for line in sys.stdin.buffer:
                send({'status': 'ok', 'results': model.score(json.loads(line))})
        except Exception as exc:
            send({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__':
    main()
