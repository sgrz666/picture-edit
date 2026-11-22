"""Persistent SDPose subprocess for Qwen RL image scoring."""

import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


FRAME = struct.Struct('!II')
MAX_PIXELS = 16_777_216


def _write_reply(stream, message):
    stream.write(json.dumps(message, allow_nan=False) + '\n')
    stream.flush()


def _serve(stream, reply, device):
    import cv2
    import numpy as np
    import torch
    from src.evaluation.sdpose import load_scoring_runtime, score_people

    runtime = load_scoring_runtime(device=device)
    versions = {name: getattr(sys.modules.get(name), '__version__', None)
                for name in ('cv2', 'numpy', 'diffusers', 'transformers', 'ultralytics', 'mmcv', 'mmengine', 'mmpose')}
    _write_reply(reply, {'status': 'ready', 'identity': {'python': sys.version,
                 'environment': sys.prefix, 'torch': str(torch.__version__), 'device': str(runtime.device),
                 'versions': versions}})
    while header := stream.read(FRAME.size):
        width, height = FRAME.unpack(header)
        if not width or not height or width * height > MAX_PIXELS:
            raise ValueError('Invalid SDPose image dimensions')
        pixels = stream.read(3 * width * height)
        rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        people, detected = score_people(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), runtime, return_stats=True)
        results = []
        for person in people[:16]:
            points, scores = person['points'], person['scores']
            finite = np.isfinite(points).all(1) & np.isfinite(scores)
            results.append({'box_xyxy': person['box_xyxy'], 'points': np.where(finite[:, None], points, 0.).tolist(),
                            'scores': np.where(finite, scores, 0.).tolist()})
        _write_reply(reply, {'status': 'ok', 'people': results, 'raw_detections': detected})


class SDPoseWorker:
    def __init__(self, python, *, device='cpu'):
        self.python = str(python)
        self.device = device

    def __enter__(self):
        repo = Path(__file__).resolve().parents[4]
        # Use the venv's original path even when its binary is a symlink.
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV')}
        self.process = subprocess.Popen([self.python, '-u', '-m', __name__, '--device', self.device],
                                        cwd=repo, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        try:
            self.identity = self._response('ready')['identity']
        except Exception:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def _response(self, status):
        line = self.process.stdout.readline(4 * 1024 * 1024)
        if not line.endswith(b'\n'):
            raise RuntimeError('SDPose worker ended before completing a reply; inspect worker stderr')
        result = json.loads(line)
        if result.get('status') != status:
            raise RuntimeError(f"SDPose worker failed: {result.get('error', 'unexpected reply')}")
        return result

    def detect(self, image):
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise ValueError('Image exceeds SDPose worker pixel bound')
        try:
            self.process.stdin.write(FRAME.pack(width, height))
            self.process.stdin.write(image.convert('RGB').tobytes())
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError('SDPose worker stopped before receiving the image') from exc
        return self._response('ok')

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
        finally:
            self.process.stdout.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    # Keep stdout protocol-only, including when libraries write directly to fd 1.
    with os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf8') as reply:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            _serve(sys.stdin.buffer, reply, args.device)
        except Exception as exc:
            _write_reply(reply, {'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__':
    main()
