"""Frozen AdaFace ViT-B KP-RPE recognition for already matched SCRFD faces."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np


def recognition_assets(model_root, rpe_extension):
    root = Path(model_root).resolve()
    code = list((root / 'models').rglob('*.py'))
    if not code or not (root / 'models' / '__init__.py').is_file():
        raise FileNotFoundError('AdaFace-KPRPE requires the local CVLFace models package')
    return [root / 'config.json', root / 'model.safetensors', Path(rpe_extension).resolve(), Path(__file__), *code]


def _load_module(name, file, *, package=False):
    spec = spec_from_file_location(name, file, submodule_search_locations=[str(file.parent)] if package else None)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import required KP-RPE asset: {file}')
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != file:
            raise ImportError(f'KP-RPE module {name} is already loaded from a different path')
        return module
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


class KPRPEIdentityMetric:
    def __init__(self, models_root, model_root, rpe_extension):
        import torch
        from insightface import model_zoo
        from insightface.utils import face_align
        from omegaconf import OmegaConf
        from safetensors.torch import load_file

        self.torch = torch
        self.face_align = face_align
        self.det = model_zoo.get_model(str(Path(models_root) / 'antelopev2' / 'scrfd_10g_bnkps.onnx'),
                                       providers=['CPUExecutionProvider'])
        self.det.prepare(ctx_id=-1, input_size=(640, 640))
        root = Path(model_root).resolve()
        extension = _load_module('rpe_index_cpp', Path(rpe_extension).resolve())
        if extension.version() != '1.2.0':
            raise RuntimeError('Incompatible CVLFace RPE extension')
        package = _load_module('_native_cvlface_models', root / 'models' / '__init__.py', package=True)
        config = OmegaConf.create(json.loads((root / 'config.json').read_text(encoding='utf8'))['conf'])
        if config.start_from or '/vit_kprpe/' not in config.yaml_path or config.output_dim != 512:
            raise ValueError('Expected an AdaFace ViT KP-RPE 512D configuration')
        with torch.random.fork_rng(devices=[]):
            self.model = package.get_model(config)
        state = load_file(root / 'model.safetensors')
        if all(name.startswith('model.') for name in state):
            state = {name.removeprefix('model.'): tensor for name, tensor in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False)

    def encode(self, bgr, landmarks):
        import cv2

        aligned, points = align_kprpe_face(bgr, landmarks, self.face_align.estimate_norm(landmarks))
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.
        inputs = self.torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
        with self.torch.inference_mode():
            return self.model(inputs, self.torch.from_numpy(points)[None]).float().cpu().numpy().reshape(-1)


def align_kprpe_face(bgr, landmarks, matrix):
    import cv2

    crop = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    points = np.asarray(landmarks) @ matrix[:, :2].T + matrix[:, 2]
    return crop, (points / 112.).astype(np.float32)
