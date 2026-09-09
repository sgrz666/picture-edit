import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import threading
import time

PROJECT = Path(os.environ.get('DEEPGEN_PROJECT', '/home/shangguanrz/project/pic-edit'))
MODEL = PROJECT / 'models/DeepGen-1.0-diffusers'
REVISION = '85c2ed257b9406d8531965c68e140dc7fdb81382'
NEGATIVE = ('blurry, low quality, low resolution, distorted, deformed, broken content, '
            'missing parts, damaged details, artifacts, glitch, noise, extra fingers, '
            'missing fingers, mutated hands, bad composition, wrong proportion, unfinished.')
TASKS = {
 'S0': 'Keep this photograph unchanged, including the same person, pose, sunglasses, hat, clothing, ski equipment and background.',
 'S1': 'Change only the red panels of the skier\'s jacket to blue. Keep the person, face, sunglasses, hat, pose, all other clothing, ski equipment and background unchanged.',
 'S2': 'Raise one hand of the skier to wave. Keep the same person and facial appearance, sunglasses, striped hat, jacket, trousers, boots and background. Do not add another person.',
 'D0': 'Keep this photograph unchanged. Preserve every existing person, their faces, clothing, poses and all background objects.',
 'D1': 'The man sitting on the left in a light-colored graphic T-shirt stands up and shakes hands with the existing standing man in a dark plaid short-sleeved shirt. These are the two people who interact. Keep both men\'s original faces and respective clothing. Keep all other people and the room unchanged. Do not add, duplicate, replace or remove anyone.',
 'D2': 'The man sitting on the left in a light-colored graphic T-shirt stands up and puts one arm around the shoulder of the existing standing man in a dark plaid short-sleeved shirt. Keep both men\'s original faces and respective clothing. Keep all other people and the room unchanged. Do not add, duplicate, replace or remove anyone.',
}
OLD = {
 'D1': 'Modify the two people to shake hands warmly in friendly greeting, body postures facing each other slightly, natural hands and arms without distortion, maintaining identical identities, clothing and background.',
 'D2': 'Change the two people\'s interaction so that the person on the left puts an arm around the right person\'s shoulder in a warm friendly embrace, natural contact, preserving identity and clothing.',
}

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def source_path(task):
    return PROJECT / 'inputs' / ('person1.jpg' if task.startswith('S') else 'dual_person.jpg')

def preprocess(image, resolution, mode):
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(image).convert('RGB')
    if mode == 'stretch':
        return image.resize((resolution, resolution), Image.Resampling.LANCZOS), [0, 0, resolution, resolution]
    if mode != 'letterbox':
        raise ValueError(mode)
    scale = resolution / max(image.size)
    w, h = [max(1, round(x * scale)) for x in image.size]
    x, y = (resolution - w) // 2, (resolution - h) // 2
    canvas = Image.new('RGB', (resolution, resolution), (128, 128, 128))
    canvas.paste(image.resize((w, h), Image.Resampling.LANCZOS), (x, y))
    return canvas, [x, y, x + w, y + h]

def quality_manifest():
    rows = []
    def add(group, task, seed, preprocess='letterbox', cfg=4.5, steps=35, old=False, negative=NEGATIVE):
        row = dict(kind='quality', group=group, task=task, seed=seed, resolution=512,
                   preprocess=preprocess, cfg=cfg, steps=steps, negative_prompt=negative,
                   prompt=OLD[task] if old else TASKS[task], control_injected=False)
        row['id'] = f'{group}_{task}_{seed}_{preprocess}_{steps}_{cfg}'
        rows.append(row)
    for task in TASKS:
        for seed in (42, 1024):
            for prep in ('stretch', 'letterbox'):
                add('Q0', task, seed, prep)
    for task in ('D1', 'D2'):
        for seed in (42, 1024):
            add('Q1', task, seed, old=True)
    for task in ('S2', 'D1'):
        for cfg in (2.5, 4.5, 6.0):
            for steps in (35, 50):
                if (cfg, steps) == (4.5, 35):
                    continue
                for seed in (42, 1024):
                    add('Q2', task, seed, cfg=cfg, steps=steps)
        for seed in (42, 1024):
            add('Q3', task, seed, negative='')
    return rows

def training_manifest(sensitivity=False):
    import itertools
    rows = []
    dims = itertools.product((512,768,1024), ('cache',), (True,), ('S0','D0'), ('S','L')) if sensitivity else itertools.product((512,768,1024), ('cache','online'), (True,False), ('S0','D0'), ('M',))
    for res, enc, ckpt, task, size in dims:
        rows.append(dict(id=f'T_{size}_{res}_{enc}_{int(ckpt)}_{task}', kind='train',group='sensitivity' if sensitivity else 'main',
                         resolution=res, encoding=enc, checkpointing=ckpt, task=task, adapter_size=size,
                         seed=42, preprocess='letterbox',prompt=TASKS[task],warmup=5,updates=20,control_injected=True))
    return rows

class Monitor:
    """One independent CUDA process; sampled NVML and stage-scoped allocator peaks."""
    def __init__(self, out):
        import torch
        import pynvml
        self.torch, self.nv = torch, pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.out = Path(out)
        self.start = time.perf_counter()
        self.stage = 'startup'
        self.stop_event = threading.Event()
        self.peak_nvml = 0
        self.sample_count = 0
        self.stages = []
        self.sample_error = None
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self):
        try:
            with (self.out / 'gpu_samples.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['elapsed_seconds','used_gib','stage'])
                while not self.stop_event.is_set():
                    used = self.nv.nvmlDeviceGetMemoryInfo(self.handle).used / 2**30
                    self.peak_nvml = max(self.peak_nvml, used)
                    self.sample_count += 1
                    writer.writerow([time.perf_counter()-self.start,used,self.stage])
                    f.flush()
                    self.stop_event.wait(.05)
        except Exception as e:
            self.sample_error = repr(e)

    def event(self, **record):
        with (self.out / 'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(elapsed_seconds=time.perf_counter()-self.start,**record),allow_nan=False)+'\n')

    @contextlib.contextmanager
    def section(self, name):
        t = self.torch
        t.cuda.synchronize()
        t.cuda.reset_peak_memory_stats()
        self.stage = name
        start = time.perf_counter()
        self.event(stage=name,event='start',allocated_gib=t.cuda.memory_allocated()/2**30,reserved_gib=t.cuda.memory_reserved()/2**30)
        try:
            yield
        finally:
            t.cuda.synchronize()
            rec = dict(stage=name,event='end',seconds=time.perf_counter()-start,
                       allocated_gib=t.cuda.memory_allocated()/2**30,reserved_gib=t.cuda.memory_reserved()/2**30,
                       peak_allocated_gib=t.cuda.max_memory_allocated()/2**30,peak_reserved_gib=t.cuda.max_memory_reserved()/2**30)
            self.stages.append(rec)
            self.event(**rec)
            self.stage = 'between_stages'

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        return dict(wall_seconds=time.perf_counter()-self.start,peak_nvml_gib=self.peak_nvml,
                    peak_allocated_gib=max((s['peak_allocated_gib'] for s in self.stages),default=None),
                    peak_reserved_gib=max((s['peak_reserved_gib'] for s in self.stages),default=None),
                    stages=self.stages,nvml_sample_interval_ms=50,nvml_error=self.sample_error,
                    nvml_sample_count=self.sample_count)

def environment():
    import torch
    import importlib.metadata as md
    pkgs = {}
    for name in ('torch','diffusers','transformers','accelerate','safetensors','numpy','nvidia-ml-py'):
        pkgs[name] = md.version(name)
    return dict(python=platform.python_version(),platform=platform.platform(),packages=pkgs,
                cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                torch_total_gib=torch.cuda.get_device_properties(0).total_memory/2**30,model_revision=REVISION)
