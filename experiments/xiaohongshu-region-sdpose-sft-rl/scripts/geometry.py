"""Estimate selected RGB frames in an isolated SMPLest-X sandbox."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prepare import read, ref, write, sha


def bind_sdpose_boxes(source, boxes):
    nodes = [node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'yolo_bbox' for t in node.targets)
             and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                     and isinstance(c.func.value, ast.Name) and c.func.value.id == 'detector'
                     and c.func.attr == 'predict' for c in ast.walk(node.value))]
    if len(nodes) != 1: raise ValueError('Exactly one estimator bbox detector assignment required')
    node = nodes[0]; lines = source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [
        ' '*node.col_offset + f'yolo_bbox = np.load({str(boxes)!r}, allow_pickle=False)[frame-1:frame]\n']
    return ''.join(lines)


def reusable_geometry(selection,previous):
    values={}
    for fid,frame in selection['frames'].items():
        old=previous.get('frames',{}).get(fid)
        if old is None:continue
        if old['rgb_sha256']!=frame['rgb_sha256'] or ref(old['parameters']['path'])!=old['parameters']:
            raise ValueError('Reused geometry RGB or parameter SHA differs: '+fid)
        values[fid]=old
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--smplest-root', type=Path, required=True)
    p.add_argument('--reuse-index',type=Path)
    a = p.parse_args(); sys.path.insert(0, str(a.repo))
    import torch
    from src.pose_edit.common import create_estimator_sandbox
    from src.pose_edit.estimator import REQUIRED_PARAMETER_KEYS
    selection = read(a.root/'data/selection.json')
    if (a.root/'data/geometry.json').exists(): raise FileExistsError('Geometry is already locked')
    outputs={} if a.reuse_index is None else reusable_geometry(selection,read(a.reuse_index))
    reuse_count=len(outputs)
    attempt = a.root/('geometry/smplestx_'+uuid.uuid4().hex[:12])
    sandbox = create_estimator_sandbox(a.smplest_root, attempt/'estimator')
    name = 'xhs'; inputs = sandbox/'demo/input_frames'/name; inputs.mkdir()
    frames = sorted((v for k,v in selection['frames'].items() if k not in outputs), key=lambda x: x['frame_id']); staged = []
    if not frames:
        write(a.root/'data/geometry.json',{'schema':'xhs_smplestx_geometry_v1','frames':outputs,
              'reused_index':ref(a.reuse_index),'reused_frames':reuse_count,'training_started':False});return
    for i, frame in enumerate(frames, 1):
        if sha(frame['rgb_path']) != frame['rgb_sha256']: raise ValueError('Selected original RGB changed')
        target = inputs/f'{i:06d}.jpg'
        # OpenCV decodes by signature; PNG inputs keep their exact bytes too.
        shutil.copyfile(frame['rgb_path'], target)
        if sha(target) != frame['rgb_sha256']: raise ValueError('Staged estimator input changed RGB bytes')
        staged.append({'index': i, 'frame_id': frame['frame_id'], 'input': ref(target)})
    import numpy as np
    boxes = np.asarray([read(frame['sdpose']['path'])['bbox_xyxy'] for frame in frames], dtype=np.float32)
    if boxes.shape != (len(frames), 4) or not np.isfinite(boxes).all() or (boxes[:,2:] <= boxes[:,:2]).any():
        raise ValueError('Finite, positive locked SDPose boxes required')
    boxes_path = attempt/'locked-sdpose-boxes.npy'
    with boxes_path.open('xb') as f: np.save(f, boxes, allow_pickle=False)
    inference = sandbox/'main/inference.py'
    inference.write_text(bind_sdpose_boxes(inference.read_text(), boxes_path))
    command = [sys.executable, str(sandbox/'scripts/run_smplestx_params_only.py'), '--num_gpus', '1',
               '--file_name', name, '--ckpt_name', 'smplest_x_h', '--start', '1', '--end', str(len(frames))]
    code = [sandbox/'main/inference.py', sandbox/'scripts/run_smplestx_params_only.py',
            a.smplest_root/'pretrained_models/smplest_x_h/config_base.py']
    models = [a.smplest_root/'pretrained_models/smplest_x_h/smplest_x_h.pth.tar',
              a.smplest_root/'pretrained_models/yolov8x.pt']
    model_refs=[ref(x) for x in models]
    if a.reuse_index is not None:
        old_index=read(a.reuse_index);old_request=old_index['request']
        if ref(old_request['path'])!=old_request:raise ValueError('Reused geometry producer request changed')
        old_models={Path(r['path']).name:r['sha256'] for r in read(old_request['path'])['models']}
        if old_models!={Path(r['path']).name:r['sha256'] for r in model_refs}:
            raise ValueError('Cannot mix reused geometry with a different estimator model')
    write(attempt/'request.json', {'command': command, 'input': staged, 'code': [ref(x) for x in code],
          'models': model_refs, 'selection': ref(a.root/'data/selection.json'),
          'camera_retargeting': False, 'crop_backend': 'locked_SDpose_YOLO11x_bbox', 'boxes': ref(boxes_path),
          'change_scope': 'run-local detector bbox assignment only; SMPLest-X model, crop processing and outputs unchanged'})
    env = dict(os.environ, PYTHONPATH=str(sandbox), PYTHONDONTWRITEBYTECODE='1', SMPLESTX_DET_CONF='.1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',YOLO_CONFIG_DIR=str(a.root/'logs/yolo-settings'))
    for key in ('CUDA_HOME', 'CUDA_PATH', 'CUDACXX', 'LD_LIBRARY_PATH'): env.pop(key, None)
    subprocess.run(command, cwd=sandbox, env=env, check=True)
    result = sandbox/f'demo/output_frames/{name}/{name}_params.pt'
    params = torch.load(result, map_location='cpu', weights_only=False)
    for key in REQUIRED_PARAMETER_KEYS:
        if params[key].shape[0] != len(frames) or not torch.isfinite(params[key]).all():
            raise ValueError('Missing/reordered/invalid estimator frame: '+key)
    directory = attempt/'parameters'; directory.mkdir()
    for index, frame in enumerate(frames):
        value = {key: tensor[index:index+1] for key, tensor in params.items()
                 if torch.is_tensor(tensor) and tensor.shape[0] == len(frames)}
        if tuple(int(value[k].flatten()[0]) for k in ('height','width')) != tuple(frame['original_hw']):
            raise ValueError('Estimated frame order/original dimensions differ')
        path = directory/(frame['frame_id']+'.pt')
        with path.open('xb') as f: torch.save(value, f)
        outputs[frame['frame_id']] = {'parameters': ref(path), 'input': staged[index],
                                    'rgb_sha256': frame['rgb_sha256']}
    write(a.root/'data/geometry.json', {'schema': 'xhs_smplestx_geometry_v1', 'request': ref(attempt/'request.json'),
          'batch_output': ref(result), 'frames': outputs, 'training_started': False,
          'reused_index':None if a.reuse_index is None else ref(a.reuse_index),'reused_frames':reuse_count})
    print(json.dumps({'geometry_frames': len(outputs), 'receipt': str(a.root/'data/geometry.json')}), flush=True)


if __name__ == '__main__': main()
