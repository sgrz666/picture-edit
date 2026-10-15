import json,pathlib,subprocess
root=pathlib.Path('/data/shangguanrz/pic-edit-experiments/pei2i_body_face_8arm_20261009/code')
files=['face_eval/identity_encoder.ts','face_eval/perceptual_encoder.ts','antelopev2/scrfd_10g_bnkps.onnx','antelopev2/glintr100.onnx','DWPose/yolox_l.onnx','DWPose/dw-ll_ucoco_384.onnx']
print(json.dumps([{'name':f,'path':str((root/'models'/f).resolve()),'exists':(root/'models'/f).is_file(),'size':(root/'models'/f).stat().st_size if (root/'models'/f).is_file() else None} for f in files],indent=2))
print(subprocess.run(['/home/shangguanrz/nihaochen/envs/qwen21/bin/python','-c','import importlib.metadata as m; print({n:m.version(n) for n in ("numpy","opencv-python","onnxruntime","insightface")})'],capture_output=True,text=True).stdout)
