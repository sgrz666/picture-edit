import os
import sys

print('Python executable:', sys.executable)
print('CUDA_HOME:', os.environ.get('CUDA_HOME'))

import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Device:', torch.cuda.get_device_name(0))

import deepspeed
print('DeepSpeed version:', deepspeed.__version__)

import transformers
print('Transformers version:', transformers.__version__)

import diffusers
print('Diffusers version:', diffusers.__version__)

import xtuner
print('XTuner version:', xtuner.__version__)

import qwen_vl_utils
print('Qwen-VL-Utils loaded successfully.')

print('\n*** ALL DEEPGEN CORE DEPENDENCIES ARE VERIFIED AND WORKING! ***')
