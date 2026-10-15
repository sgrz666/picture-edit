import numpy as np
from PIL import Image
D = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/overfit_smoke/images'

def load(p):
    return np.asarray(Image.open(p).convert('RGB'), dtype=np.float32)

pairs = [('step_0000_off', 'step_0000_on', '初始化时 OFF vs ON  (应完全一致)'),
         ('step_0000_off', 'step_0002_off', 'OFF vs OFF  (确定性检查)'),
         ('step_0000_on',  'step_0002_on',  'ON vs ON  (训练 2 步后)')]
for a, b, label in pairs:
    x, y = load(f'{D}/{a}.png'), load(f'{D}/{b}.png')
    d = np.abs(x - y)
    nz = int((d > 0).sum())
    print(f'{label}')
    print(f'   max|diff|={d.max():.1f}  均值={d.mean():.4f}  不同像素={nz} / {d.size} ({100*nz/d.size:.3f}%)')
