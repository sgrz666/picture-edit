import numpy as np
from PIL import Image

D = '/home/shangguanrz/project/pic-edit/experiments/qwen-2.1-test/experiment/results/overfit_inter01/images'
a = np.asarray(Image.open(f'{D}/step_0000_on.png').convert('RGB'), np.float32)
b = np.asarray(Image.open(f'{D}/step_0500_on.png').convert('RGB'), np.float32)
d = np.abs(a - b)
print('图像尺寸', a.shape)
print('全图 max|d| =', float(d.max()), ' 非零像素 =', int((d > 0).sum()), f'({100*(d>0).mean():.2f}%)')
print('非零像素均值 =', float(d[d > 0].mean()) if (d > 0).any() else 0)
if (d > 0).any():
    ys, xs = np.where(d.max(axis=2) > 0)
    print('差异外接框: x[%d,%d] y[%d,%d]' % (xs.min(), xs.max(), ys.min(), ys.max()))
    # 按行/列分布
    rows = np.where((d.max(axis=2) > 0).any(axis=1))[0]
    cols = np.where((d.max(axis=2) > 0).any(axis=0))[0]
    print('有差异的行数', len(rows), ' 列数', len(cols))
# 再看 off vs on（同 step）
c = np.asarray(Image.open(f'{D}/step_0500_off.png').convert('RGB'), np.float32)
d2 = np.abs(c - b)
print()
print('OFF(500) vs ON(500): max|d| =', float(d2.max()), ' 非零 =', int((d2 > 0).sum()), f'({100*(d2>0).mean():.2f}%)')
ys, xs = np.where(d2.max(axis=2) > 0)
if len(ys):
    print('  差异外接框: x[%d,%d] y[%d,%d]' % (xs.min(), xs.max(), ys.min(), ys.max()))
