import os

import numpy as np

p = '/home/shangguanrz/project/pic-edit/datasets/Hi4D_pilot_v1/final_assets_native_v1/pair00/hug00/cam4/geometry/000108.npz'
d = np.load(p, allow_pickle=True)
for k in ('semantic_A', 'semantic_B', 'contact_A', 'contact_B',
          'front_owner', 'occlusion_mask', 'interperson_overlap',
          'surface_layer_count', 'A_occluded_by_B', 'B_occluded_by_A'):
    a = d[k]
    u = np.unique(a)
    tail = '...' if len(u) > 20 else ''
    print('  %-22s dtype=%-8s min=%-8s max=%-8s 共%d个唯一值 %s%s'
          % (k, a.dtype, a.min(), a.max(), len(u), u[:20], tail))
print()
print('  normal_scene 范围 :', float(d['normal_scene'].min()), float(d['normal_scene'].max()))
print('  depth_scene 范围  :', float(d['depth_scene'].min()), float(d['depth_scene'].max()))
print('  depth_A 范围      :', float(d['depth_A'].min()), float(d['depth_A'].max()),
      ' 非零占比', float((d['depth_A'] > 0).mean()))
print('  contact_A 范围    :', float(d['contact_A'].min()), float(d['contact_A'].max()),
      ' 非零占比', float((d['contact_A'] > 0).mean()))
print('  asset_version     :', d['asset_version'])
print('  K_runtime         :', d['K_runtime'].tolist())

# Hi4D 官方 semantic 标签（Hi4D 用 15 类 body part）
print()
print('semantic_A 各类像素占比:')
vals, cnts = np.unique(d['semantic_A'], return_counts=True)
tot = d['semantic_A'].size
for v, c in zip(vals, cnts):
    print('    label %-3d  %7.3f%%' % (v, 100.0 * c / tot))
