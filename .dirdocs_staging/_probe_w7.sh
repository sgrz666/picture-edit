cd /home/shangguanrz/project/pic-edit || exit 1
echo "=== does models/dinov2-giant exist? ==="
ls -d models/dinov2-giant 2>&1
echo "=== Hi4DGeometryConverter def ==="
grep -rn 'class Hi4DGeometryConverter' -A 30 src scripts 2>/dev/null | head -60
echo "=== SMPL references in src ==="
grep -rIn --include='*.py' -e 'SMPL_' -e 'smpl' -e 'smplx' src 2>/dev/null | head -30
echo "=== antelopev2 dinov2 nested refs ==="
grep -rIn --include='*.py' --include='*.md' -e 'antelopev2/dinov2' . 2>/dev/null | grep -v gen_folder_docs | head
echo "=== smpl pkl contents ==="
python3 - <<'PY'
import pickle, numpy as np
for p in ('models/smpl/SMPL_NEUTRAL.pkl','models/smpl/smpl/SMPL_NEUTRAL.pkl'):
    with open(p,'rb',encoding='latin1') as f:
        d=pickle.load(f)
    print(p, 'keys=', list(d.keys()))
    print('  vtemplate', np.asarray(d['v_template']).shape, 'faces', np.asarray(d['faces']).shape)
    ps=np.asarray(d['posedirs']); shp=np.asarray(d['shapedirs'])
    print('  posedirs', ps.shape, 'shapedirs', shp.shape)
    print('  kintree', type(d.get('kintree_table')), np.asarray(d['kintree_table']).shape)
    js=np.asarray(d['J']); print('  J', js.shape)
    print('  gender', d.get('gender'), 'dtype', d.get('dtype'))
PY
echo "=== md5 compare two smpl pkls ==="
md5sum models/smpl/SMPL_NEUTRAL.pkl models/smpl/smpl/SMPL_NEUTRAL.pkl
