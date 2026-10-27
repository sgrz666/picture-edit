"""Public A/B/C roles and immutable five-reference protocol."""
from pathlib import Path

ROLES=('source_a','source_b','person_0_pose','person_1_pose','combined_pose')
PROFILE='dual_source_5ref'

def resolve_source_cache(run,inputs,settings):
    """Read only sealed source artifacts; never copy a run's worker requests."""
    import hashlib,json
    from PIL import Image,ImageOps
    from abc_runner import sha256
    geometry=Path(run).resolve()/'geometry'
    stage=json.loads((geometry/'stage.json').read_text(encoding='utf8'))
    scene_path=geometry/'scene.json'
    if sha256(scene_path)!=stage['outputs']['scene.json']['sha256']:raise ValueError('Cached scene SHA differs')
    scene=json.loads(scene_path.read_text(encoding='utf8'))
    old=json.loads((geometry/'request.json').read_text(encoding='utf8'))
    if old['settings']['resolution']!=settings['resolution']:raise ValueError('Source cache resolution differs')
    def pixel_sha(path):
        with Image.open(path) as im:rgb=ImageOps.exif_transpose(im).convert('RGB')
        return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    records=[];bindings=[]
    for role in ('A','B'):
        input_sha=pixel_sha(inputs['source_'+role.lower()])
        candidates=[s for s in scene['sources'] if pixel_sha(s['image'])==input_sha]
        if len(candidates)!=1:raise ValueError('Source cache must uniquely match the actual A/B image pixels')
        source=dict(candidates[0]);artifacts={};semantic={}
        for key in ('image','canvas_image','mask','mask_original','params_npz','keypoints_npz'):
            path=Path(source[key]).resolve()
            if path.is_relative_to(geometry):expected=stage['outputs'][str(path.relative_to(geometry))]['sha256']
            else:expected=source.get('cache_artifacts',{}).get(str(path))
            actual=sha256(path)
            if actual!=expected:raise ValueError(f'Cached Source artifact SHA differs: {path}')
            artifacts[str(path)]=actual;semantic[key]=actual
        source.update(cache_source_person_id=source['person_id'],person_id='source_'+role,role=role,cache_artifacts=artifacts)
        records.append(source);bindings.append(dict(person_id=source['person_id'],pixel_sha256=input_sha,artifacts=semantic))
    return records,dict(producer_fingerprint=stage['fingerprint'],scene_sha256=sha256(scene_path),sources=bindings)

def parse_role_map(text):
    try:
        pairs=[item.strip().split('=') for item in text.split(',')]
        result=dict(pairs)
    except (ValueError,TypeError) as error:
        raise ValueError('role-map must be A=left,B=right or A=right,B=left') from error
    if len(pairs)!=2 or set(result)!={'A','B'} or set(result.values())!={'left','right'}:
        raise ValueError('A/B must map to distinct left/right roles')
    return result

def reference_paths(scene):
    if [s['person_id'] for s in scene['sources']]!=['source_A','source_B']:
        raise ValueError('Sources must retain logical A/B order')
    return [s['canvas_image'] for s in scene['sources']]+[
        scene['guides'][key] for key in ('person_0','person_1','combined')]

def image_records(scene,sha256):
    from PIL import Image
    records=[]
    for index,(role,path) in enumerate(zip(ROLES,reference_paths(scene))):
        with Image.open(path) as image:
            records.append(dict(index=index+1,role=role,path=str(Path(path).resolve()),
                sha256=sha256(path),size=list(image.size)))
    return records
