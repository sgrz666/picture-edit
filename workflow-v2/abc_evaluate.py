"""Report measured pose/appearance and explicit detector uncertainty for ABC."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.pose_edit.common import atomic_json

def unavailable_report(reason):
    return {**{key:{'status':'unavailable','reason':reason} for key in
        ('person_count','pose','appearance','background')},'passed':None}

def evaluate(request):
    import cv2
    from PIL import Image,ImageDraw
    from src.pose_edit import dwpose
    from src.pose_edit.appearance import mask_box,crop_person,load_embedder
    from src.pose_edit.geometry import smplx137_to_dwpose134
    spec=importlib.util.spec_from_file_location('abc_legacy_evaluation',ROOT/'workflow-multiple/evaluate_dual.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    settings=request['settings'];out=Path(request['output_dir'])
    scene=json.loads(Path(request['scene']).read_text(encoding='utf8'))
    image_path=Path(request['image']).with_name('final_canvas.png')
    with Image.open(image_path) as im:image=im.convert('RGB')
    arrays=np.load(scene['geometry_arrays'],allow_pickle=False);K=arrays['K']
    requested=[]
    for joints in arrays['joints_137_cam']:
        uv=joints@K.T;requested.append(smplx137_to_dwpose134(uv[:,:2]/uv[:,2:]))
    sessions=dwpose.load_sessions(settings)
    detections=dwpose.detect_people(settings,cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR),sessions)
    mapping,assignment=legacy.assign(detections,requested,.5,image.size)
    basis='whole_image';assigned=[detections[i] for i in mapping] if mapping is not None else [None,None]
    region_counts=[]
    if len(detections)!=2 or mapping is None:
        basis='regions';assigned=[]
        for p in range(2):
            box=mask_box(arrays['mask'][p]);x0,y0,x1,y1=box
            dx,dy=(x1-x0)*.2,(y1-y0)*.2
            x0,y0=max(0,int(x0-dx)),max(0,int(y0-dy));x1,y1=min(image.width,int(x1+dx)),min(image.height,int(y1+dy))
            crop=image.crop((x0,y0,x1,y1));people=dwpose.detect_people(settings,cv2.cvtColor(np.asarray(crop),cv2.COLOR_RGB2BGR),sessions)
            region_counts.append(len(people))
            for person in people:
                person['points']+=np.array([x0,y0]);person['box_xyxy']=(np.asarray(person['box_xyxy'])+[x0,y0,x0,y0]).tolist()
            chosen=min(people,key=lambda q:np.linalg.norm(np.nanmean(q['points'][[8,11]],0)-np.nanmean(requested[p][[8,11]],0))) if people else None
            assigned.append(chosen)
    pose=[legacy.score_person(target,person,.5,image.size) for target,person in zip(requested,assigned)]
    # Contacts in generated RGB are 2D observations, never recovered 3D truth.
    def pelvis(points):
        return np.asarray(points)[[8,11]].mean(0)
    relative=None
    if all(p is not None and (np.asarray(p['scores'])[[8,11]]>=.5).all() for p in assigned):
        expected=pelvis(requested[1])-pelvis(requested[0]);observed=pelvis(assigned[1]['points'])-pelvis(assigned[0]['points'])
        relative=dict(expected_px=expected.tolist(),observed_px=observed.tolist(),error_px=float(np.linalg.norm(expected-observed)))
    overlay=image.copy();draw=ImageDraw.Draw(overlay)
    for p,(target,person) in enumerate(zip(requested,assigned)):
        for x,y in target[:18]:
            if np.isfinite([x,y]).all():draw.ellipse((x-4,y-4,x+4,y+4),outline='white',width=2)
        if person is not None:
            for (x,y),score in zip(person['points'][:18],person['scores'][:18]):
                if score>=.5 and np.isfinite([x,y]).all():draw.ellipse((x-2,y-2,x+2,y+2),fill=('lime' if p==0 else 'cyan'))
    overlay.save(out/'pose_overlay.png')
    report=unavailable_report('not measured')
    report['person_count']=dict(status='measured' if len(detections)==2 else 'uncertain',whole_image=len(detections),
        requested=2,region_counts=region_counts,reason='overlap may merge detections; regional detections may duplicate one body')
    report['pose']=dict(status='measured',basis=basis,assignment=assignment,persons=pose,
        missing_landmarks=[18-int(row.get('body_joints',0)) for row in pose])
    # Identity uses a separate aligned-face metric. Clothing/body DINO remains
    # an appearance proxy and is never relabelled biometric identity.
    import onnxruntime as ort
    face_session=ort.InferenceSession(settings['arcface_model'],providers=['CPUExecutionProvider'])
    def face_feature(rgb,points,scores):
        groups=(list(range(36,42)),list(range(42,48)),[30],[48],[54])
        points=np.asarray(points)[24:92];scores=np.asarray(scores)[24:92]
        if not all((scores[g]>=.5).all() and np.isfinite(points[g]).all() for g in groups):return None
        five=np.stack([points[g].mean(0) for g in groups]).astype(np.float32)
        target=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],np.float32)
        affine,_=cv2.estimateAffinePartial2D(five,target,method=cv2.LMEDS)
        if affine is None:return None
        crop=cv2.warpAffine(np.asarray(rgb),affine,(112,112),borderValue=0)
        value=((crop.astype(np.float32)-127.5)/127.5).transpose(2,0,1)[None]
        feature=face_session.run(None,{face_session.get_inputs()[0].name:value})[0][0]
        return feature/max(float(np.linalg.norm(feature)),1e-12)
    source_faces=[]
    for source in scene['sources']:
        with np.load(source['keypoints_npz'],allow_pickle=False) as observation,Image.open(source['image']) as rgb:
            source_faces.append(face_feature(rgb.convert('RGB'),observation['points'],observation['scores']))
    generated_faces=[None if p is None else face_feature(image,p['points'],p['scores']) for p in assigned]
    similarities=[[None if a is None or b is None else float(np.dot(a,b)) for b in generated_faces] for a in source_faces]
    available=sum(v is not None for row in similarities for v in row)
    report['identity']=dict(status='measured' if available==4 else ('partial' if available else 'unavailable'),
        arcface_cosine_matrix=similarities,assigned_cosine=[similarities[i][i] for i in range(2)],
        alignment='DWPose face68 five-point similarity; CPU ArcFace RGB112',
        unavailable_reason='missing or low-confidence face landmarks' if any(v is None for row in similarities for v in row) else None,
        assignment_uncertain=basis=='regions' or bool(assignment.get('ambiguous',False)))
    # Offline local weights only; no implicit torch.hub downloads.
    weights=Path(settings['appearance_dino_model'])
    if weights.is_file() and all(person is not None for person in assigned):
        import torch
        repo=Path(settings['dinov2_repo']);model=torch.hub.load(str(repo),'dinov2_vits14',source='local',pretrained=False)
        state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True);model.eval()
        def embed(im):
            array=np.asarray(im.resize((224,224)),np.float32)/255.
            value=torch.from_numpy(array).permute(2,0,1)[None]
            value=(value-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
            with torch.inference_mode():v=model(value)[0].numpy()
            return v/max(np.linalg.norm(v),1e-12)
        sources=[]
        for source in scene['sources']:
            with Image.open(source['canvas_image']) as im:rgb=im.convert('RGB')
            with Image.open(source['mask']) as mask:box=mask_box(np.asarray(mask)>0)
            sources.append(embed(crop_person(rgb,box,shape='square')))
        generated=[embed(crop_person(image,p['box_xyxy'],shape='square')) for p in assigned]
        similarity=np.stack(sources)@np.stack(generated).T
        report['appearance']=dict(status='measured',cosine_matrix=similarity.tolist(),
            assigned_cosine=np.diag(similarity).tolist(),basis='DINOv2 body/clothing appearance proxy, not biometric identity')
    source=scene['sources'][0]
    with Image.open(source['canvas_image']) as im:background=np.asarray(im.convert('RGB'),np.float32)
    with Image.open(source['mask']) as mask:source_mask=np.asarray(mask)>0
    from abc_quality import background_support,review_artifacts
    from src.data.canvas_contract import CanvasSpec
    valid=CanvasSpec.from_record(source['canvas']).valid_mask().numpy().squeeze()>0
    support=background_support(valid,source_mask,arrays['mask'])
    if support.any():
        mse=float(np.square(background-np.asarray(image,np.float32))[support].mean())
        report['background']=dict(status='measured',pixels=int(support.sum()),mae=float(np.abs(background-np.asarray(image,np.float32))[support].mean()),
            psnr_db=None if mse==0 else float(10*np.log10(255**2/mse)),basis='outside source and requested target regions; repainting allowed')
    retained=[]
    for source,target in zip(scene['sources'],scene['targets']):
        with np.load(source['params_npz'],allow_pickle=False) as s,np.load(target['params_npz'],allow_pickle=False) as t:
            retained.append({key:bool(np.array_equal(s[key],t[key])) for key in ('betas','expression','jaw_pose')})
    report.update(schema='abc_evaluation_v1',workflow_executed=True,passed=None,
        geometry_diagnostics=scene['diagnostics'],quality_verdict='inspect metrics and overlays; no trained quality threshold asserted',
        body_shape={'status':'measured_in_geometry_only','source_parameter_preservation':retained,
            'betas_preserved':all(p['betas'] for p in retained),'generated_3d_shape':'not_estimated'},
        generated_contact={'status':'unavailable','reason':'RGB does not provide verified 3D contact distance'})
    report['relative_pelvis']={'status':'measured' if relative else 'unavailable','value':relative,
        'assignment_uncertain':basis=='regions' or bool(assignment.get('ambiguous',False))}
    duplicate=False
    if all(p is not None for p in assigned):
        duplicate=bool(np.linalg.norm(pelvis(assigned[0]['points'])-pelvis(assigned[1]['points']))<.02*max(image.size))
    report['person_count']['possible_duplicate_region_assignment']=duplicate
    if duplicate:report['person_count']['status']='uncertain'
    report['visual_review']=review_artifacts(image,arrays,out)
    from abc_quality import pose_coverage
    report['pose']['coverage']=pose_coverage(pose)
    report['annotation_scope']=dict(interaction_source=scene['interaction'].get('source'),
        requested_geometry_is_generated_target=True,driver_is_not_output_ground_truth=True,
        official_finger_pose_available=False if 'official' in str(scene['interaction'].get('source','')).lower() else None,
        upper_body_handshake_caveat='Estimated lower body is not annotated truth; report target-fit metrics separately from observed driver evidence.')
    from abc_quality import contact_projection_proxy
    report['contact_2d_proxy']=contact_projection_proxy(arrays,requested,assigned)
    atomic_json(out/'evaluation.json',report)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    evaluate(json.loads(args.request.read_text(encoding='utf8')))

if __name__=='__main__':main()
