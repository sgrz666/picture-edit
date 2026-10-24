"""Explicit offline Source-only ArcFace/DINO cache extraction (never runtime)."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.data.canvas_contract import CanvasSpec
from .native_local_condition import NativeLocalReferencesV4,REFERENCE_SCHEMA
from .native_full_assets import sha256


def extract_source_references(job):
    if job.get('role')!='source_only' or 'target_rgb' in job or 'driver_rgb' in job:
        raise ValueError('Appearance extractor accepts Source only')
    sources=job['references']
    if not 1<=len(sources)<=3 or (len(sources)>1 and not job.get('explicit_multi_reference')):
        raise ValueError('Default R=1; declare all legacy multi-reference sources explicitly')
    import onnxruntime as ort
    from transformers import AutoImageProcessor,AutoModel
    from .native_full_assets import read_record
    # No package installation, download, estimator, or asset search here.
    if sha256(job['arcface_model'])!=job['arcface_sha256']: raise ValueError('ArcFace checkpoint SHA differs')
    dino_dir=Path(job['dino_model'])
    if not job['dino_files']: raise ValueError('Declare DINO model/config/processor fingerprints')
    for name,digest in job['dino_files'].items():
        if sha256(dino_dir/name)!=digest: raise ValueError('DINO model/processor SHA differs')
    session=ort.InferenceSession(job['arcface_model'],providers=['CPUExecutionProvider'])
    processor=AutoImageProcessor.from_pretrained(str(dino_dir),local_files_only=True)
    dino=AutoModel.from_pretrained(str(dino_dir),local_files_only=True).float().eval().to(job.get('device','cpu'))
    r=len(sources);arc=torch.zeros(1,2,r,512);av=torch.zeros(1,2,r,dtype=torch.bool)
    face=torch.zeros(1,2,r,256,1536);fv=av.clone();hand=torch.zeros(1,2,2,r,256,1536);hv=torch.zeros(1,2,2,r,dtype=torch.bool)
    boxes_face=torch.zeros(1,2,4);boxes_hand=torch.zeros(1,2,2,4);provenance=[]
    def patches(image,box):
        # PIL pads an out-of-image crop; coordinates are never clamped onto ROI.
        w,h=image.size;x0,y0,x1,y1=box
        if x1<=x0 or y1<=y0 or x1<=0 or y1<=0 or x0>=1 or y0>=1: return None
        crop=image.crop((int(np.floor(x0*w)),int(np.floor(y0*h)),int(np.ceil(x1*w)),int(np.ceil(y1*h))))
        inputs=processor(images=crop,return_tensors='pt',size={'height':224,'width':224},do_center_crop=False)
        with torch.no_grad(): tokens=dino(**{k:v.to(dino.device) for k,v in inputs.items()}).last_hidden_state
        if tokens.shape!=(1,257,1536): raise ValueError('Require ViT-g/14 16x16 patches without register tokens')
        return tokens[0,1:].float().cpu()
    for ref,source in enumerate(sources):
        if sha256(source['rgb']['path'])!=source['rgb']['sha256']: raise ValueError('Source RGB SHA differs')
        spec=CanvasSpec.from_record(source['canvas'])
        with Image.open(source['rgb']['path']) as im: image=spec.image(im.convert('RGB'),fill=(128,128,128))
        if source['person_ids']!=job['source_person_ids']: raise ValueError('Reference arrays must share stable physical source ordering')
        transforms=[]
        def crop_transform(box):
            x0,y0,x1,y1=np.asarray(box,np.float64)*np.array([image.width,image.height,image.width,image.height])
            x0,y0,x1,y1=np.floor(x0),np.floor(y0),np.ceil(x1),np.ceil(y1)
            original=spec.unmap_points(np.asarray(box,np.float32).reshape(2,2))
            affine=None if x1<=x0 or y1<=y0 else [[224/(x1-x0),0.,-x0*224/(x1-x0)],
                [0.,224/(y1-y0),-y0*224/(y1-y0)],[0.,0.,1.]]
            return {'original_xyxy':original.reshape(4).tolist(),'canvas_xyxy':list(box),
                'pixel_crop_unclipped':[float(x0),float(y0),float(x1),float(y1)],'canvas_to_dino224':None if affine is None else np.asarray(affine).tolist()}
        for p,person in enumerate(source['people']):
            # ROIs/5-point alignment come from declared SOURCE geometry/detection,
            # not from the driver or supervision RGB.
            fb=np.asarray(person['face_box'],np.float32);hb=np.asarray(person['hand_boxes'],np.float32)
            crop_record={'face':crop_transform(person['face_box']),'hands':[crop_transform(box) for box in person['hand_boxes']],
                         'canvas_to_arcface112':None}
            if ref==0: boxes_face[0,p]=torch.from_numpy(fb);boxes_hand[0,p]=torch.from_numpy(hb)
            if person.get('face_texture_valid',False):
                value=patches(image,fb)
                if value is not None: face[0,p,ref]=value;fv[0,p,ref]=True
            if person.get('identity_valid',False):
                import cv2
                points=np.asarray(person['arcface_five_points_canvas_pixels'],np.float32)
                if points.shape!=(5,2) or not np.isfinite(points).all(): raise ValueError('Explicit valid Source ArcFace alignment points required')
                template=np.array([[38.2946,51.6963],[73.5318,51.5014],[56.0252,71.7366],[41.5493,92.3655],[70.7299,92.2041]],np.float32)
                affine,_=cv2.estimateAffinePartial2D(points,template,method=cv2.LMEDS)
                if affine is None: raise ValueError('Source face alignment failed')
                crop_record['canvas_to_arcface112']=affine.tolist()
                crop=cv2.warpAffine(np.asarray(image),affine,(112,112),borderValue=0)
                value=(crop.astype(np.float32)-127.5)/127.5
                output=session.run(None,{session.get_inputs()[0].name:value.transpose(2,0,1)[None]})[0]
                if output.shape!=(1,512) or not np.isfinite(output).all(): raise ValueError('ArcFace must output512')
                arc[0,p,ref]=torch.from_numpy(output[0]);av[0,p,ref]=True
            for side in range(2):
                if person['hand_texture_valid'][side]:
                    value=patches(image,hb[side])
                    if value is not None: hand[0,p,side,ref]=value;hv[0,p,side,ref]=True
            transforms.append(crop_record)
        provenance.append({'rgb':source['rgb'],'canvas_id':spec.fingerprint(),'person_ids':source['person_ids'],
            'canvas':spec.metadata(),'original_to_canvas':spec.transform.tolist(),'crop_transforms':transforms,
            'roi_and_alignment':source['people'],'geometry_sha256':source['geometry_sha256']})
    metadata={'schema':REFERENCE_SCHEMA,'role':'source_only','reference_count':r,'reference_sources':provenance,
        'source_rgb_sha256':sources[0]['rgb']['sha256'],'source_canvas_id':CanvasSpec.from_record(sources[0]['canvas']).fingerprint(),
        'source_person_ids':job['source_person_ids'],'arcface_sha256':job['arcface_sha256'],'dino_files':job['dino_files'],
        'arcface_preprocessing':'source_5point_similarity; RGB112; (x-127.5)/127.5', 'dino_preprocessing':'local_processor;224;no_center_crop'}
    return NativeLocalReferencesV4(arc,av,face,fv,hand,hv,boxes_face,boxes_hand,(metadata,)).validate()
