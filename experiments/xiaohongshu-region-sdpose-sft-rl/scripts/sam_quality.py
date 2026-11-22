"""Keep the segmentation qualification gate; unavailable masks are not GT labels."""
import numpy as np


def choose_mask(masks,scores,coverage):
    nonempty=np.asarray(masks).reshape(len(masks),-1).any(1)
    valid=nonempty&np.isfinite(scores)&np.isfinite(coverage)
    accepted=valid&(coverage>=.65)
    available=np.flatnonzero(accepted if accepted.any() else valid)
    if not len(available):raise ValueError('SAM did not produce a nonempty Source mask proxy')
    rank=np.asarray(scores)+.5*np.asarray(coverage)
    return int(available[np.argmax(rank[available])]),bool(accepted.any())


class AssessedForeground:
    def __init__(self,base):self.base=base

    def __call__(self,image,frame):
        import torch
        from src.pose_control.qwen21_full.native_v5.spatial_masks import build_prompts
        points,labels,box=build_prompts(frame)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            self.base.predictor.set_image(np.array(image,copy=True))
            masks,scores,_=self.base.predictor.predict(point_coords=points,point_labels=labels,box=box,multimask_output=True)
        scores=np.asarray(scores).reshape(-1);masks=np.asarray(masks).reshape(-1,image.height,image.width)
        px=np.rint(points[:,0]).astype(int).clip(0,image.width-1);py=np.rint(points[:,1]).astype(int).clip(0,image.height-1)
        coverage=masks[:,py,px].mean(1);index,accepted=choose_mask(masks,scores,coverage)
        return masks[index].astype(np.uint8),{'sam_score':float(scores[index]),'body_prompt_coverage':float(coverage[index]),
            'foreground_fraction':float(masks[index].mean()),'candidate':index,'segmentation_accepted':accepted,
            'coverage_threshold':.65,'source_condition_mask':'SAM weak foreground proxy',
            'spatial_supervision':'available' if accepted else 'unavailable; global FM only; no regional weight redistribution'}
