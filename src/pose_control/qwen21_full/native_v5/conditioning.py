"""Ordered Qwen multi-image conditions with exact independent source canvases.

Only frozen VLM/VAE results may be stored here. No Adapter features are cached.
"""
from pathlib import Path
import hashlib
import json

import torch
from PIL import Image
from src.data.canvas_contract import CanvasSpec
from ..native_full_assets import sha256
from .person_order import verify_condition

SCHEMA = 'native_v5_qwen_condition_v2'
CACHE_SCHEMA = 'native_v5_frozen_cache_index_v2'
ROLES = {
    'single_2ref': ('source', 'target_pose'),
    'dual_4ref': ('source', 'combined_pose', 'person_0_pose', 'person_1_pose'),
    'dual_source_5ref': ('source_a', 'source_b', 'person_0_pose', 'person_1_pose', 'combined_pose'),
}
SOURCE_ROLES = ('source', 'source_a', 'source_b')


def text_sha(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def verified(ref):
    p = Path(ref['path'])
    if not p.is_absolute() or sha256(p) != ref['sha256']:
        raise ValueError('Qwen condition file/source SHA differs: ' + str(p))
    return p


def read_condition(ref, sample=None):
    spec = json.loads(verified(ref).read_text(encoding='utf8'))
    profile = spec.get('profile')
    if spec.get('schema') != SCHEMA or profile not in ROLES:
        raise ValueError('Explicit Qwen multi-image condition schema/profile required')
    images = spec.get('images', ())
    if tuple(x.get('role') for x in images) != ROLES[profile]:
        raise ValueError('Qwen reference role/order differs')
    if spec.get('primary_source_index') != 0:
        raise ValueError('Only image1 may supply appearance')
    if profile == 'dual_source_5ref' and spec.get('appearance_source_indices') != [0, 1]:
        raise ValueError('Two independent appearance source image indices required')
    prompt = spec['prompt']
    if not isinstance(prompt['text'], str) or not prompt['text'] or text_sha(prompt['text']) != prompt['sha256']:
        raise ValueError('Exact final Prompt SHA differs')
    if prompt.get('field') != 'compiled_prompt_inference':
        raise ValueError('Use the existing final multi-image Plan A prompt')
    verified(prompt['bank'])
    # A dual condition must not describe one person while showing another. The prompt addresses
    # the pair by side ("<image3> ... on the left in <image1>"), so the declared person_order
    # provenance has to agree with that sentence and with the measured source geometry.
    # Conditions assembled before the field existed carry no provenance and keep loading.
    order = spec.get('person_order')
    if order is not None:
        problems = verify_condition(prompt['text'], order)
        if problems:
            raise ValueError('Dual condition text contradicts its image order: '
                             + '; '.join(problems))
    for image in images:
        verified(image['file'])
        canvas = CanvasSpec.from_record(image['canvas'])
        if canvas.fingerprint() != image['canvas_id'] or image.get('space') not in ('original', 'canvas'):
            raise ValueError('Reference Canvas provenance differs')
        if image['role'] not in SOURCE_ROLES and image['canvas_id'] != spec['target_canvas_id']:
            raise ValueError('All pose images must use the target Canvas')
    if sample is not None:
        meta = sample['local_condition_v4'].metadata[0]
        if (images[0]['file']['sha256'] != meta['source_rgb_sha256'] or
                images[0]['canvas_id'] != meta['source_canvas_id'] or
                spec['target_canvas_id'] != meta['target_canvas_id'] or
                spec['pair_id'] != sample['pair_metadata']['pair_id']):
            raise ValueError('Qwen reference Source/target binding differs')
        bank = sample.get('source_bank_v5')
        expected = 'dual_source_5ref' if bank is not None else ('single_2ref' if sample['native_scene_kind'] == 'single' else 'dual_4ref')
        if profile != expected:
            raise ValueError('Qwen condition profile differs from scene kind')
        if bank is not None:
            bank.validate()
            for source in bank.sources:
                image=images[source.qwen_image_index]
                if image['file']['sha256']!=source.rgb_sha256 or image['canvas_id']!=source.record()['canvas_id']:
                    raise ValueError('Independent Source bank/Qwen binding differs')
        if any(x['file']['sha256'] == meta['driver_rgb_sha256'] for x in images):
            raise ValueError('Supervision RGB cannot be a Qwen reference image')
    return spec


def reference_images(spec):
    result = []
    for entry in spec['images']:
        canvas = CanvasSpec.from_record(entry['canvas'])
        with Image.open(verified(entry['file'])) as im:
            im = im.convert('RGB')
            if entry['space'] == 'original':
                im = canvas.image(im, fill=(128, 128, 128) if entry['role'] in SOURCE_ROLES else 0)
            elif (im.height, im.width) != canvas.canvas_hw:
                raise ValueError('Already-canvas guide grid differs; never resize it twice')
            else:
                im = im.copy()
        result.append(im)
    return result


def validate_pair(pair):
    profile = pair.get('qwen_condition_profile')
    if profile not in ROLES or tuple(pair.get('reference_roles', ())) != ROLES[profile]:
        raise ValueError('Explicit reference role/order required')
    grids = pair.get('reference_token_hws', ())
    if len(grids) != len(ROLES[profile]) or any(len(g) != 2 or min(g) <= 0 or g[0]*g[1] % 4 for g in grids):
        raise ValueError('Native reference grids require complete four-token slots')
    ref = pair['reference_latents']
    count = sum(h*w for h, w in grids)
    if ref.shape != (1, count, 64) or tuple(pair['source_token_hw']) != tuple(grids[0]):
        raise ValueError('Actual Qwen64 references/grid differ')
    n = grids[0][0]*grids[0][1]
    if pair['source_latent'].shape != (1, n, 64) or not torch.equal(pair['source_latent'], ref[:, :n]):
        raise ValueError('Primary Source must exclude every skeleton latent')
    h, w = pair['target_token_hw']
    if h*w % 4:
        raise ValueError('Native target grid requires complete four-token slots')
    mask = pair['img_mask']
    if (mask.dtype != torch.bool or mask.shape[0] != 1 or int(mask.sum())*4 != count+h*w or
            mask.shape[1] != pair['prompt_embeds'].shape[1]+h*w//4 or
            pair['prompt_mask'].shape != pair['prompt_embeds'].shape[:2]):
        raise ValueError('VLM image slot count differs from all references plus target')
    return pair


def primary_source(pair):
    if 'reference_latents' not in pair:
        return pair['source_latent']
    validate_pair(pair)
    n = pair['reference_token_hws'][0][0]*pair['reference_token_hws'][0][1]
    return pair['reference_latents'][:, :n]


def reference_inputs(pair, noisy_target, sigma):
    validate_pair(pair)
    th, tw = pair['target_token_hw']
    if noisy_target.shape != (1, th*tw, 64):
        raise ValueError('Target latent grid differs')
    return dict(hidden_states=torch.cat((pair['reference_latents'], noisy_target), 1), timestep=sigma,
        encoder_hidden_states=pair['prompt_embeds'], encoder_hidden_states_mask=pair['prompt_mask'],
        img_shapes=[[(1, h, w) for h, w in pair['reference_token_hws']]+[(1, th, tw)]],
        img_mask=pair['img_mask'], return_dict=False)


@torch.no_grad()
def encode_pair(pipe, sample, spec, *, include_target=True):
    """Explicit frozen preprocessing; no implicit geometry/appearance extraction."""
    from ..qwen_runtime import dataset_image_to_pil
    images = reference_images(spec)
    device = next(pipe.text_encoder.parameters()).device
    embeds, mask, image_mask = pipe.encode_prompt(spec['prompt']['text'], image=images, device=device)
    if mask is None:
        mask = torch.ones(embeds.shape[:2], device=device, dtype=torch.bool)
    vae_device = next(pipe.vae.parameters()).device

    def encode(image):
        value = pipe.image_processor.preprocess(image.convert('RGBA'), width=image.width, height=image.height)
        value = value.unsqueeze(2).to(device=vae_device, dtype=pipe.vae.dtype)
        latent = pipe._encode_vae_image(value, torch.Generator(device=vae_device).manual_seed(0))
        return latent.flatten(3).squeeze(2).transpose(1, 2).cpu().contiguous()

    refs = [encode(im) for im in images]
    th, tw = sample['target_token_hw']
    pair = dict(qwen_condition_profile=spec['profile'], reference_latents=torch.cat(refs, 1),
        reference_token_hws=[(im.height//16, im.width//16) for im in images],
        reference_roles=[x['role'] for x in spec['images']], source_latent=refs[0],
        reference_sha256s=[x['file']['sha256'] for x in spec['images']],
        source_token_hw=tuple(sample['source_token_hw']), target_token_hw=(th, tw),
        prompt_embeds=embeds.cpu().contiguous(), prompt_mask=mask.cpu().bool(),
        img_mask=torch.cat((image_mask.cpu().bool(), torch.ones(1, th*tw//4, dtype=torch.bool)), 1),
        prompt_sha256=spec['prompt']['sha256'])
    if include_target:pair['target_latent']=encode(dataset_image_to_pil(sample['tgt_image']))
    return validate_pair(pair)


@torch.no_grad()
def encode_legacy_source(pipe,sample,prompt):
    """Legacy one-reference weights on an exact rectangular Source canvas."""
    from ..qwen_runtime import dataset_image_to_pil
    image=dataset_image_to_pil(sample['src_image'])
    device=next(pipe.text_encoder.parameters()).device
    embeds,mask,image_mask=pipe.encode_prompt(prompt,image=[image],device=device)
    vd=next(pipe.vae.parameters()).device
    value=pipe.image_processor.preprocess(image.convert('RGBA'),width=image.width,height=image.height)
    value=value.unsqueeze(2).to(device=vd,dtype=pipe.vae.dtype)
    latent=pipe._encode_vae_image(value,torch.Generator(device=vd).manual_seed(0)).flatten(3).squeeze(2).transpose(1,2)
    th,tw=sample['target_token_hw']
    return dict(source_latent=latent,source_token_hw=sample['source_token_hw'],target_token_hw=(th,tw),
        prompt_embeds=embeds,prompt_mask=mask if mask is not None else torch.ones(embeds.shape[:2],device=device,dtype=torch.bool),
        img_mask=torch.cat((image_mask,torch.ones(1,th*tw//4,device=device,dtype=torch.bool)),1))
