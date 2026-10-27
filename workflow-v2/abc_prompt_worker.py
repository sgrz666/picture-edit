"""PE-I2I with two appearance sources and three retargeted pose references."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from abc_contracts import reference_paths,image_records
from abc_runner import sha256
from src.pose_edit.common import atomic_json
from src.pose_edit.prompt_contract import PE_SAMPLING,parse_pe_answer,validate_generator_prompt

ROLE_CONTRACT=(
    'Produce exactly two people in the scene and composition of <image1>. '
    'Person A identity, face, clothing and body shape come only from <image1>; '
    'person B identity, face, clothing and body shape come only from <image2>. '
    'The requested pose of person A is <image3>, person B is <image4>, '
    'and their joint layout, facing and contact are <image5>. '
    'The three skeleton images are pose instructions, never appearance or background references. '
    'Do not import the background of <image2>, duplicate people, render skeletons, or change clothing.')

def pe_instruction(user_prompt,role_mapping):
    mapping=','.join(f'{key}={role_mapping[key]}' for key in ('A','B'))
    return (ROLE_CONTRACT+' The C roles were already retargeted into image3/image4: '+mapping+'. '
        'Describe concrete visible body actions and mutual contacts from image3-image5. '
        'Return official JSON with rewritten_prompt, ratio_follow="<image1>", wh_ratio="". '
        'Preserve this user request: '+user_prompt)

def compile_prompt(text):
    validate_generator_prompt(text,allowed=(1,2,3,4,5))
    return ROLE_CONTRACT+' '+text.strip()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf8'));settings=request['settings']
    out=Path(request['output_dir']);scene=json.loads(Path(request['scene']).read_text(encoding='utf8'))
    from PIL import Image
    import numpy as np
    import torch
    from transformers import AutoProcessor,AutoModelForImageTextToText
    model_dir=Path(settings['pe_model']);system=(model_dir/'system_prompt.txt').read_text(encoding='utf8')
    system+='\nFor this task the user provides exactly FIVE images with the explicit roles below. '+ROLE_CONTRACT
    instruction=pe_instruction(request['user_prompt'],request['role_mapping'])
    (out/'user_prompt.txt').write_text(request['user_prompt']+'\n',encoding='utf8')
    (out/'effective_system_prompt.txt').write_text(system,encoding='utf8')
    images=[]
    for path in reference_paths(scene):
        with Image.open(path) as image:images.append(image.convert('RGB'))
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(str(model_dir),dtype=torch.bfloat16,local_files_only=True).to('cuda').eval()
    started=time.perf_counter();attempts=[]
    for attempt in range(2):
        seed=settings['pe_seed']+attempt;torch.manual_seed(seed);np.random.seed(seed)
        retry='' if not attempts else '\nPrevious answer rejected: '+attempts[-1]['error']+'. Fix the JSON and image roles.'
        messages=[{'role':'system','content':[{'type':'text','text':system}]},
            {'role':'user','content':[*[{'type':'image','image':im} for im in images],{'type':'text','text':instruction+retry}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
            enable_thinking=True,return_dict=True,return_tensors='pt').to('cuda')
        with torch.inference_mode():
            output=model.generate(**inputs,**{**PE_SAMPLING,'max_new_tokens':settings.get('pe_max_new_tokens',8192)})
        raw=processor.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        (out/f'pe_attempt_{attempt+1}.txt').write_text(raw,encoding='utf8')
        try:
            result=parse_pe_answer(raw);compiled=compile_prompt(result['rewritten_prompt'])
        except ValueError as error:
            attempts.append({'attempt':attempt+1,'error':str(error)});atomic_json(out/'pe_attempts.json',attempts);continue
        attempts.append({'attempt':attempt+1,'accepted':True});break
    else:raise RuntimeError('PE role/JSON validation failed twice; see pe_attempts.json')
    (out/'pe_raw.txt').write_text(raw,encoding='utf8')
    (out/'generation_prompt.txt').write_text(compiled+'\n',encoding='utf8')
    atomic_json(out/'pe_result.json',dict(result=result,compiled_prompt=compiled,attempts=attempts,
        roles=image_records(scene,sha256),user_prompt=request['user_prompt'],instruction=instruction,
        seed=seed,seconds=time.perf_counter()-started,semantic_review='role checks only; inspect action/contact in generated result'))

if __name__=='__main__':main()
