#!/usr/bin/env python3
"""Exercise the frozen full-run validation code and verify real CPU/CUDA RNG restoration."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--pilot-cache', required=True, type=Path)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.campaign / 'source_runtime'))
    import torch
    from core import read_json, write_json
    from runtime import (FlowMatchEulerDiscreteScheduler, LoraConfig, QwenImage21Transformer2DModel,
                         ResourceMonitor, open_cache, setup, validate)
    config = read_json(args.campaign / 'config.json')
    root = Path(config['workspace_root'])
    setup(config)
    index = open_cache(read_json(root / 'configs/pilot.json'), args.data, args.pilot_cache)
    out = root / 'report' / args.name
    out.mkdir(parents=True, exist_ok=False)
    with ResourceMonitor(config, out / 'resources.json') as monitor:
        model = QwenImage21Transformer2DModel.from_pretrained(config['model_root'], subfolder='transformer',
                                                            torch_dtype=torch.bfloat16)
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=config['rank'], lora_alpha=config['alpha'],
                                    lora_dropout=config['dropout'], init_lora_weights='gaussian',
                                    target_modules=config['lora_modules']))
        model.to('cuda').train()
        model.enable_gradient_checkpointing()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(config['model_root'], subfolder='scheduler')
        before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        validate(model, config, args.data, args.pilot_cache, index, scheduler, out, 0, monitor)
        cpu_preserved = torch.equal(before_cpu, torch.get_rng_state())
        cuda_preserved = torch.equal(before_cuda, torch.cuda.get_rng_state())
        no_gradients = all(p.grad is None for p in model.parameters())
        if not (cpu_preserved and cuda_preserved and no_gradients and model.training):
            raise RuntimeError('Validation changed training RNG, gradient buffers or model mode')
        result = {'pass': True, 'cpu_rng_exact': cpu_preserved, 'cuda_rng_exact': cuda_preserved,
                  'no_gradients': no_gradients, 'training_mode_restored': model.training,
                  'samples': 10, 'density_values': config['validation_density_values'],
                  'scope': 'frozen full-run validation implementation; existing held-out cache only'}
        write_json(out / 'result.json', result)
        print(result, flush=True)


if __name__ == '__main__':
    main()
