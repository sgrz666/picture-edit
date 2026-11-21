"""Run one existing V7 action with a recorded per-process CUDA allocator cap."""
from pathlib import Path
import argparse
import json
import sys
import time

PROJECT=Path('/home/shangguanrz/project/pic-edit')
sys.path.insert(0,str(PROJECT if PROJECT.exists() else Path(__file__).resolve().parents[4]))


def hidden_offload_hooks(telemetry):
    """Move only large saved Hidden activations, never model weights or masks."""
    def pack(tensor):
        selected=tensor.device.type=='cuda' and tensor.ndim==3 and tensor.shape[-1]==4096 and tensor.numel()>=1024**2
        if selected:
            telemetry['offloaded_tensors']=telemetry.get('offloaded_tensors',0)+1
            telemetry['offloaded_bytes']=telemetry.get('offloaded_bytes',0)+tensor.numel()*tensor.element_size()
            return True,tensor.device,tensor.detach().to('cpu')
        return False,tensor.detach()
    def unpack(packed):
        return packed[2].to(packed[1]) if packed[0] else packed[1]
    return pack,unpack


def apply_execution_policy(backend,*,vae_tiling,memory_fraction,offload_hidden=False):
    import inspect
    from src.pose_control.qwen21_full.native_v6.training.io import file_sha
    vae=backend.pipe.vae
    if vae_tiling:vae.enable_tiling()
    if bool(vae.use_tiling)!=vae_tiling:raise RuntimeError('VAE tiling policy differs from request')
    backend.identity['pilot_execution']={'vae_tiling':vae_tiling,'memory_fraction':memory_fraction,'offload_hidden':offload_hidden,
        'hidden_offload_policy':'saved_hidden_cuda_3d_4096_min1M_to_cpu_v1',
        'tile_settings':{name:getattr(vae,name) for name in ('tile_sample_min_height','tile_sample_min_width',
                                                           'tile_sample_stride_height','tile_sample_stride_width')},
        'wrapper_sha256':file_sha(__file__),'vae_implementation_sha256':file_sha(inspect.getfile(type(vae)))}


def validate_memory_fraction(value):
    if not 0<value<=.95:raise ValueError('GPU fraction must be positive and leave at least 5% headroom')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-fraction',type=float,default=.5)
    parser.add_argument('--resource-report',type=Path,required=True)
    parser.add_argument('--vae-tiling',action='store_true')
    parser.add_argument('--offload-hidden',action='store_true')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    validate_memory_fraction(args.memory_fraction)
    if args.resource_report.exists():raise FileExistsError(args.resource_report)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    free,total=torch.cuda.mem_get_info()
    if total*args.memory_fraction>free-2*1024**3:
        raise RuntimeError('Insufficient free GPU memory for this fixed budget and 2GiB margin')
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    start=time.time();status='failed';error=None
    try:
        from src.pose_control.qwen21_full.native_v7.training import NativeV7Trainer,NativeBackendV7,TrainingConfigV7,stage_preset_v7
        from src.pose_control.qwen21_full.native_v62.training.cli import main as training_main
        offload_totals={}
        class PilotBackend(NativeBackendV7):
            def __init__(self,*values,**kwargs):
                super().__init__(*values,**kwargs)
                apply_execution_policy(self,vae_tiling=args.vae_tiling,memory_fraction=args.memory_fraction,offload_hidden=args.offload_hidden)
            def forward(self,*values,**kwargs):
                if not args.offload_hidden:return super().forward(*values,**kwargs)
                with torch.autograd.graph.saved_tensors_hooks(*hidden_offload_hooks(offload_totals)):
                    return super().forward(*values,**kwargs)
        class PilotTrainer(NativeV7Trainer):
            backend_type=PilotBackend
            def _start(self,*values,**kwargs):
                context=super()._start(*values,**kwargs)
                original=context['optimizer'].step
                logged=False
                def measured_step(*step_args,**step_kwargs):
                    nonlocal logged
                    value=original(*step_args,**step_kwargs)
                    if not logged:
                        telemetry={'stage':self.config.stage.phase,'first_optimizer_update_completed':True,
                            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                            'allocated_bytes':torch.cuda.memory_allocated(),'free_bytes':torch.cuda.mem_get_info()[0]}
                        report=args.resource_report.with_name(args.resource_report.stem+'-first-update.json')
                        with report.open('x') as file:json.dump(telemetry,file,indent=2)
                        print(json.dumps(telemetry),flush=True);logged=True
                    return value
                context['optimizer'].step=measured_step
                return context
        training_main(args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments,
                      config_type=TrainingConfigV7,trainer_type=PilotTrainer,stage_factory=stage_preset_v7)
        status='passed'
    except Exception as exception:
        error=repr(exception)
        raise
    finally:
        result={'status':status,'error':error,'seconds':time.time()-start,'memory_fraction':args.memory_fraction,'vae_tiling':args.vae_tiling,
            'offload_hidden':args.offload_hidden,'offload_telemetry':locals().get('offload_totals',{}),
            'free_before_bytes':free,'total_gpu_bytes':total,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'action':args.arguments}
        args.resource_report.parent.mkdir(parents=True,exist_ok=True)
        with args.resource_report.open('x') as file:json.dump(result,file,indent=2)
        print(json.dumps(result),flush=True)


if __name__=='__main__':main()
