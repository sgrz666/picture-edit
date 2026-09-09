import sys
sys.path.insert(0, "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
import torch
from diffusers import DiffusionPipeline

model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Let us run __call__ internals manually to debug
pipe._load_extras()
prompt = "a cute red apple"
cfg_prompt = ""
b = 1
text_inputs = pipe.prepare_text2image_prompts([prompt, cfg_prompt])
hidden_states = pipe.connector_module.meta_queries[None].expand(2 * b, pipe.num_queries, -1)
inputs = pipe.prepare_forward_input(query_embeds=hidden_states, **text_inputs)
print("inputs_embeds:", inputs["inputs_embeds"].shape, inputs["inputs_embeds"].min().item(), inputs["inputs_embeds"].max().item(), inputs["inputs_embeds"].mean().item())

with torch.no_grad():
    output = pipe.llm(**inputs, return_dict=True, output_hidden_states=True)
    hidden_states = output.hidden_states
    print(f"num hidden states: {len(hidden_states)}")
    num_layers = len(hidden_states) - 1
    selected_layers = list(range(num_layers - 1, 0, -6))
    print("selected_layers:", selected_layers)
    selected_hiddens = [hidden_states[i] for i in selected_layers]
    merged_hidden = torch.cat(selected_hiddens, dim=-1)
    print("merged_hidden:", merged_hidden.shape, merged_hidden.min().item(), merged_hidden.max().item())
    pooled_out, seq_out = pipe.connector_module.llm2dit(merged_hidden)
    print("pooled_out:", pooled_out.shape, pooled_out.min().item(), pooled_out.max().item(), pooled_out.mean().item(), pooled_out.std().item())
    print("seq_out:", seq_out.shape, seq_out.min().item(), seq_out.max().item(), seq_out.mean().item(), seq_out.std().item())

    # Now let us check SD3 transformer
    latents = torch.randn(1, 16, 64, 64, device="cuda", dtype=torch.bfloat16)
    pipe.scheduler.set_timesteps(10, device="cuda")
    timesteps = pipe.scheduler.timesteps
    print("timesteps:", timesteps)
    t = timesteps[0]
    latent_model_input = torch.cat([latents] * 2)
    timestep = t.expand(latent_model_input.shape[0])
    prompt_embeds = torch.cat([seq_out[1:], seq_out[:1]], dim=0) # [uncond, cond]
    pooled_prompt_embeds = torch.cat([pooled_out[1:], pooled_out[:1]], dim=0)
    print("prompt_embeds:", prompt_embeds.shape, "pooled_prompt_embeds:", pooled_prompt_embeds.shape)
    
    noise_pred = pipe.transformer(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        return_dict=False
    )[0]
    print("noise_pred:", noise_pred.shape, noise_pred.min().item(), noise_pred.max().item(), noise_pred.mean().item(), noise_pred.std().item())
    
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guidance_scale = 4.5
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    print("noise_pred_guided:", noise_pred_guided.shape, noise_pred_guided.min().item(), noise_pred_guided.max().item(), noise_pred_guided.mean().item(), noise_pred_guided.std().item())
    
    step_out = pipe.scheduler.step(noise_pred_guided, t, latents, return_dict=False)[0]
    print("latents after step 1:", step_out.shape, step_out.min().item(), step_out.max().item(), step_out.mean().item(), step_out.std().item())