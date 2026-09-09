import sys
sys.path.insert(0, "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers")
import torch
from diffusers import DiffusionPipeline
from PIL import Image
from einops import rearrange

model_path = "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe._load_extras()

prompt = "a beautiful cute red apple on a wooden table, high resolution photography"
cfg_prompt = "blurry, low quality, noise, artifacts"
b = 1

text_inputs = pipe.prepare_text2image_prompts([prompt, cfg_prompt])
hidden_states = pipe.connector_module.meta_queries[None].expand(2 * b, pipe.num_queries, -1)
inputs = pipe.prepare_forward_input(query_embeds=hidden_states, **text_inputs)

with torch.no_grad():
    output = pipe.llm(**inputs, return_dict=True, output_hidden_states=True)
    hidden_states = output.hidden_states
    num_layers = len(hidden_states) - 1
    selected_layers = list(range(num_layers - 1, 0, -6))
    selected_hiddens = [hidden_states[i] for i in selected_layers]
    merged_hidden = torch.cat(selected_hiddens, dim=-1)
    pooled_out, seq_out = pipe.connector_module.llm2dit(merged_hidden)

    # 25 steps Euler
    generator = torch.Generator(device="cuda").manual_seed(42)
    latents = torch.randn(1, 16, 64, 64, generator=generator, device="cuda", dtype=torch.bfloat16)
    
    pipe.scheduler.set_timesteps(25, device="cuda")
    timesteps = pipe.scheduler.timesteps
    print("timesteps:", timesteps)
    
    # Prompt embeds: [negative, positive]
    prompt_embeds = torch.cat([seq_out[1:], seq_out[:1]], dim=0)
    pooled_prompt_embeds = torch.cat([pooled_out[1:], pooled_out[:1]], dim=0)
    
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2)
        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]
        uncond, text = noise_pred.chunk(2)
        noise_pred_guided = uncond + 4.5 * (text - uncond)
        latents = pipe.scheduler.step(noise_pred_guided, t, latents, return_dict=False)[0]
        if i % 5 == 0 or i == len(timesteps) - 1:
            print(f"Step {i}: latents min={latents.min():.3f}, max={latents.max():.3f}, mean={latents.mean():.3f}, std={latents.std():.3f}")

    print("Decoding latents...")
    pixels = pipe.latents_to_pixels(latents)
    print("pixels min/max/mean/std:", pixels.min().item(), pixels.max().item(), pixels.mean().item(), pixels.std().item())
    img = pixels[0]
    img = rearrange(img, 'c h w -> h w c')
    img = torch.clamp(127.5 * img + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    Image.fromarray(img).save("/home/shangguanrz/project/pic-edit/demooutput/manual_test_apple.png")
    print("Saved manual_test_apple.png")