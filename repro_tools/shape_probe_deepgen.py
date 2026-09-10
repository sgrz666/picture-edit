from pathlib import Path

import torch
from PIL import Image
from diffusers import DiffusionPipeline


MODEL_PATH = Path(
    "/home/lvjy/projects/picture-edit/models/DeepGen-1.0-diffusers"
)
INPUT_PATH = Path(
    "/home/lvjy/projects/picture-edit/inputs/person1.jpg"
)
OUTPUT_PATH = Path(
    "/home/lvjy/projects/picture-edit/"
    "repro_outputs/deepgen_baseline/shape_probe_1step.png"
)

PROMPT = (
    "Change the person's pose to wave one hand in a friendly greeting, "
    "while preserving the same person's identity, face, clothing, and background."
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

printed_transformer = False
pos_embed_count = 0
first_block_pre_printed = False
first_block_post_printed = False


def shape(x):
    if torch.is_tensor(x):
        return tuple(x.shape)
    return type(x).__name__


def describe_cond_latents(cond):
    print("\n===== COND / SOURCE LATENTS =====")

    if cond is None:
        print("cond_hidden_states = None")
        return []

    print("outer type =", type(cond).__name__)
    print("CFG/sample entries =", len(cond))

    token_counts = []

    for sample_idx, refs in enumerate(cond):
        print(
            f"sample[{sample_idx}] refs = {len(refs)} "
            f"type={type(refs).__name__}"
        )

        sample_tokens = 0

        for ref_idx, ref in enumerate(refs):
            print(
                f"  ref[{ref_idx}] latent shape = {tuple(ref.shape)} "
                f"dtype={ref.dtype}"
            )

            # ref shape is expected to be [C,H,W]
            if ref.ndim == 3:
                patch = 2
                n = (ref.shape[-2] // patch) * (ref.shape[-1] // patch)
                sample_tokens += n

                print(
                    f"  ref[{ref_idx}] expected DiT tokens "
                    f"= {n}"
                )

        token_counts.append(sample_tokens)

    return token_counts


print("===== LOAD DEEPGEN =====")

pipe = DiffusionPipeline.from_pretrained(
    str(MODEL_PATH),
    torch_dtype=DTYPE,
    trust_remote_code=True,
    local_files_only=True,
)

pipe.to(DEVICE)

print("pipeline class    =", pipe.__class__.__name__)
print("transformer class =", pipe.transformer.__class__.__name__)
print("vae class         =", pipe.vae.__class__.__name__)

print("\n===== TRANSFORMER CONFIG =====")
print("in_channels        =", pipe.transformer.config.in_channels)
print("patch_size         =", pipe.transformer.config.patch_size)
print("num_layers         =", pipe.transformer.config.num_layers)
print("num_heads          =", pipe.transformer.config.num_attention_heads)
print("attention_head_dim =", pipe.transformer.config.attention_head_dim)

hidden_dim = (
    pipe.transformer.config.num_attention_heads
    * pipe.transformer.config.attention_head_dim
)

print("hidden_dim         =", hidden_dim)
vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
print("vae_scale_factor   =", vae_scale_factor)


# ------------------------------------------------------------
# 1. Wrap transformer.forward:
#    observe actual noisy target latent + source cond latents
# ------------------------------------------------------------

original_transformer_forward = pipe.transformer.forward


def transformer_forward_probe(*args, **kwargs):
    global printed_transformer

    if not printed_transformer:
        printed_transformer = True

        print("\n" + "=" * 70)
        print("TRANSFORMER INPUT PROBE")
        print("=" * 70)

        hs = kwargs.get("hidden_states")
        cond = kwargs.get("cond_hidden_states")
        enc = kwargs.get("encoder_hidden_states")
        pooled = kwargs.get("pooled_projections")
        timestep = kwargs.get("timestep")

        if hs is None and len(args) > 0:
            hs = args[0]

        print("\n===== TARGET / NOISY LATENT =====")
        print("hidden_states shape =", shape(hs))

        if torch.is_tensor(hs):
            print("dtype =", hs.dtype)
            print("device =", hs.device)

            patch = pipe.transformer.config.patch_size
            target_tokens = (
                hs.shape[-2] // patch
            ) * (
                hs.shape[-1] // patch
            )

            print("target tokens/sample =", target_tokens)
        else:
            target_tokens = None

        ref_token_counts = describe_cond_latents(cond)

        print("\n===== TEXT / SCB CONDITION =====")
        print("encoder_hidden_states =", shape(enc))
        print("pooled_projections    =", shape(pooled))
        print("timestep              =", shape(timestep))

        print("\n===== EXPECTED CONCATENATION =====")

        if target_tokens is not None:
            for i, ref_tokens in enumerate(ref_token_counts):
                total = target_tokens + ref_tokens
                print(
                    f"sample[{i}]: target={target_tokens} "
                    f"+ refs={ref_tokens} "
                    f"=> expected image tokens={total}"
                )

        print("=" * 70 + "\n")

    return original_transformer_forward(*args, **kwargs)


pipe.transformer.forward = transformer_forward_probe


# ------------------------------------------------------------
# 2. Hook PatchEmbed:
#    verify latent -> DiT token transformation
# ------------------------------------------------------------

def pos_embed_hook(module, inputs, output):
    global pos_embed_count

    if pos_embed_count >= 8:
        return

    pos_embed_count += 1

    x = inputs[0] if inputs else None

    print(
        f"[POS_EMBED #{pos_embed_count}] "
        f"in={shape(x)} -> out={shape(output)}"
    )


pos_handle = pipe.transformer.pos_embed.register_forward_hook(
    pos_embed_hook
)


# ------------------------------------------------------------
# 3. Hook first Transformer block:
#    by here target + source should already be concatenated/padded
# ------------------------------------------------------------

first_block = pipe.transformer.transformer_blocks[0]


def first_block_pre_hook(module, args, kwargs):
    global first_block_pre_printed

    if first_block_pre_printed:
        return

    first_block_pre_printed = True

    hs = kwargs.get("hidden_states")
    enc = kwargs.get("encoder_hidden_states")
    mask = kwargs.get("attention_mask")
    temb = kwargs.get("temb")

    print("\n" + "=" * 70)
    print("FIRST DIT BLOCK INPUT")
    print("=" * 70)

    print("joined hidden_states  =", shape(hs))
    print("encoder_hidden_states =", shape(enc))
    print("attention_mask        =", shape(mask))
    print("temb                  =", shape(temb))

    if torch.is_tensor(mask):
        print(
            "valid image tokens/sample =",
            mask.sum(dim=1).detach().cpu().tolist()
        )

    print("=" * 70 + "\n")


def first_block_post_hook(module, args, kwargs, output):
    global first_block_post_printed

    if first_block_post_printed:
        return

    first_block_post_printed = True

    print("\n===== FIRST DIT BLOCK OUTPUT =====")

    if isinstance(output, tuple):
        for i, x in enumerate(output):
            print(f"output[{i}] =", shape(x))
    else:
        print("output =", shape(output))


pre_handle = first_block.register_forward_pre_hook(
    first_block_pre_hook,
    with_kwargs=True,
)

post_handle = first_block.register_forward_hook(
    first_block_post_hook,
    with_kwargs=True,
)


# ------------------------------------------------------------
# 4. Run official-style image editing.
#    Only 1 denoising step: enough to inspect shapes.
# ------------------------------------------------------------

print("\n===== RUN 1-STEP SHAPE PROBE =====")

source_image = Image.open(INPUT_PATH).convert("RGB")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    result = pipe(
        prompt=PROMPT,
        image=source_image,
        negative_prompt="",
        height=512,
        width=512,
        num_inference_steps=1,
        guidance_scale=4.0,
        seed=42,
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.images[0].save(OUTPUT_PATH)


# remove hooks
pos_handle.remove()
pre_handle.remove()
post_handle.remove()


print("\n===== FINAL =====")
print("probe completed")
print("probe image =", OUTPUT_PATH)
print(
    "peak VRAM GB =",
    round(torch.cuda.max_memory_allocated() / 1024**3, 3)
)

print("\nNOTE:")
print(
    "The 1-step image is NOT a quality result. "
    "It exists only to complete a real DeepGen editing forward pass."
)
