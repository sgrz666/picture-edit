import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch import Tensor
from omegaconf import OmegaConf
import wandb

from src.flux.modules.layers import LoRALinearLayer, LastLayer
from src.flux.train_utils import *
from src.flux.util import load_ae, load_clip, load_flow_model2, load_t5

import datetime
import logging
import os
import torch.distributed as dist
from src.flux.fsdp_utils import setup_model, build_dataloader, save_model_checkpoint, save_optimizer_checkpoint
from tqdm import tqdm
try:
    from image_datasets.combined_dataset_ar_prepared import MultiHumanDataset
except ImportError as exc:
    MultiHumanDataset = None
    _DATASET_IMPORT_ERROR = exc
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, prepare_dual, prepare_dual_train, prepare_dual_train_ar
import time
import contextlib
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from einops import rearrange
import random
import torch.nn.functional as F
from src.flux.xflux_pipeline import XFluxSampler
from PIL import Image, ImageDraw, ImageFont

################################ Split head for Img_in and Final_Layer#########################
class ImgInSplit(nn.Module): # must call after loading pre-trained model
    def __init__(self, old_img_in: nn.Linear, keep_ori_weights: bool = False, zero_init: bool = False):
        super().__init__()
        assert not (keep_ori_weights and zero_init), "keep_ori_weights and zero_init cannot be both True"
        self.old_img_in = old_img_in

        self.pose_in = copy.deepcopy(old_img_in)
        if not keep_ori_weights:
            if zero_init:
                nn.init.zeros_(self.pose_in.weight)
                nn.init.zeros_(self.pose_in.bias)
            else:
                nn.init.normal_(self.pose_in.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.pose_in.bias)


    def forward(self, x: Tensor, img_seq_len: int) -> Tensor:
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"
        B, L, D = x.shape
        pose_len = L - img_seq_len

        x_pose = x[:, :pose_len, :]
        x_img = x[:, pose_len:, :]

        x_img = self.old_img_in(x_img)
        x_pose = self.pose_in(x_pose)

        return torch.cat([x_pose, x_img], dim=1)

    def forward_pose_only(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"

        x_pose = x
        x_pose = self.pose_in(x_pose)

        return x_pose


class LastLayerSplitTwoMod(nn.Module): # two vec condition version
    """
    Same math as the original LastLayer, but with
      • two independent output heads (linear1, linear2)
      • two independent AdaLN modulators (ada1, ada2)

    Args
    ----
    old_layer : a *loaded* LastLayer whose weights you want to duplicate.
    """

    def __init__(self, old_layer: "LastLayer", keep_ori_weights: bool = False, zero_init: bool = False):
        super().__init__()
        self.old_layer = old_layer

        # duplicate AdaLN MLPs
        self.adaLN_modulation_pose = copy.deepcopy(old_layer.adaLN_modulation)
        if not keep_ori_weights:
            if zero_init:
                nn.init.zeros_(self.adaLN_modulation_pose[1].weight)
                nn.init.zeros_(self.adaLN_modulation_pose[1].bias)
            else:
                nn.init.normal_(self.adaLN_modulation_pose[1].weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.adaLN_modulation_pose[1].bias)

        # duplicate output heads
        self.linear_pose_img = copy.deepcopy(old_layer.linear)
        if not keep_ori_weights:
            if zero_init:
                nn.init.zeros_(self.linear_pose_img.weight)
                nn.init.zeros_(self.linear_pose_img.bias)
            else:
                nn.init.normal_(self.linear_pose_img.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.linear_pose_img.bias)

    # ---------------------------------------------------------------------
    def forward(self, x: Tensor, vec1: Tensor, vec2: Tensor, img_seq_len: int) -> Tensor:
        """
        x     : (B, 2*T, hidden_size)
        vec1  : (B, hidden_size)  – conditioning for the *first* half
        vec2  : (B, hidden_size)  – conditioning for the *second* half
        """
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"
        B, L, D = x.shape
        pose_len = L - img_seq_len

        x_pose = x[:, :pose_len, :]
        x_img = x[:, pose_len:, :]

        # branch 1
        shift, scale = self.old_layer.adaLN_modulation(vec1).chunk(2, dim=1)
        x_img = (1 + scale[:, None, :]) * self.old_layer.norm_final(x_img) + shift[:, None, :]
        x_img = self.old_layer.linear(x_img)

        # branch 2
        shift_pose, scale_pose = self.adaLN_modulation_pose(vec2).chunk(2, dim=1)
        x_pose = (1 + scale_pose[:, None, :]) * self.old_layer.norm_final(x_pose) + shift_pose[:, None, :] # ERROR!
        x_pose = self.linear_pose_img(x_pose)

        return torch.cat([x_pose, x_img], dim=1)

    def forward_pose_only(self, x: Tensor, vec2: Tensor) -> Tensor:
        """
        x     : (B, 2*T, hidden_size)
        vec1  : (B, hidden_size)  – conditioning for the *first* half
        vec2  : (B, hidden_size)  – conditioning for the *second* half
        """
        assert x.dim() == 3, "x should be in shape (B, L1+L2, D)"
        x_pose = x

        # branch 2
        shift_pose, scale_pose = self.adaLN_modulation_pose(vec2).chunk(2, dim=1)
        x_pose = (1 + scale_pose[:, None, :]) * self.old_layer.norm_final(x_pose) + shift_pose[:, None, :] # ERROR!
        x_pose = self.linear_pose_img(x_pose)

        return x_pose


def replace_split_head(dit, args):
    old_img_in = dit.img_in
    dit.img_in = ImgInSplit(old_img_in, keep_ori_weights=args.keep_ori_weights, zero_init=args.zero_init)

    old_final_layer = dit.final_layer
    dit.final_layer = LastLayerSplitTwoMod(old_final_layer, keep_ori_weights=args.keep_ori_weights, zero_init=args.zero_init)

def reduce_loss(loss: torch.Tensor) -> float:
    """
    loss : scalar tensor on *this* rank (already averaged over local-batch)
    returns : python float = mean(loss) over all ranks
    """
    with torch.no_grad():
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)   # Σ over ranks
        loss /= dist.get_world_size()                 # → average
    return loss.item()


def sample_step(dit, args, global_step, wandbrun, rank, offload=True):
    image_name = "Masked Attention" if args.use_attn_mask else "Full Attention"
    local_gpu = torch.cuda.current_device()
    print(f"Sampling images for step {global_step}...")
    sampler = XFluxSampler(clip=None, t5=None, ae=None, model=dit, device=f"cuda:{local_gpu}", offload=offload)

    if args.customize_txt_ids:
        prompts = args.sample_prompts['prompt_list'] # ["A", "B"]
        bounding_boxes_in_order = args.sample_prompts['bounding_boxes_in_order']
    else:
        prompts = [v for k, v in args.sample_prompts.items() if k.startswith('prompt_')] # ["A", "B"]
        bounding_boxes_in_order = None
    general_prompt = args.sample_prompts['general_prompt']
    all_rounds_images = []

    # Use iterative sampling with multiple rounds
    rounds_output, clip, t5, vae = sampler.forward_multiperson(
        prompts=prompts,
        general_prompt=general_prompt,
        width=args.sample_width,
        height=args.sample_height,
        num_steps=args.sample_steps,
        seed=args.sample_seeds[0],
        customize_img_ids=args.customize_img_ids,
        customize_txt_ids=args.customize_txt_ids,
        bounding_boxes_in_order=bounding_boxes_in_order,
        generate_img_ids_type=args.generate_img_ids_type,
        background_color=args.background_color,
        attn_mask_type=args.get("attn_mask_type", None), # None default -- if None, draw back to json2.py
        apply_smaller_pose=args.apply_smaller_pose,
        guidance=3.5 #args.guidance
    )

    # add visualization codes here
    # Helper to create a centered text banner of given width
    def _create_text_banner(text: str, width: int, font: ImageFont.FreeTypeFont, padding: int = 10, bg_color: str = "white", text_color: str = "black"):
        draw_dummy = ImageDraw.Draw(Image.new('RGB', (1, 1)))

        # Split text into lines that fit the banner width
        max_text_width = width - 2 * padding
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            # Measure width of the test line
            if hasattr(draw_dummy, "textbbox"):
                bbox = draw_dummy.textbbox((0, 0), test_line, font=font)
                line_w = bbox[2] - bbox[0]
                line_h = bbox[3] - bbox[1]
            else:
                try:
                    line_w, line_h = draw_dummy.textsize(test_line, font=font)
                except AttributeError:
                    bbox = font.getbbox(test_line)
                    line_w = bbox[2] - bbox[0]
                    line_h = bbox[3] - bbox[1]
            if line_w <= max_text_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Determine banner height based on number of lines
        text_height = line_h  # height of one line
        banner_height = len(lines) * text_height + (len(lines) + 1) * padding
        banner = Image.new('RGB', (width, banner_height), color=bg_color)
        draw = ImageDraw.Draw(banner)

        y = padding
        for line in lines:
            if hasattr(draw_dummy, "textbbox"):
                bbox = draw_dummy.textbbox((0, 0), line, font=font)
                text_w = bbox[2] - bbox[0]
            else:
                try:
                    text_w, _ = draw_dummy.textsize(line, font=font)
                except AttributeError:
                    bbox = font.getbbox(line)
                    text_w = bbox[2] - bbox[0]
            draw.text(((width - text_w) // 2, y), line, fill=text_color, font=font)
            y += text_height + padding

        return banner

    # Prepare font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    round_keys = sorted(rounds_output.keys(), key=lambda x: int(x))
    per_round_images = []

    for idx, key in enumerate(round_keys):
        round_data = rounds_output[key]
        pose_img = round_data["pose_img"]
        if args.apply_smaller_pose > 1:
            pose_img = pose_img.resize((pose_img.width * args.apply_smaller_pose, pose_img.height * args.apply_smaller_pose), Image.Resampling.LANCZOS)
        real_img = round_data["real_img"]

        # Ensure both images are RGB PIL Images of same height
        if pose_img.mode != 'RGB':
            pose_img = pose_img.convert('RGB')
        if real_img.mode != 'RGB':
            real_img = real_img.convert('RGB')

        concat_width = pose_img.width + real_img.width
        concat_height = max(pose_img.height, real_img.height)
        concat_img = Image.new('RGB', (concat_width, concat_height), color='white')
        concat_img.paste(pose_img, (0, 0))
        concat_img.paste(real_img, (pose_img.width, 0))

        title_banner = _create_text_banner(f"Round {key}", concat_width, font)
        prompt_text = prompts[idx] if idx < len(prompts) else ""
        prompt_banner = _create_text_banner(prompt_text, concat_width, font)

        total_h = title_banner.height + concat_img.height + prompt_banner.height
        round_img = Image.new('RGB', (concat_width, total_h), color='white')
        y_offset = 0
        round_img.paste(title_banner, (0, y_offset)); y_offset += title_banner.height
        round_img.paste(concat_img, (0, y_offset)); y_offset += concat_img.height
        round_img.paste(prompt_banner, (0, y_offset))

        per_round_images.append(round_img)

    # Determine final composite dimensions
    final_width = max(img.width for img in per_round_images)
    general_banner = _create_text_banner(general_prompt, final_width, font)
    final_height = general_banner.height + sum(img.height for img in per_round_images)

    final_img = Image.new('RGB', (final_width, final_height), color='white')
    y_offset = 0
    final_img.paste(general_banner, (0, y_offset)); y_offset += general_banner.height
    for img in per_round_images:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height

    # Log the image to wandb if available and on rank 0
    if wandbrun is not None and rank == 0:
        wandbrun.log({f"sample/{image_name}": wandb.Image(final_img, caption=general_prompt)}, step=global_step)

    # Optionally, save locally for inspection
    save_path = os.path.join("output", "samples", f"sample_{global_step}.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    final_img.save(save_path, format="JPEG", quality=95)

    del clip, t5, vae
    dit.to(f"cuda:{local_gpu}")


def get_train_noise_schedule(timesteps, time_key, ratio, batch_size):
    if time_key == "independent":
        t_indices = [random.randint(0, 999) for _ in range(batch_size)]
        t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")

        t_indice_pose = [random.randint(0, 999) for _ in range(batch_size)]
        t_pose = torch.tensor([timesteps[idx] for idx in t_indice_pose], device="cuda")

    elif time_key == "independent_lowpose":
        t_indices = [random.randint(0, 999) for _ in range(batch_size)]
        t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")
        k = random.random()
        if k < ratio:
            t_indice_pose = [random.randint(0, 999) for _ in range(batch_size)]
            t_pose = torch.tensor([timesteps[idx] for idx in t_indice_pose], device="cuda")
        else:
            t_pose_indices = [random.randint(max(600, idx), 999) for idx in t_indices]
            t_pose = torch.tensor([timesteps[idx] for idx in t_pose_indices], device="cuda")

    elif time_key == "independent_highimg":
        t_indice_pose = [random.randint(0, 999) for _ in range(batch_size)]
        t_pose = torch.tensor([timesteps[idx] for idx in t_indice_pose], device="cuda")
        k = random.random()
        if k < ratio:
            t_indices = [random.randint(0, 999) for _ in range(batch_size)]
            t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")
        else:
            t_indices = [random.randint(0, min(idx, 500)) for idx in t_indice_pose]
            t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")

    elif time_key == "same":
        t_indices = [random.randint(0, 999) for _ in range(batch_size)]
        t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")
        t_pose = torch.clone(t)

    elif time_key == "same_lowpose":
        t_indices = [random.randint(0, 999) for _ in range(batch_size)]
        t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")
        k = random.random()
        if k < ratio:
            t_pose = torch.clone(t)
        else:
            t_pose_indices = [random.randint(max(600, idx), 999) for idx in t_indices]
            t_pose = torch.tensor([timesteps[idx] for idx in t_pose_indices], device="cuda")
    elif time_key == "same_highimg":
        t_indice_pose = [random.randint(0, 999) for _ in range(batch_size)]
        t_pose = torch.tensor([timesteps[idx] for idx in t_indice_pose], device="cuda")
        k = random.random()
        if k < ratio:
            t = torch.clone(t_pose)
        else:
            t_indices = [random.randint(0, min(idx, 500)) for idx in t_indice_pose]
            t = torch.tensor([timesteps[idx] for idx in t_indices], device="cuda")
    else:
        raise ValueError(f"Invalid time_key: {time_key}")

    return t, t_pose

def get_flow_data(x_1, t):
    x_0 = torch.randn_like(x_1).to("cuda")
    x_t = (1 - t[:, None, None]) * x_1 + t[:, None, None] * x_0
    u = x_0 - x_1
    return x_t, u


def create_attention_mask(batch_size: int, text_seqlen: int, cond_seqlen: int, pose_seqlen: int, real_seqlen: int) -> torch.Tensor:
    """
    Generates an attention mask for a concatenated sequence.

    The sequence is structured as:
    [text_sequence, condition_sequence, pose_sequence, real_sequence]

    The masking rule applied is:
    - The 'real_sequence' cannot attend to (cannot 'see') the 'condition_sequence'.
    - All other parts of the sequence can fully attend to each other.

    Args:
        text_seqlen (int): The length of the text sequence.
        cond_seqlen (int): The length of the condition sequence.
        pose_seqlen (int): The length of the pose sequence.
        real_seqlen (int): The length of the real sequence.

    Returns:
        torch.Tensor: A 2D PyTorch tensor representing the attention mask.
                    The shape is [total_sequence_length, total_sequence_length].
                    Values are 0.0 for allowed attention and -1e9 for masked attention.
                    This mask can be directly added to attention logits before softmax.
    """
    # Calculate the total length of the combined sequence
    total_len = text_seqlen + cond_seqlen + pose_seqlen + real_seqlen

    # Initialize the mask with zeros using PyTorch.
    # Zeros indicate full attention (no masking).
    # We use float32 because these masks are typically added to attention scores.
    mask = torch.zeros((batch_size, total_len, total_len), dtype=torch.float32)

    # Define the start and end indices for each segment within the total sequence.
    # These indices are crucial for correctly slicing the mask.
    # text_sequence occupies indices [0, text_seqlen - 1]
    # condition_sequence occupies indices [text_seqlen, text_seqlen + cond_seqlen - 1]
    # pose_sequence occupies indices [text_seqlen + cond_seqlen, text_seqlen + cond_seqlen + pose_seqlen - 1]
    # real_sequence occupies indices [text_seqlen + cond_seqlen + pose_seqlen, total_len - 1]

    # Calculate the start and end indices for the 'condition_sequence' keys
    # (the part that 'real_sequence' queries should not see).
    cond_keys_start = text_seqlen
    cond_keys_end = text_seqlen + cond_seqlen

    # Calculate the start and end indices for the 'real_sequence' queries
    # (the part that should not see 'condition_sequence' keys).
    real_queries_start = text_seqlen + cond_seqlen + pose_seqlen
    real_queries_end = total_len

    # Apply the masking rule: real_sequence queries cannot see condition_sequence keys.
    # In an attention mask M[i, j], 'i' is the query index and 'j' is the key index.
    # Setting a value to -1e9 (a very small number) effectively makes its
    # softmax probability zero after exponentiation, thus masking it out.
    # PyTorch tensors support direct slicing and assignment, no need for numpy conversion.
    mask[:, real_queries_start:real_queries_end, cond_keys_start:cond_keys_end] = -1e9

    return mask


def train_one_step(dit, batch, args, timesteps, t5=None, clip=None):
    # dataset_source, example_type, \
    #     x_1_condition, x_1_condition_c, x_1_condition_h, x_1_condition_w, \
    #     x_1_pose, x_1_pose_c, x_1_pose_h, x_1_pose_w, \
    #     x_1_real, x_1_real_c, x_1_real_h, x_1_real_w, \
    #     text_caption, general_scene_caption, \
    #     is_condition_pose_real_signal, is_pose_to_generate_real_signal, is_real_image_real_signal, \
    #     bounding_boxes_in_order = batch # when customize_img_ids is true, bounding_boxes_in_order is not None

    dataset_source = batch['dataset_source']
    example_type = batch['example_type']

    x_1_condition = batch['condition_pose_image']
    x_1_condition_c = batch['condition_pose_image_c']
    x_1_condition_h = batch['condition_pose_image_h']
    x_1_condition_w = batch['condition_pose_image_w']

    x_1_pose = batch['pose_to_generate_image']
    x_1_pose_c = batch['pose_to_generate_image_c']
    x_1_pose_h = batch['pose_to_generate_image_h']
    x_1_pose_w = batch['pose_to_generate_image_w']

    x_1_real = batch['real_image']
    x_1_real_c = batch['real_image_c']
    x_1_real_h = batch['real_image_h']
    x_1_real_w = batch['real_image_w']


    text_caption = batch['text_caption']
    general_scene_caption = batch['general_scene_caption']

    is_condition_pose_real_signal = batch['is_condition_pose_real_signal']
    is_pose_to_generate_real_signal = batch['is_pose_to_generate_real_signal']
    is_real_image_real_signal = batch['is_real_image_real_signal']

    bounding_boxes_in_order = batch['bounding_boxes_in_order'] # if customize_ids is False, this is None

    local_gpu = torch.cuda.current_device()
    loss_pose_weight = batch['loss_pose_weight'].to(f"cuda:{local_gpu}") # [bs, seqlen]
    x_1_condition = x_1_condition.to(f"cuda:{local_gpu}") # [bs, seqlen, dim]
    x_1_pose = x_1_pose.to(f"cuda:{local_gpu}") # [bs, seqlen, dim]
    x_1_real = x_1_real.to(f"cuda:{local_gpu}") # [bs, seqlen, dim]
    is_real_image_real_signal = is_real_image_real_signal.to(f"cuda:{local_gpu}")


    if args.get('attn_mask_type', None) is None:
        ar_attn_mask = None
    elif args.attn_mask_type == "mask_condition_for_real":
        ar_attn_mask = create_attention_mask(x_1_condition.shape[0], 512, x_1_condition.shape[1], x_1_pose.shape[1], x_1_real.shape[1])
        ar_attn_mask = ar_attn_mask.to(f"cuda:{local_gpu}")
    else:
        raise ValueError(f"Invalid attn_mask_type: {args.attn_mask_type}")


    B = x_1_condition.shape[0] # batch size
    with torch.no_grad():
        inp = prepare_dual_train_ar(t5=t5,
                                    clip=clip,
                                    bs=B,
                                    h_condition=x_1_condition_h,
                                    w_condition=x_1_condition_w,
                                    # pose
                                    h_pose=x_1_pose_h,
                                    w_pose=x_1_pose_w,
                                    # real
                                    h_real=x_1_real_h,
                                    w_real=x_1_real_w,
                                    device=f"cuda:{local_gpu}",
                                    prompt=text_caption,
                                    clip_prompt=general_scene_caption,
                                    pose_id_value=args.pose_id_value,
                                    # TODO always not add h now!
                                    add_pose_clip=args.get('add_pose_clip', False),
                                    bounding_boxes_in_order=bounding_boxes_in_order,
                                    customize_img_ids=args.customize_img_ids,
                                    customize_txt_ids=args.customize_txt_ids,
                                    generate_img_ids_type=args.generate_img_ids_type,
                                    apply_smaller_pose=args.apply_smaller_pose
                                    )
        # patchify is done in dataset.py !

        # TODO: no inner guidance code now

    t_real, t_pose = get_train_noise_schedule(timesteps, args.time_key, args.ratio, B)

    t_cond_indices = [999 for _ in range(B)]
    t_cond = torch.tensor([timesteps[idx] for idx in t_cond_indices], device="cuda")

    x_t_real, u_real = get_flow_data(x_1_real, t_real) # if x_1_real is all black -> zero -> just scaled noise (and no loss for that, u_real not used) TODO check!
    x_t_pose, u_pose = get_flow_data(x_1_pose, t_pose)
    # condition no need to add noise

    x_t = torch.cat([x_1_condition, x_t_pose, x_t_real], dim=1) # pose first TODO check triple lora layer logic
    visual_ids = torch.cat([inp['img_ids_condition'], inp['img_ids_pose'], inp['img_ids_real']], dim=1) # pose first

    guidance_vec = torch.full((x_t.shape[0],), args.guidance, device=x_t.device, dtype=x_t.dtype) # TODO what is guidance vector?
    if args.resize_to_square:
        assert x_1_real.shape[1] == args.img_seq_len, "x_1_real.shape[1] must be equal to args.img_seq_len"
        assert x_1_condition.shape[1] == args.cond_seq_len, "x_1_condition.shape[1] must be equal to args.cond_seq_len"


    model_pred = dit(img=x_t, # cat before in model
                    img_ids=visual_ids, # cat before in model
                    txt=inp['txt'],
                    txt_ids=inp['txt_ids'],
                    y=inp['vec'],
                    y_pose=inp['vec_pose'],
                    timesteps=t_real, # now in batch shape!!
                    timesteps_pose=t_pose, ## for double-stream
                    timesteps_cond=t_cond, ## for ar
                    guidance=guidance_vec,
                    image_sequence_len=x_1_real.shape[1],
                    cond_sequence_len=x_1_condition.shape[1],
                    use_attn_mask=args.use_attn_mask, ## for double-stream
                    ar_attn_mask=ar_attn_mask)

    # loss computation
    cond_len = x_1_condition.shape[1]
    pose_len = x_1_pose.shape[1]
    real_len = x_1_real.shape[1]
    # model_pred_cond = model_pred[:, :cond_len, :] condition always no loss!
    model_pred_pose = model_pred[:, cond_len:cond_len+pose_len, :]
    model_pred_real = model_pred[:, cond_len+pose_len:, :]

    # Calculate loss_pose for all samples
    # loss_pose_per_sample = F.mse_loss(model_pred_pose.float(), u_pose.float(), reduction="none").mean(dim=[1,2])
    # loss_pose = loss_pose_per_sample.mean() # Mean over the batch for logging
    # ----- pose loss with weights -----
    pose_mse       = F.mse_loss(model_pred_pose.float(), u_pose.float(), reduction="none") # [bs, seqlen, dim]
    w_pose = loss_pose_weight # [bs, seqlen]
    pose_mse_lat   = pose_mse.mean(dim=2)   # mean over dim first -- [bs, seqlen]
    loss_pose_per_sample = (pose_mse_lat * w_pose).sum(dim=1) / (w_pose.sum(dim=1) + 1e-6)
    loss_pose      = loss_pose_per_sample.mean() # mean over the batch for logging
    # ----- pose loss with weights -----


    # Only calculate loss_real for samples where real_image_active is True
    # Mask out loss_real for samples where real_image_active is False
    # Use a small epsilon to avoid division by zero if no real images are active in a batch
    loss_real_per_sample = F.mse_loss(model_pred_real.float(), u_real.float(), reduction="none").mean(dim=[1,2])
    loss_real = (loss_real_per_sample * is_real_image_real_signal).sum() / (is_real_image_real_signal.sum() + 1e-6)

    # Combine losses. For pose_only samples, loss_real will be zero or very close to zero if placeholder generated.
    loss = args.lambda_real * loss_real + args.lambda_pose * loss_pose
    loss = loss / args.gradient_accumulation_steps

    return loss, loss_real, loss_pose


def main():
    args = OmegaConf.load(parse_args())
    args.exp_name, args.save_dir = generate_exp_name("ar_triplelora_v0", args, "bs", "rank", "use_dataset", "resize_to_square", "resolution", "customize_img_ids", "customize_txt_ids", "generate_img_ids_type", "background_color", "loss_pose_background_lambda", "double_real_lora", "single_real_lora","real_lr_scale", "attn_mask_type", "apply_smaller_pose", "guidance")

    args.training_width = args.resolution
    args.training_height = args.resolution
    args.sample_width = args.resolution
    args.sample_height = args.resolution
    args.img_seq_len = (args.resolution // 16) * (args.resolution // 16) #  TODO check here is 1024 in original repo
    args.cond_seq_len = (args.resolution // 16) * (args.resolution // 16) # TODO check here is 1024 in original repo
    save_dir = Path.cwd() / args.save_dir / args.exp_name
    os.makedirs(save_dir, exist_ok=True)
    # save configs
    with open(save_dir / "config.yaml", "w") as f:
        OmegaConf.save(config=args, f=f)

    # save programe file
    with open(save_dir / "program.py", "w") as f:
        f.write(open(__file__).read())

    rank = dist.get_rank()
    if args.use_wandb:
        wandb_run = setup_wandb(args, rank)
    logging.info("***** Preparing model *****")
    local_gpu = torch.cuda.current_device()

    t5 = load_t5(f"cuda:{local_gpu}", max_length=512)
    clip = load_clip(f"cuda:{local_gpu}")

    # load dit to all rank's cpu: now every rank hold a copy of dit on cpu
    dit = load_flow_model2(args.model_name, device="cpu") # handle gradient checkpointing in fsdp_utils.py
    ##### replace module / add lora #########################################################
    if args.use_lora:
        print("Using triple LoRA version")
        replace_attn_processor_triplelora_ar(dit, args) # add lora to transformer_blocks (attn & mlp)
    else:
        print("not using LoRA, finetuning all parameters")

    replace_split_head(dit, args) # split head for img_in and final_layer

    ###### set trainable parameters ############################################################
    trainable_names = args.trainable_names # ['img_in', 'final_layer']
    if args.use_lora:
        trainable_names.append('_lora') # attn_lora, proj_lora, mod_lora
        disable_grad(dit, trainable_names) # dit.train() inside disable_grad()
    else:
        dit.train() # train all parameters

    dit.to(torch.bfloat16)
    ##### FSDP setup #########################################################################
    logging.info("***** FSDP setup *****")
    dit, optimizer, global_step = setup_model(dit, args) # TODO will need to update parameter group lr before every optimizer step

    ###### prepare dataloader ############################################################
    logging.info("***** Prepare dataloader *****")
    if MultiHumanDataset is None:
        raise ImportError(
            "Training requires `image_datasets.combined_dataset_ar_prepared.MultiHumanDataset`, "
            "but the `image_datasets` package is not present in this checkout. Restore the "
            "dataset loader before launching training."
        ) from _DATASET_IMPORT_ERROR
    train_dataset = MultiHumanDataset(controlnet_dataset_hf_repo_id=args.controlnet_dataset_hf_repo_id,vae_model_name=args.model_name, vae_offload=args.offload_vae, use_dataset=args.use_dataset, training_width=args.training_width, training_height=args.training_height,
                                     customize_txt_ids=args.customize_txt_ids, customize_img_ids=args.customize_img_ids,
                                     resize_to_square=args.resize_to_square,
                                     background_color=args.background_color,
                                     loss_pose_background_lambda=args.loss_pose_background_lambda,
                                     apply_smaller_pose=args.apply_smaller_pose) # TODO add args!

    def custom_collate_fn(batch):
        # Unpack the batch into separate lists for each returned item from __getitem__
        (
            dataset_sources,
            example_types,
            condition_pose_images, condition_pose_image_cs, condition_pose_image_hs, condition_pose_image_ws,
            pose_to_generate_images, pose_to_generate_image_cs, pose_to_generate_image_hs, pose_to_generate_image_ws,
            real_images, real_image_cs, real_image_hs, real_image_ws,
            text_captions_raw, # This can be str or list[str]
            general_scene_captions,
            is_condition_pose_real_signals, is_pose_to_generate_real_signals, is_real_image_real_signals,
            bounding_boxes_in_order_raw, loss_pose_weight_raw # This can be list[list[int]] or None
        ) = zip(*batch) # `zip(*batch)` unpacks the list of tuples into tuples of lists

        # --- Batching Fixed-Size Tensors ---
        batched_data = {
            'dataset_source': list(dataset_sources), # Keep as list of strings
            'example_type': list(example_types),     # Keep as list of strings

            'condition_pose_image': torch.stack(condition_pose_images),
            'condition_pose_image_c': torch.tensor(condition_pose_image_cs, dtype=torch.long),
            'condition_pose_image_h': torch.tensor(condition_pose_image_hs, dtype=torch.long),
            'condition_pose_image_w': torch.tensor(condition_pose_image_ws, dtype=torch.long),

            'pose_to_generate_image': torch.stack(pose_to_generate_images),
            'pose_to_generate_image_c': torch.tensor(pose_to_generate_image_cs, dtype=torch.long),
            'pose_to_generate_image_h': torch.tensor(pose_to_generate_image_hs, dtype=torch.long),
            'pose_to_generate_image_w': torch.tensor(pose_to_generate_image_ws, dtype=torch.long),
            'loss_pose_weight': torch.stack(loss_pose_weight_raw),

            'real_image': torch.stack(real_images),
            'real_image_c': torch.tensor(real_image_cs, dtype=torch.long),
            'real_image_h': torch.tensor(real_image_hs, dtype=torch.long),
            'real_image_w': torch.tensor(real_image_ws, dtype=torch.long),

            'general_scene_caption': list(general_scene_captions), # Keep as list of strings

            'is_condition_pose_real_signal': torch.tensor(is_condition_pose_real_signals, dtype=torch.bool),
            'is_pose_to_generate_real_signal': torch.tensor(is_pose_to_generate_real_signals, dtype=torch.bool),
            'is_real_image_real_signal': torch.tensor(is_real_image_real_signals, dtype=torch.bool),
        }

        # --- Handle `text_caption` (can be string or list of strings) ---
        # Check if any sample in the batch has a list of captions (customize_txt_ids=True)
        if any(isinstance(tc, list) for tc in text_captions_raw):
            # If so, assume all should be treated as lists and pad them
            max_caption_list_len = max(len(tc) if isinstance(tc, list) else 1 for tc in text_captions_raw)

            padded_text_captions = []
            for tc in text_captions_raw:
                if isinstance(tc, str): # If it's a single string, wrap it in a list
                    tc = [tc]
                padding_needed = max_caption_list_len - len(tc)
                padded_text_captions.append(tc + [""] * padding_needed)
            batched_data['text_caption'] = padded_text_captions # This will be a list of lists of strings
        else:
            # If all are single strings, just keep them as a list of strings
            batched_data['text_caption'] = list(text_captions_raw) # This will be a list of strings


        # --- Handle `bounding_boxes_in_order` (can be list[list[int]] or None) ---
        # Check if any sample in the batch has bounding boxes (customize_img_ids=True)
        if any(bboxes is not None for bboxes in bounding_boxes_in_order_raw):
            # Filter out None values to find max length among actual bbox lists
            actual_bbox_lists = [bboxes for bboxes in bounding_boxes_in_order_raw if bboxes is not None]
            if actual_bbox_lists: # Ensure there's at least one non-None bbox list
                max_bboxes_in_batch = max(len(bboxes) for bboxes in actual_bbox_lists)
            else: # All were None, handle this case by returning None for bboxes
                batched_data['bounding_boxes_in_order'] = None
                return batched_data

            padded_bboxes_batch = []
            for bboxes_for_one_image in bounding_boxes_in_order_raw:
                if bboxes_for_one_image is None:
                    # If a sample had None, pad it to max_bboxes_in_batch with [-1,-1,-1,-1]
                    padded_bboxes_for_one_image = [[-1, -1, -1, -1]] * max_bboxes_in_batch
                else:
                    num_bboxes = len(bboxes_for_one_image)
                    padding_needed = max_bboxes_in_batch - num_bboxes
                    padded_bboxes_for_one_image = bboxes_for_one_image + [[-1, -1, -1, -1]] * padding_needed

                padded_bboxes_batch.append(torch.tensor(padded_bboxes_for_one_image, dtype=torch.long))

            batched_data['bounding_boxes_in_order'] = torch.stack(padded_bboxes_batch, dim=0)
        else:
            # If all samples in the batch had None for bounding_boxes_in_order
            batched_data['bounding_boxes_in_order'] = None

        return batched_data

    # if args.customize_img_ids:
    #     train_dataloader = build_dataloader(train_dataset, args, collate_fn=custom_collate_fn)
    # else:
    #     train_dataloader = build_dataloader(train_dataset, args)
    train_dataloader = build_dataloader(train_dataset, args, collate_fn=custom_collate_fn)

    # train_dataloader = build_dataloader(train_dataset, args, collate_fn=custom_collate_fn) # distributed sampler -- dataloader will only see its own rank's slice when epoch starts

    # num_minibatches_per_epoch_per_rank = len(train_dataloader) # call DistributedSampler.__len__ will be len(dataset) / world_size / local_batch_size
    num_steps_per_epoch_per_rank = len(train_dataloader) // args.gradient_accumulation_steps # "step" means the step that optimizer.step() is called
    num_epochs = math.ceil(args.max_train_steps / num_steps_per_epoch_per_rank)
    start_epoch = global_step // num_steps_per_epoch_per_rank
    logging.info(f"Effective batch size: total_bs")
    logging.info(f"Effective steps per epoch: {num_steps_per_epoch_per_rank}")
    logging.info(f"Effective epochs: {num_epochs}")
    ###### sample step once before training start #######################################
    logging.info("***** Sample step once before training start *****")
    # sample_step(dit, args, global_step, wandb_run, rank, offload=args.offload_when_sample)

    logging.info("***** Running training *****")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not rank == 0, # If you're not the local main process, disable the progress bar (no progress bar output).
    )

    ###### prepare all timesteps ##############################################################
    timesteps = get_schedule(
                999,
                (args.training_width // 8) * (args.training_height // 8) // 4, # TODO check here is 1024 in original repo
                shift=True,
            ) # [1.0, 0.999, 0.998, ..., 0.002, 0.001, 0.000] 1000 length-list

    step_start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        train_dataloader.sampler.set_epoch(epoch) # TODO ! important to call this every epoch to make sure the order of the minibatches is the same across epochs!

        for step, batch in enumerate(train_dataloader): # will only enumerate the minibatches of the current rank
            # step_start_time = time.time()
            micro = step % args.gradient_accumulation_steps
            loss, loss_real, loss_pose = train_one_step(dit, batch, args, timesteps, t5, clip)
            track_loss = reduce_loss(loss.detach().clone()) # loss.detach().clone() creates a separate, gradient-free copy of the tensor.
            track_loss_real = reduce_loss(loss_real.detach().clone())
            track_loss_pose = reduce_loss(loss_pose.detach().clone())
            loss.backward() # per-rank local loss, TODO need to print a all-reduce loss across all ranks as train_loss

            if micro == args.gradient_accumulation_steps - 1: # this step's grad is synced with all-reduce across all ranks when calling loss.backward()
                # but the synced gradients will be scattered to each rank's local model again, so optimizer.step() on each rank will only update its local model, but using the synced gradients
                # do step and clear grad
                # do lr scheduler manually here
                lr = get_lr(global_step, args)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                grad_norm = FSDP.clip_grad_norm_(dit, args.grad_clip)
                if grad_norm < args.max_norm or global_step < 200:
                    optimizer.step()
                else:
                    logging.info(f"Grad norm {grad_norm} is greater than max norm {args.max_norm}, skipping step")
                optimizer.zero_grad()
                progress_bar.update(1)
                torch.cuda.synchronize() # for accurate timing
                step_end_time = time.time()
                training_time_per_step = step_end_time - step_start_time
                step_start_time = time.time()

                if rank == 0 and global_step % args.wandb_log_every == 0 and args.use_wandb:
                    wandb_run.log({
                            "track_loss": track_loss,
                            "track_loss_real": track_loss_real,
                            "track_loss_pose": track_loss_pose,
                            "rank0_loss": loss.detach().clone().item(), # loss on each process,
                            "rank0_loss_real": loss_real.detach().clone().item(),
                            "rank0_loss_pose": loss_pose.detach().clone().item(),
                            "synched_grad_norm": float(grad_norm),
                            "lr": float(lr),
                            "time_per_global_step": training_time_per_step,
                            "Epoch": epoch
                        }, step=global_step)
                global_step += 1

                if global_step % args.sample_every == 0: # let all ranks "see" this function, but only rank0 will log inside this function!
                    sample_step(dit, args, global_step, wandb_run, rank, offload=args.offload_when_sample)
                    #pass

                if global_step % args.save_checkpoint_every == 0:
                    save_model_checkpoint(dit, rank, args, global_step) # let all ranks "see" this function, but only rank0 will get the full state_dict inside the function!
                    save_optimizer_checkpoint(dit, optimizer, rank, args, global_step) # save based on epoch, so may have some overwriting! but it's fine!



if __name__ == "__main__":
    main()




# torchrun --nproc_per_node 8 --master_port <port> train.py --config configs/ar_triplelora.yaml
