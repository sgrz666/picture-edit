import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope, adjust_condition_ids_inplace
import torch.nn.functional as F

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim # 128
        self.theta = theta # theta=10_000,
        self.axes_dim = axes_dim # axes_dim=[16, 56, 56], # 16 + 56 + 56 = 128

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1] # 3d rope
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1) # add a head dimension here [ B, 1, l, d_head/2, 2, 2]


class EmbedNDLearnable(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim # 128
        self.theta = theta # theta=10_000,
        self.axes_dim = axes_dim # axes_dim=[16, 56, 56], # 16 + 56 + 56 = 128

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1] # 3d rope
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)



def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

def build_mask(txt_len, pose_len, img_len, device):
    L = txt_len + pose_len + img_len
    mask = torch.ones((L, L), dtype=torch.bool, device=device)

    # Pose queries = indices in [pose_start, pose_end)
    pose_start = txt_len
    pose_end   = txt_len + pose_len

    # Image keys   = indices in [img_start, img_end)
    img_start  = pose_end
    img_end    = pose_end + img_len

    # We want: "pose queries do NOT see image keys".
    # So for rows in [pose_start, pose_end), for cols in [img_start, img_end), set to False
    mask[pose_start:pose_end, img_start:img_end] = False

    # Everything else remains True
    return mask


def build_mask_batch(txt_len: int, pose_len: int, img_len: int, device: torch.device, real_image_active: Tensor) -> Tensor:
    """
    Builds a batch-specific attention mask.

    For samples where real_image_active is False (pose_only data):
      - Pose tokens are prevented from attending to image tokens.
      - Other attention pathways (e.g., text to image, image to text/pose) are NOT explicitly masked by this function.
        If you need to completely block real image from interacting when real_image_active is False,
        further modifications might be needed depending on your model's attention structure.
        However, the current goal is "pose sequence do not see real".

    For samples where real_image_active is True (original data):
      - No special masking is applied; all tokens can attend to each other.

    Args:
        txt_len (int): Length of text tokens.
        pose_len (int): Length of pose tokens.
        img_len (int): Length of real image tokens.
        device (torch.device): Device for the mask.
        real_image_active (Tensor): Boolean tensor of shape (B,) indicating
                                    if the real image stream is active for each sample.
    Returns:
        Tensor: Batch attention mask of shape (B, L, L) where L = txt_len + pose_len + img_len.
                `False` indicates masked (attention blocked), `True` indicates unmasked (attention allowed).
    """
    batch_size = real_image_active.shape[0]
    L = txt_len + pose_len + img_len # Total sequence length for attention

    # Initialize the final batch mask with all True (allowing all attention)
    # Shape: (Batch_Size, Total_Sequence_Length, Total_Sequence_Length)
    final_batch_mask = torch.ones((batch_size, L, L), dtype=torch.bool, device=device)

    # --- Define the specific region to be masked for 'pose_only' samples ---
    # This region corresponds to "pose queries looking at image keys".

    # Pose queries indices (rows in the attention matrix)
    pose_start = txt_len
    pose_end   = txt_len + pose_len

    # Image keys indices (columns in the attention matrix)
    img_start  = pose_end
    # img_end is implicitly img_start + img_len for slicing

    # Create a 2D mask (L, L) that is False only in the pose-to-image region.
    # This `restricted_mask_2d` represents the mask for a *single* 'pose_only' sample.
    restricted_mask_2d = torch.ones((L, L), dtype=torch.bool, device=device)
    # Set the pose-to-image block to False (mask attention)
    restricted_mask_2d[pose_start:pose_end, img_start:(img_start + img_len)] = False

    # --- Apply this 2D restricted mask conditionally to the batch mask ---
    # `~real_image_active` is a boolean tensor of shape (B,)
    # It will be True for samples where `real_image_active` is False (i.e., 'pose_only' samples).
    # When used for indexing `final_batch_mask[~real_image_active]`, it selects only the
    # mask slices corresponding to the 'pose_only' samples in the batch.

    # For these selected 'pose_only' slices, we assign the `restricted_mask_2d`.
    final_batch_mask[~real_image_active] = restricted_mask_2d

    return final_batch_mask


class DoubleStreamBlockLoraProcessorDualLoRA(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, text_lora=False):
        super().__init__()
        # Pose (P) and Real/main (R) LoRAs for image
        self.qkv_lora1_p = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1_p = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora1_r = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1_r = LoRALinearLayer(dim, dim, rank, network_alpha)

        # Text side can remain single
        self.text_lora = text_lora
        if self.text_lora:
            self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)

        self.lora_weight = lora_weight
        self.use_attn_mask = False  # optional, as in your original code


    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        Sequence order is [txt, pose, image].
        'img' is [pose, image], so we ALWAYS split by:
          pose_len = img.shape[1] - l1
          where l1 = attention_kwargs["image_sequence_len"]
        'vec_pose' is also guaranteed to be present.
        """
        # Unpack what we need
        vec_pose   = attention_kwargs["vec_pose"]
        l1         = attention_kwargs["image_sequence_len"]  # main image length
        pose_len   = img.shape[1] - l1  # how many pose tokens

        # 1) Compute modulations from vec (main) and vec_pose for both image & text
        img_mod_main1, img_mod_main2 = attn.img_mod(vec)
        txt_mod_main1, txt_mod_main2 = attn.txt_mod(vec)

        img_mod_pose1, img_mod_pose2 = attn.img_mod(vec_pose)
        txt_mod_pose1, txt_mod_pose2 = attn.txt_mod(vec_pose)

        # 2) Split the image into [pose, main]
        img_pose, img_main = img[:, :pose_len], img[:, pose_len:]

        # 3) Modulate each portion separately
        # Pose portion with pose mods
        img_pose_mod = attn.img_norm1(img_pose)
        img_pose_mod = (1 + img_mod_pose1.scale) * img_pose_mod + img_mod_pose1.shift

        # Main portion with main mods
        img_main_mod = attn.img_norm1(img_main)
        img_main_mod = (1 + img_mod_main1.scale) * img_main_mod + img_mod_main1.shift

        # 4) Text modulation: combine main + pose mod
        txt_mod = attn.txt_norm1(txt)
        combined_scale_txt = txt_mod_main1.scale #+ txt_mod_pose1.scale
        combined_shift_txt = txt_mod_main1.shift #+ txt_mod_pose1.shift
        txt_mod = (1 + combined_scale_txt) * txt_mod + combined_shift_txt

        # 5) QKV for the pose portion (using "pose" LoRA)
        img_pose_qkv = attn.img_attn.qkv(img_pose_mod) \
                       + self.qkv_lora1_p(img_pose_mod) * self.lora_weight
        # shape [B, L_pose, 3*heads*dim], rearrange into [K=3, B, heads, L_pose, dim]
        img_pose_q, img_pose_k, img_pose_v = rearrange(
            img_pose_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads
        )
        # Possibly norm if you want separate calls
        img_pose_q, img_pose_k = attn.img_attn.norm(img_pose_q, img_pose_k, img_pose_v)

        # 6) QKV for the main portion (using "real" LoRA)
        img_main_qkv = attn.img_attn.qkv(img_main_mod) \
                       + self.qkv_lora1_r(img_main_mod) * self.lora_weight
        img_main_q, img_main_k, img_main_v = rearrange(
            img_main_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads
        )
        img_main_q, img_main_k = attn.img_attn.norm(img_main_q, img_main_k, img_main_v)

        # 7) Combine pose+main for the image side
        # So our final image Q, K, V is [pose chunk, main chunk] along L dimension
        img_q = torch.cat([img_pose_q, img_main_q], dim=3)  # cat along the sequence dimension
        img_k = torch.cat([img_pose_k, img_main_k], dim=3)
        img_v = torch.cat([img_pose_v, img_main_v], dim=3)

        # 8) QKV for text (same as your previous single LoRA)
        if self.text_lora:
            txt_qkv = attn.txt_attn.qkv(txt_mod) \
                    + self.qkv_lora2(txt_mod) * self.lora_weight
        else:
            txt_qkv = attn.txt_attn.qkv(txt_mod)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads
        )
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # 9) Final cat of text + image in the sequence dimension
        q = torch.cat([txt_q, img_q], dim=3)
        k = torch.cat([txt_k, img_k], dim=3)
        v = torch.cat([txt_v, img_v], dim=3)

        # Possibly build/use an attention mask if self.use_attn_mask == True
        if self.use_attn_mask:
            # Hard-coded or dynamic build
            attn_mask = build_mask(txt_len=txt.shape[1], pose_len=pose_len, img_len=l1, device=q.device)
        else:
            attn_mask = None

        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask)
        # attn_out shape: [B, L_total, heads*dim] once rearranged
        # We assume your `attention` re-arranges back to [B, L, hidden]

        # 10) Split the attn output: first txt.shape[1] = text, then pose_len + main_len
        txt_attn = attn_out[:, : txt.shape[1]]
        img_attn = attn_out[:, txt.shape[1] : ]
        # Now split the image portion again into pose, main
        img_attn_pose = img_attn[:, :pose_len]
        img_attn_main = img_attn[:, pose_len:]

        #---------------------------------------------------------------------
        # 11) Residual + MLP blocks for the image
        #---------------------------------------------------------------------
        # -- main portion
        img_main = img_main + img_mod_main1.gate * attn.img_attn.proj(img_attn_main) \
                   + img_mod_main1.gate * self.proj_lora1_r(img_attn_main) * self.lora_weight
        img_main = img_main + img_mod_main2.gate * attn.img_mlp(
            (1 + img_mod_main2.scale) * attn.img_norm2(img_main) + img_mod_main2.shift
        )
        # -- pose portion
        img_pose = img_pose + img_mod_pose1.gate * attn.img_attn.proj(img_attn_pose) \
                   + img_mod_pose1.gate * self.proj_lora1_p(img_attn_pose) * self.lora_weight
        img_pose = img_pose + img_mod_pose2.gate * attn.img_mlp(
            (1 + img_mod_pose2.scale) * attn.img_norm2(img_pose) + img_mod_pose2.shift
        )

        #---------------------------------------------------------------------
        # 12) Residual + MLP blocks for the text
        #---------------------------------------------------------------------
        # Combine gates for text
        combined_gate1 = txt_mod_main1.gate #+ txt_mod_pose1.gate
        combined_gate2 = txt_mod_main2.gate #+ txt_mod_pose2.gate
        combined_scale2 = txt_mod_main2.scale #+ txt_mod_pose2.scale
        combined_shift2 = txt_mod_main2.shift #+ txt_mod_pose2.shift

        # Sub-block 1
        if self.text_lora:
            txt = txt + combined_gate1 * attn.txt_attn.proj(txt_attn) \
                    + combined_gate1 * self.proj_lora2(txt_attn) * self.lora_weight
        else:
            txt = txt + combined_gate1 * attn.txt_attn.proj(txt_attn)
        # Sub-block 2
        txt = txt + combined_gate2 * attn.txt_mlp(
            (1 + combined_scale2) * attn.txt_norm2(txt) + combined_shift2
        )

        #---------------------------------------------------------------------
        # 13) Re-combine image in [pose, main] order
        #---------------------------------------------------------------------
        img = torch.cat((img_pose, img_main), dim=1)

        return img, txt


class DoubleStreamBlockLoraProcessorDualAttnMod(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1, text_lora=False):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.text_lora = text_lora
        if self.text_lora:
            self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight
        self.use_attn_mask = False

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        Now the sequence order is assumed to be: [txt, pose, image].
        Therefore 'img' here actually contains [pose, image].
        If vec_pose is not None, we split 'img' into 'pose' then 'main' image.

        image_sequence_len = l1 = number of image tokens in 'img'.
        The remainder (img.shape[1] - l1) are the pose tokens.
        """
        vec_pose = attention_kwargs.get('vec_pose', None)
        l1       = attention_kwargs.get('image_sequence_len', None)
        # l1 is the image sequence length (the "main" portion)

        #---------------------------------------------------------------------
        # 1) Get modulations from vec (always) and optionally from vec_pose
        #---------------------------------------------------------------------
        img_mod1, img_mod2 = attn.img_mod(vec)  # "normal" image mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec)  # "normal" text mod

        img_mod1_pose = img_mod2_pose = None
        txt_mod1_pose = txt_mod2_pose = None
        if vec_pose is not None:
            img_mod1_pose, img_mod2_pose = attn.img_mod(vec_pose)
            txt_mod1_pose, txt_mod2_pose = attn.txt_mod(vec_pose)

        #---------------------------------------------------------------------
        # 2) If vec_pose is None, treat 'img' as all main image (no pose)
        #---------------------------------------------------------------------
        if vec_pose is None:
            img_main = img
            img_pose = None
        else:
            # The new sequence order is [txt, pose, image].
            # 'img' = [pose, image], so we split by 'pose_len' = total - l1
            pose_len = img.shape[1] - l1
            img_pose, img_main = img[:, :pose_len], img[:, pose_len:]
            # pose chunk is first, then the main image chunk is second

        #---------------------------------------------------------------------
        # 3) Modulate the 'pose' chunk (if present) + 'main' chunk
        #---------------------------------------------------------------------
        # Main portion with the "vec" mods
        img_modulated_main = attn.img_norm1(img_main)
        img_modulated_main = (1 + img_mod1.scale) * img_modulated_main + img_mod1.shift

        if img_pose is not None:
            # Pose portion with the "vec_pose" mods
            img_modulated_pose = attn.img_norm1(img_pose)
            img_modulated_pose = (1 + img_mod1_pose.scale) * img_modulated_pose + img_mod1_pose.shift
            # Re-combine in the order [pose, main]
            img_modulated = torch.cat((img_modulated_pose, img_modulated_main), dim=1)
        else:
            img_modulated = img_modulated_main

        #---------------------------------------------------------------------
        # 4) Text modulation (same as before, combine if vec_pose present)
        #---------------------------------------------------------------------
        txt_modulated = attn.txt_norm1(txt)
        if vec_pose is None:
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        else:
            combined_scale = txt_mod1.scale + txt_mod1_pose.scale
            combined_shift = txt_mod1.shift + txt_mod1_pose.shift
            txt_modulated = (1 + combined_scale) * txt_modulated + combined_shift

        #---------------------------------------------------------------------
        # 5) Q, K, V for image
        #---------------------------------------------------------------------
        img_qkv = attn.img_attn.qkv(img_modulated) \
                  + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D",
            K=3, H=attn.num_heads
        )
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        #---------------------------------------------------------------------
        # 6) Q, K, V for text
        #---------------------------------------------------------------------
        if self.text_lora:
            txt_qkv = attn.txt_attn.qkv(txt_modulated) \
                    + self.qkv_lora2(txt_modulated) * self.lora_weight
        else:
            txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D",
            K=3, H=attn.num_heads
        )
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        #---------------------------------------------------------------------
        # 7) Full attention: text + (pose + main)
        #   The 'img_q' now corresponds to [pose, main] if vec_pose is not None,
        #   or just [main] if vec_pose is None.
        #---------------------------------------------------------------------
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        if not self.use_attn_mask:
            attn_mask = None
        else:
            attn_mask = build_mask(txt_len=512, pose_len=1024, img_len=1024, device=q.device) # when sampling, vec_pose is None, so pose_len and img_len are not passed in

        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask)

        # text_attn is the first txt.shape[1] positions
        txt_attn, img_attn = attn_out[:, :txt.shape[1]], attn_out[:, txt.shape[1]:]

        # If we had [pose, main] for the image portion, we must split accordingly
        if vec_pose is None:
            img_attn_main = img_attn
            img_attn_pose = None
        else:
            # pose_len = length of 'img_pose'
            pose_len = img_pose.shape[1]
            img_attn_pose = img_attn[:, :pose_len]
            img_attn_main = img_attn[:, pose_len:]

        #---------------------------------------------------------------------
        # 8) Residual + MLP blocks for the image
        #---------------------------------------------------------------------
        # First, the main portion
        img_main = img_main + img_mod1.gate * attn.img_attn.proj(img_attn_main) \
                   + img_mod1.gate * self.proj_lora1(img_attn_main) * self.lora_weight
        img_main = img_main + img_mod2.gate * attn.img_mlp(
            (1 + img_mod2.scale) * attn.img_norm2(img_main) + img_mod2.shift
        )

        # Then, if we have pose portion, apply pose mods
        if img_pose is not None:
            img_pose = img_pose + img_mod1_pose.gate * attn.img_attn.proj(img_attn_pose) \
                       + img_mod1_pose.gate * self.proj_lora1(img_attn_pose) * self.lora_weight
            img_pose = img_pose + img_mod2_pose.gate * attn.img_mlp(
                (1 + img_mod2_pose.scale) * attn.img_norm2(img_pose) + img_mod2_pose.shift
            )

        #---------------------------------------------------------------------
        # 9) Residual + MLP blocks for the text
        #---------------------------------------------------------------------
        if vec_pose is None:
            # Old logic
            if self.text_lora:
                txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) \
                      + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
            else:
                txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
            txt = txt + txt_mod2.gate * attn.txt_mlp(
                (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
            )
        else:
            combined_gate1 = txt_mod1.gate + txt_mod1_pose.gate
            combined_gate2 = txt_mod2.gate + txt_mod2_pose.gate
            combined_scale2 = txt_mod2.scale + txt_mod2_pose.scale
            combined_shift2 = txt_mod2.shift + txt_mod2_pose.shift

            # Sub-block 1
            if self.text_lora:
                txt = txt + combined_gate1 * attn.txt_attn.proj(txt_attn) \
                      + combined_gate1 * self.proj_lora2(txt_attn) * self.lora_weight
            else:
                txt = txt + combined_gate1 * attn.txt_attn.proj(txt_attn)
            # Sub-block 2
            txt = txt + combined_gate2 * attn.txt_mlp(
                (1 + combined_scale2) * attn.txt_norm2(txt) + combined_shift2
            )

        #---------------------------------------------------------------------
        # 10) Re-combine image in [pose, main] order and return
        #---------------------------------------------------------------------
        if img_pose is not None:
            img = torch.cat((img_pose, img_main), dim=1)
        else:
            img = img_main

        return img, txt



class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        #print("img shape in oubleStreamBlockLoraProcessor", img.shape)
        # Unpack optional arguments
        #print("img in doubleStreamBlockLoraProcessor shape before forward: ", img.shape)

        vec_pose = attention_kwargs.get('vec_pose', None)
        l1 = attention_kwargs.get('image_sequence_len', None)
        #print("l1 in doubleStreamBlockLoraProcessor: ", l1)

        img_mod1, img_mod2 = attn.img_mod(vec) # "normal" image mod
        txt_mod1, txt_mod2 = attn.txt_mod(vec) # "normal" text mod

        # Default: these pose-specific mods are None unless vec_pose is not None
        img_mod1_pose = img_mod2_pose = None
        txt_mod1_pose = txt_mod2_pose = None

        # If we do have a vec_pose, get the pose-specific modulations
        if vec_pose is not None:
            img_mod1_pose, img_mod2_pose = attn.img_mod(vec_pose)
            txt_mod1_pose, txt_mod2_pose = attn.txt_mod(vec_pose)

        # --------------------------------------------------
        # 1) SPLIT IMAGE (if vec_pose is not None) and apply modulations
        # --------------------------------------------------
        if vec_pose is None:
            # No pose vector: keep the old logic for the entire image
            img_main = img
            img_pose = None
        else:
            # Split the image into two parts
            img_main, img_pose = img[:, :l1], img[:, l1:]  # each shape: [bs, l1, dim], [bs, l2, dim]
            #print("img_main in doubleStreamBlockLoraProcessor shape: ", img_main.shape)
            #print("img_pose in doubleStreamBlockLoraProcessor shape: ", img_pose.shape)

        # Modulate img_main with the "normal" (vec) image mods
        img_modulated_main = attn.img_norm1(img_main)
        img_modulated_main = (1 + img_mod1.scale) * img_modulated_main + img_mod1.shift
        # If we have a pose part, apply the pose mod to it
        if img_pose is not None:
            img_modulated_pose = attn.img_norm1(img_pose)
            img_modulated_pose = (1 + img_mod1_pose.scale) * img_modulated_pose + img_mod1_pose.shift
            # Then we combine them again for attention
            img_modulated = torch.cat((img_modulated_main, img_modulated_pose), dim=1)
            #print("img_modulated in doubleStreamBlockLoraProcessor shape: ", img_modulated.shape)
        else:
            img_modulated = img_modulated_main

        # --------------------------------------------------
        # 2) TEXT MODULATION (add the two sets if vec_pose is not None)
        # --------------------------------------------------
        # Always start by normalizing the txt:
        txt_modulated = attn.txt_norm1(txt)

        if vec_pose is None:
            # Old behavior
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        else:
            # Combine the two text mods (vec + vec_pose)
            combined_scale = txt_mod1.scale + txt_mod1_pose.scale
            combined_shift = txt_mod1.shift + txt_mod1_pose.shift
            # combined_gate  = txt_mod1.gate  + txt_mod1_pose.gate
            # (Similarly you could combine other parameters if needed.)

            txt_modulated = (1 + combined_scale) * txt_modulated + combined_shift
            #print("txt_modulated in doubleStreamBlockLoraProcessor shape: ", txt_modulated.shape)
            # We'll use combined_gate later for the residual part

         # --------------------------------------------------
        # 3) Q, K, V for image
        # --------------------------------------------------
        img_qkv = attn.img_attn.qkv(img_modulated) \
                + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)  # optional norm on Q, K
        #print("img_q in doubleStreamBlockLoraProcessor shape: ", img_q.shape)
        #print("img_k in doubleStreamBlockLoraProcessor shape: ", img_k.shape)
        # --------------------------------------------------
        # 4) Q, K, V for txt
        # --------------------------------------------------
        txt_qkv = attn.txt_attn.qkv(txt_modulated) \
                + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)
        #print("txt_q in doubleStreamBlockLoraProcessor shape: ", txt_q.shape)
        #print("txt_k in doubleStreamBlockLoraProcessor shape: ", txt_k.shape)
        # --------------------------------------------------
        # 5) Full attention: text + (image_main + image_pose)
        # --------------------------------------------------
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        #print("q in doubleStreamBlockLoraProcessor shape: ", q.shape)
        #print("k in doubleStreamBlockLoraProcessor shape: ", k.shape)
        #print("v in doubleStreamBlockLoraProcessor shape: ", v.shape)
        attn_out = attention(q, k, v, pe=pe)

        # text_attn is the first txt.shape[1] positions, then the rest is image
        txt_attn, img_attn = attn_out[:, : txt.shape[1]], attn_out[:, txt.shape[1] :]

        # Now split image_attn back into main vs pose if needed
        if vec_pose is None:
            img_attn_main = img_attn
            img_attn_pose = None
        else:
            img_attn_main = img_attn[:, : l1]
            img_attn_pose = img_attn[:, l1 :]
        # --------------------------------------------------
        # 6) Residual + MLP blocks for image
        # --------------------------------------------------
        # For main portion, use img_mod1, img_mod2
        img_main = img_main + img_mod1.gate * attn.img_attn.proj(img_attn_main) \
                + img_mod1.gate * self.proj_lora1(img_attn_main) * self.lora_weight
        img_main = img_main + img_mod2.gate * attn.img_mlp(
            (1 + img_mod2.scale) * attn.img_norm2(img_main) + img_mod2.shift
        )

        # If we have a pose portion, do the same with *pose mods*
        if img_pose is not None:
            img_pose = img_pose + img_mod1_pose.gate * attn.img_attn.proj(img_attn_pose) \
                    + img_mod1_pose.gate * self.proj_lora1(img_attn_pose) * self.lora_weight
            img_pose = img_pose + img_mod2_pose.gate * attn.img_mlp(
                (1 + img_mod2_pose.scale) * attn.img_norm2(img_pose) + img_mod2_pose.shift
            )

        # --------------------------------------------------
        # 7) Residual + MLP blocks for text
        # --------------------------------------------------
        # If we have vec_pose, we use the combined gates from above, otherwise normal ones
        if vec_pose is None:
            # Old logic
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) \
                + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
            txt = txt + txt_mod2.gate * attn.txt_mlp(
                (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
            )
        else:
            combined_gate1 = txt_mod1.gate + txt_mod1_pose.gate
            combined_gate2 = txt_mod2.gate + txt_mod2_pose.gate

            # combined_scale1 = txt_mod1.scale + txt_mod1_pose.scale
            # combined_shift1 = txt_mod1.shift + txt_mod1_pose.shift

            combined_scale2 = txt_mod2.scale + txt_mod2_pose.scale
            combined_shift2 = txt_mod2.shift + txt_mod2_pose.shift

            # First sub-block
            txt = txt + combined_gate1 * attn.txt_attn.proj(txt_attn) \
                + combined_gate1 * self.proj_lora2(txt_attn) * self.lora_weight
            # Second sub-block
            txt = txt + combined_gate2 * attn.txt_mlp(
                (1 + combined_scale2) * attn.txt_norm2(txt) + combined_shift2
            )
        # --------------------------------------------------
        # 8) Re-combine image if needed and return
        # --------------------------------------------------
        if img_pose is not None:
            img = torch.cat((img_main, img_pose), dim=1)
        else:
            img = img_main

        return img, txt


        # # prepare image for attention
        # img_modulated = attn.img_norm1(img)
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        # img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # # prepare txt for attention
        # txt_modulated = attn.txt_norm1(txt)
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        # txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # # run actual attention
        # q = torch.cat((txt_q, img_q), dim=2)
        # k = torch.cat((txt_k, img_k), dim=2)
        # v = torch.cat((txt_v, img_v), dim=2)

        # attn1 = attention(q, k, v, pe=pe)
        # txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # # calculate the img bloks
        # img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        # img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # # calculate the txt bloks
        # txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        # txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        # return img, txt

class IPDoubleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, image_proj, ip_scale=1.0, **attention_kwargs):

        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)


        # IP-adapter processing
        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)

        img = img + ip_scale * ip_attention

        return img, txt

class DoubleStreamBlockProcessor: # transformer block with attentionblock not separatedly write, in one class
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        img: shape [bs, l_img (+ l_pose if vec_pose is not None), dim]
        txt: shape [bs, l_txt, dim]
        vec: modulation vector (required)
        pe:  positional encodings (already sized for txt + possibly 2-part img if vec_pose != None)
        attention_kwargs may contain:
            - 'vec_pose': optional extra modulation vector
            - 'image_sequence_len': length l1 for the 'main' image tokens (if splitting)
        """

        vec_pose = attention_kwargs.get('vec_pose', None)
        # If we do have a pose vector, we also need the length l1 for the main image region
        l1 = attention_kwargs.get('image_sequence_len', None)
        single_stream_attention_mask = attention_kwargs.get('single_stream_attention_mask', None)

        # --------------------------------------------------
        # 1) Extract main modulations from vec
        # --------------------------------------------------
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # Default to no pose mod
        img_mod1_pose = None
        img_mod2_pose = None
        txt_mod1_pose = None
        txt_mod2_pose = None

        # --------------------------------------------------
        # 2) If we have vec_pose, also compute its modulations
        # --------------------------------------------------
        if vec_pose is not None:
            img_mod1_pose, img_mod2_pose = attn.img_mod(vec_pose)
            txt_mod1_pose, txt_mod2_pose = attn.txt_mod(vec_pose)

        # --------------------------------------------------
        # 3) Split the image if vec_pose is provided
        # --------------------------------------------------
        if vec_pose is None:
            # No splitting, everything is as in the original
            img_main = img
            img_pose = None
        else:
            # Split into two chunks: [0 : l1] and [l1 : ]
            img_main, img_pose = img[:, :l1], img[:, l1:]

        # --------------------------------------------------
        # 4) Modulate the image main portion (required) + optional pose portion
        # --------------------------------------------------
        # Main portion
        img_main_modulated = attn.img_norm1(img_main)
        img_main_modulated = (1 + img_mod1.scale) * img_main_modulated + img_mod1.shift

        if img_pose is not None:
            # Pose portion
            img_pose_modulated = attn.img_norm1(img_pose)
            img_pose_modulated = (1 + img_mod1_pose.scale) * img_pose_modulated + img_mod1_pose.shift
            # Combine to feed into attention
            img_modulated = torch.cat([img_main_modulated, img_pose_modulated], dim=1)
        else:
            # No pose → same as original
            img_modulated = img_main_modulated

        # --------------------------------------------------
        # 5) Modulate text (add the two mod sets if vec_pose)
        # --------------------------------------------------
        txt_modulated = attn.txt_norm1(txt)

        if vec_pose is None:
            # Original mod
            txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
            txt_gate1, txt_gate2 = txt_mod1.gate, txt_mod2.gate
            txt_scale2, txt_shift2 = txt_mod2.scale, txt_mod2.shift
        else:
            # Combine (add) the mod outputs from vec + vec_pose
            combined_scale1 = txt_mod1.scale + txt_mod1_pose.scale
            combined_shift1 = txt_mod1.shift + txt_mod1_pose.shift
            combined_gate1  = txt_mod1.gate  + txt_mod1_pose.gate

            combined_scale2 = txt_mod2.scale + txt_mod2_pose.scale
            combined_shift2 = txt_mod2.shift + txt_mod2_pose.shift
            combined_gate2  = txt_mod2.gate  + txt_mod2_pose.gate

            # Apply the combined first mod
            txt_modulated = (1 + combined_scale1) * txt_modulated + combined_shift1
            # Keep these for later residual/MLP gates
            txt_gate1, txt_gate2 = combined_gate1, combined_gate2
            txt_scale2, txt_shift2 = combined_scale2, combined_shift2

        # --------------------------------------------------
        # 6) Prepare Q, K, V for images
        # --------------------------------------------------
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim
        )
        # Often normalizing Q, K
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # --------------------------------------------------
        # 7) Prepare Q, K, V for text
        # --------------------------------------------------
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim
        )
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # --------------------------------------------------
        # 8) Concatenate for multi-modal cross-attention
        # --------------------------------------------------

        # Original: text + single image
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)


        # run the attention
        attn_out = attention(q, k, v, pe=pe, attn_mask=single_stream_attention_mask)

        # text_attn is the first txt.shape[1] positions
        txt_attn = attn_out[:, : txt.shape[1]]
        img_attn = attn_out[:, txt.shape[1] :]

        # If we have a pose part, split it out from the main portion
        if img_pose is not None:
            img_attn_main = img_attn[:, : l1]
            img_attn_pose = img_attn[:, l1 :]
        else:
            img_attn_main = img_attn
            img_attn_pose = None

        # --------------------------------------------------
        # 9) Residual + MLP for image main
        # --------------------------------------------------
        img_main = img_main + img_mod1.gate * attn.img_attn.proj(img_attn_main)
        img_main = img_main + img_mod2.gate * attn.img_mlp(
            (1 + img_mod2.scale) * attn.img_norm2(img_main) + img_mod2.shift
        )

        # If we have a pose portion, do the same
        if img_pose is not None:
            img_pose = img_pose + img_mod1_pose.gate * attn.img_attn.proj(img_attn_pose)
            img_pose = img_pose + img_mod2_pose.gate * attn.img_mlp(
                (1 + img_mod2_pose.scale) * attn.img_norm2(img_pose) + img_mod2_pose.shift
            )

        # --------------------------------------------------
        # 10) Residual + MLP for text
        # --------------------------------------------------
        # If vec_pose is None, we are using txt_mod1 / txt_mod2 gating.
        # If vec_pose is not None, we computed combined gates/scale above.
        txt = txt + txt_gate1 * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_gate2 * attn.txt_mlp(
            (1 + txt_scale2) * attn.txt_norm2(txt) + txt_shift2
        )

        # --------------------------------------------------
        # 11) Re-combine image if needed and return
        # --------------------------------------------------
        if img_pose is not None:
            img = torch.cat([img_main, img_pose], dim=1)
        else:
            img = img_main

        return img, txt
        # img_mod1, img_mod2 = attn.img_mod(vec)
        # txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # # prepare image for attention
        # img_modulated = attn.img_norm1(img)
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        # img_qkv = attn.img_attn.qkv(img_modulated)
        # img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # # prepare txt for attention
        # txt_modulated = attn.txt_norm1(txt)
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        # txt_qkv = attn.txt_attn.qkv(txt_modulated)
        # txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # # run actual attention
        # q = torch.cat((txt_q, img_q), dim=2) # if img_pose is provided, q will be [txt_q, img_q, img_q_pose]
        # k = torch.cat((txt_k, img_k), dim=2)
        # v = torch.cat((txt_v, img_v), dim=2)

        # attn1 = attention(q, k, v, pe=pe) # if img_pose is provided, pe will be [txt_pe, img_pe, img_pose_pe]
        # txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :] # TODO check -- if img_pose is provided, img will be [img, img_pose]

        # # calculate the img bloks
        # img = img + img_mod1.gate * attn.img_attn.proj(img_attn) # x = x + A(norm(x))
        # img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # # calculate the txt bloks
        # txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        # txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        # return img, txt

class DoubleStreamBlockPoseLoraProcessorOneStream(nn.Module):
    """
    A processor that handles:
      - Pose image sequences only (`img` is solely pose)
      - Pose modulation always has an additional LoRA offset
      - Separate LoRA for pose QKV/proj (always active)
      - Optional LoRA for text if text_lora=True
    """
    def __init__(self,
                 dim: int,
                 rank: int = 4,
                 network_alpha: float = None,
                 lora_weight: float = 1.0,
                 text_lora: bool = False): # Removed pose_lora, real_lora as they are no longer relevant
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.use_attn_mask = False # Kept as per original, but removed real_image_active related logic

        # LoRA for pose QKV/proj (always active)
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pose = LoRALinearLayer(dim, dim, rank, network_alpha)

        # LoRA for pose modulation (always active)
        # The "Modulation" block is normally a single nn.Linear(dim -> 6*dim)
        # We'll add a rank-R offset to it for the pose embedding
        self.pose_mod_lora = LoRALinearLayer(dim, 6 * dim, rank, network_alpha)

        # If text_lora=True, we also add LoRA to the text side
        self.text_lora = text_lora
        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora_text = LoRALinearLayer(dim, dim, rank, network_alpha)
        else:
            # Placeholders to keep code simpler; these will not be used if text_lora is False
            self.qkv_lora_text = None
            self.proj_lora_text = None

    def forward(self, attn: nn.Module, img: Tensor, txt: Tensor, pe: Tensor, vec: Tensor, **attention_kwargs) -> tuple[Tensor, Tensor]:
        """
        Forward pass for the single-stream pose LoRA processor.

        Args:
            attn (nn.Module): The `DoubleStreamBlock` instance, containing
                              sub-modules like `img_mod`, `txt_mod`, `img_attn`, `txt_attn`,
                              `img_norm1`, `img_norm2`, `img_mlp`, `txt_norm1`, `txt_norm2`, `txt_mlp`.
                              It also provides `num_heads` and `head_dim`.
            img (Tensor): Input tensor representing only pose image sequences.
                          Shape: `[batch_size, pose_sequence_length, dim]`.
            txt (Tensor): Input tensor representing text embeddings.
                          Shape: `[batch_size, text_sequence_length, dim]`.
            pe (Tensor): Positional embedding for attention.
            vec_pose (Tensor): The embedding for the pose images, used for modulation.
            attention_kwargs (dict): Additional keyword arguments, can include
                                     `text_lora` for dynamic override.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the processed image tensor
                                   (now only pose) and the processed text tensor.

        """
        vec_pose = vec
        # 1) Determine the pose sequence length. `img` now directly corresponds to `pose_img`.
        pose_seqlen = img.shape[1]
        # `real_seqlen` is implicitly 0 as there is no real image stream.

        # `vec_pose` is a direct argument and used for pose modulation.

        # If you want to allow runtime override for text_lora, uncomment the line below:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) `pose_img` is the input `img` itself, no splitting needed.
        pose_img = img  # [batch_size, pose_seqlen, dim]

        # ---------------------------------------------------------------------
        # HELPER: Compute ModulationOut with LoRA offset for Pose.
        # This function encapsulates the logic for applying LoRA to the pose modulation.
        def pose_modulation_forward(mod_block: nn.Module, pose_vector: Tensor) -> tuple[ModulationOut, ModulationOut]:
            """
            Computes the modulation parameters for pose images, including a LoRA offset.
            Args:
                mod_block (nn.Module): The base modulation block (e.g., `attn.img_mod`).
                pose_vector (Tensor): The pose embedding vector (`vec_pose`).
            Returns:
                tuple[ModulationOut, ModulationOut]: Two sets of (shift, scale, gate) parameters.
            """
            # Calculate the base modulation output from the original modulation block
            # print("pose_vector.shape", pose_vector.shape)
            # print("mod_block.lin.weight.shape", mod_block.lin.weight.shape)
            # print("pose_vector dtype", pose_vector.dtype)
            # print("mod_block.lin.weight dtype", mod_block.lin.weight.dtype)
            base = mod_block.lin(nn.functional.silu(pose_vector))  # [batch_size, 6 * dim]

            # Calculate the LoRA offset from the pose modulation LoRA layer
            # The offset is scaled by `self.lora_weight`
            offset = self.pose_mod_lora(nn.functional.silu(pose_vector)) * self.lora_weight

            # Combine the base modulation output with the LoRA offset
            combined = base + offset  # [batch_size, 6 * dim]

            # Reshape and chunk the combined output into two sets of (shift, scale, gate)
            # Each set consists of three components (shift, scale, gate), each of `dim` size.
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # First set: (shift, scale, gate)
                ModulationOut(*chunked[3:])   # Second set: (shift, scale, gate)
            )

        # HELPER: Compute ModulationOut for text.
        # This function applies the text modulation block. LoRA is handled within apply_qkv_with_lora.
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor) -> tuple[ModulationOut, ModulationOut]:
            """
            Computes the modulation parameters for text embeddings.
            Args:
                mod_block (nn.Module): The base text modulation block (e.g., `attn.txt_mod`).
                text_vector (Tensor): The text embedding vector (or `vec_pose` if intended by design).
            Returns:
                tuple[ModulationOut, ModulationOut]: Two sets of (shift, scale, gate) parameters.
            """
            # Call the forward method of the base text modulation block
            return mod_block(text_vector)

        # 3) Gather the modulation parameters for pose and text streams.
        # Pose uses its dedicated LoRA-based modulation (always active).
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)

        # Text uses its modulation block. The original code used `vec_pose` for text modulation;
        # this behavior is maintained as per the request to "keep other things all the same".
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose)

        # ---------------------------------------------------------------------
        # HELPER: Apply QKV projection with optional LoRA offset.
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool) -> Tensor:
            """
            Applies the QKV linear projection to input tokens, optionally adding a LoRA offset.
            Args:
                x (Tensor): Input tokens. Shape: `[batch_size, sequence_length, dim]`.
                attn_module (SelfAttention): The base SelfAttention module (e.g., `attn.img_attn` or `attn.txt_attn`).
                lora_downup (LoRALinearLayer): The LoRA layer for the QKV projection.
                do_lora (bool): Flag indicating whether to apply the LoRA offset.
            Returns:
                Tensor: The combined QKV tensor. Shape: `[batch_size, sequence_length, 3 * dim]`.
            """
            base_qkv = attn_module.qkv(x)  # Compute base QKV from the original linear layer
            if do_lora:
                # If LoRA is active, compute the LoRA offset and scale it
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # Add the offset to the base QKV
            else:
                return base_qkv # Return base QKV if LoRA is not active

        # 4) Apply Pre-Attention Layer Normalization and Shift/Scale for each stream.
        # Pose stream processing:
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift

        # Text stream processing:
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # 5) Compute QKV (Query, Key, Value) for each portion.
        # For pose, LoRA is always applied (do_lora=True).
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, True)
        # For text, LoRA is applied conditionally based on `self.text_lora`.
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)

        # 6) Rearrange the QKV tensors and combine them for the attention operation.
        # Get batch size and sequence lengths for each stream.
        bs, pose_len, _ = pose_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # Helper to split a combined QKV tensor into individual Q, K, V components
        # and reshape them for multi-head attention.
        def split_qkv(x: Tensor, num_heads: int, head_dim: int) -> tuple[Tensor, Tensor, Tensor]:
            """
            Splits a QKV tensor into Query, Key, and Value tensors,
            and rearranges them for multi-head attention.
            Shape transformation: [B, L, 3*H*D] -> [3, B, H, L, D]
            """
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=num_heads, D=head_dim)
            return q, k, v

        # Split QKV for pose and text streams using the `attn` module's head properties.
        pose_q, pose_k, pose_v = split_qkv(pose_qkv, attn.num_heads, attn.head_dim)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv, attn.num_heads, attn.head_dim)

        # Apply pre-attention normalization (if any) to Q and K.
        # This typically involves scaling Q and K before the dot product.
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # Concatenate the Q, K, and V tensors from text and pose streams.
        # The concatenation order is text first, then pose.
        q = torch.cat([txt_q, pose_q], dim=-2)  # Concatenate along the sequence dimension
        k = torch.cat([txt_k, pose_k], dim=-2)
        v = torch.cat([txt_v, pose_v], dim=-2)

        # attn_mask is None
        attn_out = attention(q, k, v, pe=pe)

        # 8) Split the attention output back into text and pose components.
        txt_attn_out  = attn_out[:, :txt_len]
        pose_attn_out = attn_out[:, txt_len: ] # The remainder of the sequence belongs to pose

        # 9) Apply Residual connection + Gating for Pose stream.
        # The LoRA for pose projection (`self.proj_lora_pose`) is always applied.
        pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                            + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight

        # Apply the MLP part with the second modulation for the pose stream.
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 10) Apply Residual connection + Gating for Text stream.
        # LoRA for text projection (`self.proj_lora_text`) is applied conditionally
        # based on `self.text_lora`.
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        # Apply the MLP part with the second modulation for the text stream.
        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 11) The output `img` tensor now solely consists of the processed pose image data.
        img = pose_img  # Shape: [batch_size, pose_len, dim]

        return img, txt


class DoubleStreamBlockPoseLoraProcessorAR(nn.Module):
    """
    A processor that handles:
      - Cond + Pose + Real image sequences (split from `img`)
      - Pose and Cond modulation have an additional LoRA offset
      - Separate LoRA for pose QKV/proj vs. real QKV/proj
      - Optional LoRA for text if text_lora=True
    """
    def __init__(self,
                 dim: int,
                 rank=4,
                 network_alpha=None,
                 lora_weight=1.0,
                 text_lora=False,
                 pose_lora=True,
                 real_lora=False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.use_attn_mask = False
        self.real_image_active = None

        # Separate LoRA for pose/cond QKV/proj (cond and pose share LoRA based on prompt)
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pose = LoRALinearLayer(dim, dim, rank, network_alpha)

        # LoRA for pose/cond modulation (cond and pose share LoRA based on prompt)
        self.pose_mod_lora = LoRALinearLayer(dim, 6 * dim, rank, network_alpha)

        # If text_lora=True, we also add LoRA to the text side
        self.text_lora = text_lora
        self.pose_lora = pose_lora # This now applies to both cond and pose image streams
        self.real_lora = real_lora

        self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)
        # if self.real_lora:
        #     # Separate LoRA for real QKV/proj
        #     self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        #     self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)
        # else:
        #     self.qkv_lora_real = None
        #     self.proj_lora_real = None

        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora_text = LoRALinearLayer(dim, dim, rank, network_alpha)
        else:
            # placeholders to keep code simpler; won't be used
            self.qkv_lora_text = None
            self.proj_lora_text = None

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        attn: the DoubleStreamBlock
          - attn.img_mod -> Modulation block for images
          - attn.txt_mod -> Modulation block for text
          - attn.img_attn -> SelfAttention for images
          - attn.txt_attn -> SelfAttention for text
        img: [bs, (cond_seqlen + pose_seqlen + real_seqlen), dim]
        txt: [bs, 512, dim]
        vec: the 'real' image embedding
        pe: positional embedding for attention (now 3-signal version)
        attention_kwargs includes:
          - cond_sequence_len: the length of the conditioning images
          - image_sequence_len: the length of the real images
          - vec_cond: the embedding for the conditioning images
          - vec_pose: the embedding for the pose images
          - (optionally) text_lora: if you want a dynamic override
        """
        # 1) Parse out the cond, pose and real lengths
        cond_seqlen = attention_kwargs["cond_sequence_len"]
        real_seqlen = attention_kwargs["image_sequence_len"]  # length of real image tokens
        pose_seqlen = img.shape[1] - cond_seqlen - real_seqlen # Total len - cond_len - real_len

        vec_cond = attention_kwargs["vec_cond"]
        vec_pose = attention_kwargs["vec_pose"]

        ar_attn_mask = attention_kwargs["ar_attn_mask"] # can be None

        bbox_id_embeddings = attention_kwargs.get("bbox_id_embeddings", None)
        mockpe = attention_kwargs.get("mockpe", None)
        pe_learnable = attention_kwargs.get("pe_learnable", False)
        batch_shift = attention_kwargs.get("batch_shift", None)
        batch_resize_scale = attention_kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = attention_kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = attention_kwargs.get("learned_shift_scale_ratio", None)
        ori_h = attention_kwargs.get("ori_h", None)
        ori_w = attention_kwargs.get("ori_w", None)


        # If you want to allow runtime override for text_lora, do:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) Split cond, pose tokens from real tokens
        cond_img = img[:, :cond_seqlen]                                   # [bs, cond_seqlen, dim]
        pose_img = img[:, cond_seqlen:cond_seqlen + pose_seqlen]          # [bs, pose_seqlen, dim]
        real_img = img[:, cond_seqlen + pose_seqlen:]                     # [bs, real_seqlen, dim]

        # print("DEBUG -----------------------------------")
        # print("cond_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", cond_img.shape)
        # print("pose_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", pose_img.shape)
        # print("real_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", real_img.shape)
        # print("DEBUG -----------------------------------")

        # ---------------------------------------------------------------------
        # HELPER: compute ModulationOut with LoRA offset for Pose/Cond
        def pose_modulation_forward(mod_block: nn.Module, vector: Tensor):
            """
            This replicates mod_block(vec) but inserts a LoRA offset.
            - Normally: out = mod_block.lin(silu(vec))
            - We'll add pose_mod_lora(silu(vec)) * lora_weight
            Then chunk the result into two ModulationOut sets of (shift, scale, gate).
            This helper is used for both cond and pose streams.
            """
            base = mod_block.lin(nn.functional.silu(vector))  # [bs, 6*dim]
            offset = self.pose_mod_lora(nn.functional.silu(vector)) * self.lora_weight
            combined = base + offset  # [bs, 6*dim]
            # shape to [bs, 1, 6*dim], chunk into 2 sets of 3
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # (shift, scale, gate)
                ModulationOut(*chunked[3:])   # (shift, scale, gate)
            )

        # HELPER: compute ModulationOut for real (no LoRA offset)
        def real_modulation_forward(mod_block: nn.Module, real_vector: Tensor):
            return mod_block(real_vector)  # calls mod_block.forward()

        # HELPER: compute ModulationOut for text, optional LoRA
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor):
            # Normal path: mod_block(text_vector)
            return mod_block(text_vector)

        # 3) Gather the modulation parameters
        # Cond uses separate LoRA-based mod
        cond_mod1, cond_mod2 = pose_modulation_forward(attn.img_mod, vec_cond)
        # Pose uses separate LoRA-based mod
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)
        # Real uses normal mod
        real_mod1, real_mod2 = real_modulation_forward(attn.img_mod, vec)
        # Text uses normal mod (original code used vec_pose for txt_mod)
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose)

        # ---------------------------------------------------------------------
        # HELPER: apply QKV with LoRA offset
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool):
            """
            x: [bs, seq_len, dim] input tokens (after norm & shift/scale).
            attn_module.qkv: the base QKV linear
            lora_downup: the LoRA offset (qkv_lora_pose or qkv_lora_real or qkv_lora_text)
            """
            base_qkv = attn_module.qkv(x)  # [bs, seq_len, 3*dim]
            if do_lora:
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # shape [bs, seq_len, 3*dim]
            else:
                return base_qkv

        # 4) Pre-Attention Norm + Shift/Scale for each stream
        # Cond
        cond_modulated = attn.img_norm1(cond_img)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        # Pose
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift
        # Real
        real_modulated = attn.img_norm1(real_img)
        real_modulated = (1 + real_mod1.scale) * real_modulated + real_mod1.shift
        # Text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # print("DEBUG -----------------------------------")
        # print("cond_modulated.shape", cond_modulated.shape)
        # print("pose_modulated.shape", pose_modulated.shape)
        # print("real_modulated.shape", real_modulated.shape)
        # print("txt_modulated.shape", txt_modulated.shape)
        # print("DEBUG -----------------------------------")

        # 5) QKV for each portion
        # Cond uses pose_lora (as per "cond and pose will gone through exactly same process")
        cond_qkv = apply_qkv_with_lora(cond_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        real_qkv = apply_qkv_with_lora(real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)  # might have no LoRA if text_lora=False

        # 6) Rearrange and combine
        # shape for each qkv is: [bs, seq_len, 3*dim]
        bs, cond_len, _ = cond_qkv.shape
        _, pose_len, _ = pose_qkv.shape
        _, real_len, _ = real_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # print("DEBUG -----------------------------------")
        # print("cond_qkv.shape", cond_qkv.shape)
        # print("pose_qkv.shape", pose_qkv.shape)
        # print("real_qkv.shape", real_qkv.shape)
        # print("txt_qkv.shape", txt_qkv.shape)
        # print("DEBUG -----------------------------------")

        # Split into q, k, v (B L (3 H D)) -> (3, B, H, L, D)
        # using attn.num_heads / attn.head_dim from the block
        def split_qkv(x, attn_module):
            # Pass attn_module to get num_heads and head_dim
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=attn_module.num_heads)
            return q, k, v

        # Split using the respective attention module for num_heads and head_dim
        # Assuming img_attn and txt_attn have these attributes.
        cond_q, cond_k, cond_v = split_qkv(cond_qkv, attn.img_attn)
        pose_q, pose_k, pose_v = split_qkv(pose_qkv, attn.img_attn)
        real_q, real_k, real_v = split_qkv(real_qkv, attn.img_attn)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv,  attn.txt_attn)

        # run the typical "attn.img_attn.norm" or "attn.txt_attn.norm" for each
        # (some implementations do a small pre-softmax normalization)
        cond_q, cond_k = attn.img_attn.norm(cond_q, cond_k, cond_v)
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        real_q, real_k = attn.img_attn.norm(real_q, real_k, real_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # print("DEBUG -----------------------------------")
        # print("cond_q.shape", cond_q.shape)
        # print("pose_q.shape", pose_q.shape)
        # print("real_q.shape", real_q.shape)
        # print("txt_q.shape", txt_q.shape)
        # print("DEBUG -----------------------------------")

        # Now we concatenate all streams: text first, then cond, then pose, then real
        q = torch.cat([txt_q, cond_q, pose_q, real_q], dim=-2)  # along sequence dimension
        k = torch.cat([txt_k, cond_k, pose_k, real_k], dim=-2)
        v = torch.cat([txt_v, cond_v, pose_v, real_v], dim=-2)

        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert mockpe is not None, "mockpe is required when pe_learnable is True"
            assert learned_shift_scale_ratio is not None, "learned_shift_scale_ratio is required when pe_learnable is True"
            mock_shift_scale = mockpe(k, v) # [bs, 3, 1]
            mock_shift_scale = mock_shift_scale.squeeze(-1) # [bs, 3]
            shift_x_learned, shift_y_learned, scale_learned = mock_shift_scale[:, 0], mock_shift_scale[:, 1], mock_shift_scale[:, 2] # [bs]
            shift_learned = torch.stack([shift_x_learned, shift_y_learned], dim=1) # (B, 2)

            use_learned_shift_scale = torch.rand(()) <= learned_shift_scale_ratio  # scalar boolean -- switch once for whole batch
            if use_learned_shift_scale:
                batch_shift_to_use = shift_learned
                batch_resize_scale_to_use = scale_learned
            else:
                batch_shift_to_use = batch_shift
                batch_resize_scale_to_use = batch_resize_scale

            pe = adjust_condition_ids_inplace(batch_ids=pe, batch_multi_person_last_indicator=batch_multi_person_last_indicator, batch_shift=batch_shift_to_use, batch_resize_scale=batch_resize_scale_to_use, ori_h=ori_h, ori_w=ori_w)
            pe = EmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(pe)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 7) Run attention
        # Update build_mask and build_mask_batch arguments
        # if self.real_image_active is None:
        #     if self.use_attn_mask: # when sampling, vec_pose is None
        #         # Assuming hardcoded lengths for mask are txt_len=512, cond_len=1024, pose_len=1024, img_len=1024
        #         # These should ideally be derived from actual sequence lengths if dynamic.
        #         attn_mask = build_mask(txt_len=txt_len, cond_len=cond_len, pose_len=pose_len, img_len=real_len, device=q.device)
        #     else:
        #         attn_mask = None
        # else:
        #     attn_mask = build_mask_batch(txt_len=txt_len, cond_len=cond_len, pose_len=pose_len, img_len=real_len, device=q.device, real_image_active=self.real_image_active)

        # print("DEBUG -----------------------------------")
        # print("q.shape", q.shape)
        # print("k.shape", k.shape)
        # print("v.shape", v.shape)
        # print("pe.shape", pe.shape)
        # print("DEBUG -----------------------------------")

        attn_out = attention(q, k, v, pe=pe, attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings)  # shape [bs, (txt_len+cond_len+pose_len+real_len), dim]

        # 8) Split the attention output back to text, cond, pose, real
        txt_attn_out  = attn_out[:, :txt_len]
        cond_attn_out = attn_out[:, txt_len : txt_len + cond_len]
        pose_attn_out = attn_out[:, txt_len + cond_len : txt_len + cond_len + pose_len]
        real_attn_out = attn_out[:, txt_len + cond_len + pose_len:]

        # 9) Residual + gating for Cond (shares pose_lora)
        if self.pose_lora: # This pose_lora flag now applies to cond too
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out) \
                                + cond_mod1.gate * self.proj_lora_pose(cond_attn_out) * self.lora_weight
        else:
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out)
        # MLP part with second modulation
        cond_img = cond_img + cond_mod2.gate * attn.img_mlp(
            (1 + cond_mod2.scale) * attn.img_norm2(cond_img) + cond_mod2.shift
        )


        # 10) Residual + gating for Pose (shares pose_lora)
        if self.pose_lora:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                                + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight
        else:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out)
        # MLP part with second modulation
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 11) Residual + gating for Real
        if self.real_lora:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out) \
                                + real_mod1.gate * self.proj_lora_real(real_attn_out) * self.lora_weight
        else:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out)
        # MLP part with second modulation
        real_img = real_img + real_mod2.gate * attn.img_mlp(
            (1 + real_mod2.scale) * attn.img_norm2(real_img) + real_mod2.shift
        )

        # 12) Residual + gating for Text
        # If text_lora=False, no offset; if text_lora=True, use self.proj_lora_text
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 13) Re-combine cond, pose and real into a single `img` again
        img = torch.cat([cond_img, pose_img, real_img], dim=1)  # [bs, cond_len + pose_len + real_len, dim]

        return img, txt


class DoubleStreamBlockPoseLoraProcessorAR_version2(nn.Module):
    """
    A processor that handles:
      - Cond + Pose + Real image sequences (split from `img`)
      - Pose and Cond modulation have an additional LoRA offset
      - Separate LoRA for pose QKV/proj vs. real QKV/proj
      - Optional LoRA for text if text_lora=True
    """
    def __init__(self,
                 dim: int,
                 rank=4,
                 network_alpha=None,
                 lora_weight=1.0,
                 text_lora=False,
                 pose_lora=True,
                 pre_real_lora=False,
                 real_lora=False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.use_attn_mask = False
        self.real_image_active = None

        # Separate LoRA for pose/cond QKV/proj (cond and pose share LoRA based on prompt)
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pose = LoRALinearLayer(dim, dim, rank, network_alpha)

        # LoRA for pose/cond modulation (cond and pose share LoRA based on prompt)
        self.pose_mod_lora = LoRALinearLayer(dim, 6 * dim, rank, network_alpha)

        # If text_lora=True, we also add LoRA to the text side
        self.text_lora = text_lora
        self.pose_lora = pose_lora # This now applies to both cond and pose image streams
        self.real_lora = real_lora
        self.pre_real_lora = pre_real_lora

        self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)

        self.qkv_lora_pre_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pre_real = LoRALinearLayer(dim, dim, rank, network_alpha)

        # if self.real_lora:
        #     # Separate LoRA for real QKV/proj
        #     self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        #     self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)
        # else:
        #     self.qkv_lora_real = None
        #     self.proj_lora_real = None

        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora_text = LoRALinearLayer(dim, dim, rank, network_alpha)
        else:
            # placeholders to keep code simpler; won't be used
            self.qkv_lora_text = None
            self.proj_lora_text = None

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        attn: the DoubleStreamBlock
          - attn.img_mod -> Modulation block for images
          - attn.txt_mod -> Modulation block for text
          - attn.img_attn -> SelfAttention for images
          - attn.txt_attn -> SelfAttention for text
        img: [bs, (cond_seqlen + pose_seqlen + real_seqlen), dim]
        txt: [bs, 512, dim]
        vec: the 'real' image embedding
        pe: positional embedding for attention (now 3-signal version)
        attention_kwargs includes:
          - cond_sequence_len: the length of the conditioning images
          - image_sequence_len: the length of the real images
          - vec_cond: the embedding for the conditioning images
          - vec_pose: the embedding for the pose images
          - (optionally) text_lora: if you want a dynamic override
        """
        # 1) Parse out the cond, pose and real lengths
        cond_seqlen = attention_kwargs["cond_sequence_len"]
        real_seqlen = attention_kwargs["image_sequence_len"]  # length of real image tokens
        pre_real_seqlen = real_seqlen
        pose_seqlen = img.shape[1] - cond_seqlen - real_seqlen - pre_real_seqlen # Total len - cond_len - real_len

        vec_cond = attention_kwargs["vec_cond"]
        vec_pose = attention_kwargs["vec_pose"]
        vec_pre_real = attention_kwargs["vec_pre_real"]
        assert vec_pre_real is not None, "vec_pre_real is required for version2"

        ar_attn_mask = attention_kwargs["ar_attn_mask"] # can be None

        bbox_id_embeddings = attention_kwargs.get("bbox_id_embeddings", None)
        mockpe = attention_kwargs.get("mockpe", None)
        pe_learnable = attention_kwargs.get("pe_learnable", False)
        batch_shift = attention_kwargs.get("batch_shift", None)
        batch_resize_scale = attention_kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = attention_kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = attention_kwargs.get("learned_shift_scale_ratio", None)
        ori_h = attention_kwargs.get("ori_h", None)
        ori_w = attention_kwargs.get("ori_w", None)


        # If you want to allow runtime override for text_lora, do:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) Split cond, pose tokens from real tokens
        cond_img = img[:, :cond_seqlen]                                   # [bs, cond_seqlen, dim]
        pose_img = img[:, cond_seqlen:cond_seqlen + pose_seqlen]          # [bs, pose_seqlen, dim]
        pre_real_img = img[:, cond_seqlen + pose_seqlen:cond_seqlen + pose_seqlen + pre_real_seqlen] # [bs, pre_real_seqlen, dim]
        real_img = img[:, cond_seqlen + pose_seqlen + pre_real_seqlen:]                     # [bs, real_seqlen, dim]

        # ---------------------------------------------------------------------
        # HELPER: compute ModulationOut with LoRA offset for Pose/Cond
        def pose_modulation_forward(mod_block: nn.Module, vector: Tensor):
            """
            This replicates mod_block(vec) but inserts a LoRA offset.
            - Normally: out = mod_block.lin(silu(vec))
            - We'll add pose_mod_lora(silu(vec)) * lora_weight
            Then chunk the result into two ModulationOut sets of (shift, scale, gate).
            This helper is used for both cond and pose streams.
            """
            base = mod_block.lin(nn.functional.silu(vector))  # [bs, 6*dim]
            offset = self.pose_mod_lora(nn.functional.silu(vector)) * self.lora_weight
            combined = base + offset  # [bs, 6*dim]
            # shape to [bs, 1, 6*dim], chunk into 2 sets of 3
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # (shift, scale, gate)
                ModulationOut(*chunked[3:])   # (shift, scale, gate)
            )

        # HELPER: compute ModulationOut for real (no LoRA offset)
        def real_modulation_forward(mod_block: nn.Module, real_vector: Tensor):
            return mod_block(real_vector)  # calls mod_block.forward()

        # HELPER: compute ModulationOut for text, optional LoRA
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor):
            # Normal path: mod_block(text_vector)
            return mod_block(text_vector)

        # 3) Gather the modulation parameters
        # Cond uses separate LoRA-based mod
        cond_mod1, cond_mod2 = pose_modulation_forward(attn.img_mod, vec_cond)
        # Pose uses separate LoRA-based mod
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)
        # Real uses normal mod
        real_mod1, real_mod2 = real_modulation_forward(attn.img_mod, vec)
        pre_real_mod1, pre_real_mod2 = real_modulation_forward(attn.img_mod, vec_pre_real)
        # Text uses normal mod (original code used vec_pose for txt_mod)
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose)

        # ---------------------------------------------------------------------
        # HELPER: apply QKV with LoRA offset
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool):
            """
            x: [bs, seq_len, dim] input tokens (after norm & shift/scale).
            attn_module.qkv: the base QKV linear
            lora_downup: the LoRA offset (qkv_lora_pose or qkv_lora_real or qkv_lora_text)
            """
            base_qkv = attn_module.qkv(x)  # [bs, seq_len, 3*dim]
            if do_lora:
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # shape [bs, seq_len, 3*dim]
            else:
                return base_qkv

        # 4) Pre-Attention Norm + Shift/Scale for each stream
        # Cond
        cond_modulated = attn.img_norm1(cond_img)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        # Pose
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift
        # Real
        real_modulated = attn.img_norm1(real_img)
        real_modulated = (1 + real_mod1.scale) * real_modulated + real_mod1.shift
        pre_real_modulated = attn.img_norm1(pre_real_img)
        pre_real_modulated = (1 + pre_real_mod1.scale) * pre_real_modulated + pre_real_mod1.shift
        # Text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # 5) QKV for each portion
        # Cond uses pose_lora (as per "cond and pose will gone through exactly same process")
        cond_qkv = apply_qkv_with_lora(cond_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        real_qkv = apply_qkv_with_lora(real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        pre_real_qkv = apply_qkv_with_lora(pre_real_modulated, attn.img_attn, self.qkv_lora_pre_real, self.pre_real_lora)
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)  # might have no LoRA if text_lora=False

        # 6) Rearrange and combine
        # shape for each qkv is: [bs, seq_len, 3*dim]
        bs, cond_len, _ = cond_qkv.shape
        _, pose_len, _ = pose_qkv.shape
        _, real_len, _ = real_qkv.shape
        _, pre_real_len, _ = pre_real_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # Split into q, k, v (B L (3 H D)) -> (3, B, H, L, D)
        # using attn.num_heads / attn.head_dim from the block
        def split_qkv(x, attn_module):
            # Pass attn_module to get num_heads and head_dim
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=attn_module.num_heads)
            return q, k, v

        # Split using the respective attention module for num_heads and head_dim
        # Assuming img_attn and txt_attn have these attributes.
        cond_q, cond_k, cond_v = split_qkv(cond_qkv, attn.img_attn)
        pose_q, pose_k, pose_v = split_qkv(pose_qkv, attn.img_attn)
        real_q, real_k, real_v = split_qkv(real_qkv, attn.img_attn)
        pre_real_q, pre_real_k, pre_real_v = split_qkv(pre_real_qkv, attn.img_attn)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv,  attn.txt_attn)

        # run the typical "attn.img_attn.norm" or "attn.txt_attn.norm" for each
        # (some implementations do a small pre-softmax normalization)
        cond_q, cond_k = attn.img_attn.norm(cond_q, cond_k, cond_v)
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        real_q, real_k = attn.img_attn.norm(real_q, real_k, real_v)
        pre_real_q, pre_real_k = attn.img_attn.norm(pre_real_q, pre_real_k, pre_real_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # Now we concatenate all streams: text first, then cond, then pose, then real
        q = torch.cat([txt_q, cond_q, pose_q, pre_real_q, real_q], dim=-2)  # along sequence dimension
        k = torch.cat([txt_k, cond_k, pose_k, pre_real_k, real_k], dim=-2)
        v = torch.cat([txt_v, cond_v, pose_v, pre_real_v, real_v], dim=-2)

        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert False, "pe_learnable is not used anymore"

        assert bbox_id_embeddings is None, "bbox_id_embeddings is not used anymore"

        attn_out = attention(q, k, v, pe=pe, attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings)  # shape [bs, (txt_len+cond_len+pose_len+real_len), dim]

        # 8) Split the attention output back to text, cond, pose, real
        txt_attn_out  = attn_out[:, :txt_len]
        cond_attn_out = attn_out[:, txt_len : txt_len + cond_len]
        pose_attn_out = attn_out[:, txt_len + cond_len : txt_len + cond_len + pose_len]
        pre_real_attn_out = attn_out[:, txt_len + cond_len + pose_len : txt_len + cond_len + pose_len + pre_real_len]
        real_attn_out = attn_out[:, txt_len + cond_len + pose_len + pre_real_len:]

        # 9) Residual + gating for Cond (shares pose_lora)
        if self.pose_lora: # This pose_lora flag now applies to cond too
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out) \
                                + cond_mod1.gate * self.proj_lora_pose(cond_attn_out) * self.lora_weight
        else:
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out)
        # MLP part with second modulation
        cond_img = cond_img + cond_mod2.gate * attn.img_mlp(
            (1 + cond_mod2.scale) * attn.img_norm2(cond_img) + cond_mod2.shift
        )


        # 10) Residual + gating for Pose (shares pose_lora)
        if self.pose_lora:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                                + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight
        else:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out)
        # MLP part with second modulation
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 11) Residual + gating for Real
        if self.real_lora:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out) \
                                + real_mod1.gate * self.proj_lora_real(real_attn_out) * self.lora_weight
        else:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out)

        if self.pre_real_lora:
            pre_real_img = pre_real_img + pre_real_mod1.gate * attn.img_attn.proj(pre_real_attn_out) \
                                + pre_real_mod1.gate * self.proj_lora_pre_real(pre_real_attn_out) * self.lora_weight
        else:
            pre_real_img = pre_real_img + pre_real_mod1.gate * attn.img_attn.proj(pre_real_attn_out)
        # MLP part with second modulation
        real_img = real_img + real_mod2.gate * attn.img_mlp(
            (1 + real_mod2.scale) * attn.img_norm2(real_img) + real_mod2.shift
        )
        pre_real_img = pre_real_img + pre_real_mod2.gate * attn.img_mlp(
            (1 + pre_real_mod2.scale) * attn.img_norm2(pre_real_img) + pre_real_mod2.shift
        )

        # 12) Residual + gating for Text
        # If text_lora=False, no offset; if text_lora=True, use self.proj_lora_text
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 13) Re-combine cond, pose and real into a single `img` again
        img = torch.cat([cond_img, pose_img, pre_real_img, real_img], dim=1)  # [bs, cond_len + pose_len + pre_real_len + real_len, dim]

        return img, txt



    def forward_version1(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        attn: the DoubleStreamBlock
          - attn.img_mod -> Modulation block for images
          - attn.txt_mod -> Modulation block for text
          - attn.img_attn -> SelfAttention for images
          - attn.txt_attn -> SelfAttention for text
        img: [bs, (cond_seqlen + pose_seqlen + real_seqlen), dim]
        txt: [bs, 512, dim]
        vec: the 'real' image embedding
        pe: positional embedding for attention (now 3-signal version)
        attention_kwargs includes:
          - cond_sequence_len: the length of the conditioning images
          - image_sequence_len: the length of the real images
          - vec_cond: the embedding for the conditioning images
          - vec_pose: the embedding for the pose images
          - (optionally) text_lora: if you want a dynamic override
        """
        # 1) Parse out the cond, pose and real lengths
        cond_seqlen = attention_kwargs["cond_sequence_len"]
        real_seqlen = attention_kwargs["image_sequence_len"]  # length of real image tokens
        pose_seqlen = img.shape[1] - cond_seqlen - real_seqlen # Total len - cond_len - real_len

        vec_cond = attention_kwargs["vec_cond"]
        vec_pose = attention_kwargs["vec_pose"]

        ar_attn_mask = attention_kwargs["ar_attn_mask"] # can be None

        bbox_id_embeddings = attention_kwargs.get("bbox_id_embeddings", None)
        mockpe = attention_kwargs.get("mockpe", None)
        pe_learnable = attention_kwargs.get("pe_learnable", False)
        batch_shift = attention_kwargs.get("batch_shift", None)
        batch_resize_scale = attention_kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = attention_kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = attention_kwargs.get("learned_shift_scale_ratio", None)
        ori_h = attention_kwargs.get("ori_h", None)
        ori_w = attention_kwargs.get("ori_w", None)


        # If you want to allow runtime override for text_lora, do:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) Split cond, pose tokens from real tokens
        cond_img = img[:, :cond_seqlen]                                   # [bs, cond_seqlen, dim]
        pose_img = img[:, cond_seqlen:cond_seqlen + pose_seqlen]          # [bs, pose_seqlen, dim]
        real_img = img[:, cond_seqlen + pose_seqlen:]                     # [bs, real_seqlen, dim]

        # print("DEBUG -----------------------------------")
        # print("cond_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", cond_img.shape)
        # print("pose_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", pose_img.shape)
        # print("real_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", real_img.shape)
        # print("DEBUG -----------------------------------")

        # ---------------------------------------------------------------------
        # HELPER: compute ModulationOut with LoRA offset for Pose/Cond
        def pose_modulation_forward(mod_block: nn.Module, vector: Tensor):
            """
            This replicates mod_block(vec) but inserts a LoRA offset.
            - Normally: out = mod_block.lin(silu(vec))
            - We'll add pose_mod_lora(silu(vec)) * lora_weight
            Then chunk the result into two ModulationOut sets of (shift, scale, gate).
            This helper is used for both cond and pose streams.
            """
            base = mod_block.lin(nn.functional.silu(vector))  # [bs, 6*dim]
            offset = self.pose_mod_lora(nn.functional.silu(vector)) * self.lora_weight
            combined = base + offset  # [bs, 6*dim]
            # shape to [bs, 1, 6*dim], chunk into 2 sets of 3
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # (shift, scale, gate)
                ModulationOut(*chunked[3:])   # (shift, scale, gate)
            )

        # HELPER: compute ModulationOut for real (no LoRA offset)
        def real_modulation_forward(mod_block: nn.Module, real_vector: Tensor):
            return mod_block(real_vector)  # calls mod_block.forward()

        # HELPER: compute ModulationOut for text, optional LoRA
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor):
            # Normal path: mod_block(text_vector)
            return mod_block(text_vector)

        # 3) Gather the modulation parameters
        # Cond uses separate LoRA-based mod
        cond_mod1, cond_mod2 = pose_modulation_forward(attn.img_mod, vec_cond)
        # Pose uses separate LoRA-based mod
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)
        # Real uses normal mod
        real_mod1, real_mod2 = real_modulation_forward(attn.img_mod, vec)
        # Text uses normal mod (original code used vec_pose for txt_mod)
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose)

        # ---------------------------------------------------------------------
        # HELPER: apply QKV with LoRA offset
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool):
            """
            x: [bs, seq_len, dim] input tokens (after norm & shift/scale).
            attn_module.qkv: the base QKV linear
            lora_downup: the LoRA offset (qkv_lora_pose or qkv_lora_real or qkv_lora_text)
            """
            base_qkv = attn_module.qkv(x)  # [bs, seq_len, 3*dim]
            if do_lora:
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # shape [bs, seq_len, 3*dim]
            else:
                return base_qkv

        # 4) Pre-Attention Norm + Shift/Scale for each stream
        # Cond
        cond_modulated = attn.img_norm1(cond_img)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        # Pose
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift
        # Real
        real_modulated = attn.img_norm1(real_img)
        real_modulated = (1 + real_mod1.scale) * real_modulated + real_mod1.shift
        # Text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # print("DEBUG -----------------------------------")
        # print("cond_modulated.shape", cond_modulated.shape)
        # print("pose_modulated.shape", pose_modulated.shape)
        # print("real_modulated.shape", real_modulated.shape)
        # print("txt_modulated.shape", txt_modulated.shape)
        # print("DEBUG -----------------------------------")

        # 5) QKV for each portion
        # Cond uses pose_lora (as per "cond and pose will gone through exactly same process")
        cond_qkv = apply_qkv_with_lora(cond_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        real_qkv = apply_qkv_with_lora(real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)  # might have no LoRA if text_lora=False

        # 6) Rearrange and combine
        # shape for each qkv is: [bs, seq_len, 3*dim]
        bs, cond_len, _ = cond_qkv.shape
        _, pose_len, _ = pose_qkv.shape
        _, real_len, _ = real_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # print("DEBUG -----------------------------------")
        # print("cond_qkv.shape", cond_qkv.shape)
        # print("pose_qkv.shape", pose_qkv.shape)
        # print("real_qkv.shape", real_qkv.shape)
        # print("txt_qkv.shape", txt_qkv.shape)
        # print("DEBUG -----------------------------------")

        # Split into q, k, v (B L (3 H D)) -> (3, B, H, L, D)
        # using attn.num_heads / attn.head_dim from the block
        def split_qkv(x, attn_module):
            # Pass attn_module to get num_heads and head_dim
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=attn_module.num_heads)
            return q, k, v

        # Split using the respective attention module for num_heads and head_dim
        # Assuming img_attn and txt_attn have these attributes.
        cond_q, cond_k, cond_v = split_qkv(cond_qkv, attn.img_attn)
        pose_q, pose_k, pose_v = split_qkv(pose_qkv, attn.img_attn)
        real_q, real_k, real_v = split_qkv(real_qkv, attn.img_attn)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv,  attn.txt_attn)

        # run the typical "attn.img_attn.norm" or "attn.txt_attn.norm" for each
        # (some implementations do a small pre-softmax normalization)
        cond_q, cond_k = attn.img_attn.norm(cond_q, cond_k, cond_v)
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        real_q, real_k = attn.img_attn.norm(real_q, real_k, real_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # print("DEBUG -----------------------------------")
        # print("cond_q.shape", cond_q.shape)
        # print("pose_q.shape", pose_q.shape)
        # print("real_q.shape", real_q.shape)
        # print("txt_q.shape", txt_q.shape)
        # print("DEBUG -----------------------------------")

        # Now we concatenate all streams: text first, then cond, then pose, then real
        q = torch.cat([txt_q, cond_q, pose_q, real_q], dim=-2)  # along sequence dimension
        k = torch.cat([txt_k, cond_k, pose_k, real_k], dim=-2)
        v = torch.cat([txt_v, cond_v, pose_v, real_v], dim=-2)

        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert mockpe is not None, "mockpe is required when pe_learnable is True"
            assert learned_shift_scale_ratio is not None, "learned_shift_scale_ratio is required when pe_learnable is True"
            mock_shift_scale = mockpe(k, v) # [bs, 3, 1]
            mock_shift_scale = mock_shift_scale.squeeze(-1) # [bs, 3]
            shift_x_learned, shift_y_learned, scale_learned = mock_shift_scale[:, 0], mock_shift_scale[:, 1], mock_shift_scale[:, 2] # [bs]
            shift_learned = torch.stack([shift_x_learned, shift_y_learned], dim=1) # (B, 2)

            use_learned_shift_scale = torch.rand(()) <= learned_shift_scale_ratio  # scalar boolean -- switch once for whole batch
            if use_learned_shift_scale:
                batch_shift_to_use = shift_learned
                batch_resize_scale_to_use = scale_learned
            else:
                batch_shift_to_use = batch_shift
                batch_resize_scale_to_use = batch_resize_scale

            pe = adjust_condition_ids_inplace(batch_ids=pe, batch_multi_person_last_indicator=batch_multi_person_last_indicator, batch_shift=batch_shift_to_use, batch_resize_scale=batch_resize_scale_to_use, ori_h=ori_h, ori_w=ori_w)
            pe = EmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(pe)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 7) Run attention
        # Update build_mask and build_mask_batch arguments
        # if self.real_image_active is None:
        #     if self.use_attn_mask: # when sampling, vec_pose is None
        #         # Assuming hardcoded lengths for mask are txt_len=512, cond_len=1024, pose_len=1024, img_len=1024
        #         # These should ideally be derived from actual sequence lengths if dynamic.
        #         attn_mask = build_mask(txt_len=txt_len, cond_len=cond_len, pose_len=pose_len, img_len=real_len, device=q.device)
        #     else:
        #         attn_mask = None
        # else:
        #     attn_mask = build_mask_batch(txt_len=txt_len, cond_len=cond_len, pose_len=pose_len, img_len=real_len, device=q.device, real_image_active=self.real_image_active)

        # print("DEBUG -----------------------------------")
        # print("q.shape", q.shape)
        # print("k.shape", k.shape)
        # print("v.shape", v.shape)
        # print("pe.shape", pe.shape)
        # print("DEBUG -----------------------------------")

        attn_out = attention(q, k, v, pe=pe, attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings)  # shape [bs, (txt_len+cond_len+pose_len+real_len), dim]

        # 8) Split the attention output back to text, cond, pose, real
        txt_attn_out  = attn_out[:, :txt_len]
        cond_attn_out = attn_out[:, txt_len : txt_len + cond_len]
        pose_attn_out = attn_out[:, txt_len + cond_len : txt_len + cond_len + pose_len]
        real_attn_out = attn_out[:, txt_len + cond_len + pose_len:]

        # 9) Residual + gating for Cond (shares pose_lora)
        if self.pose_lora: # This pose_lora flag now applies to cond too
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out) \
                                + cond_mod1.gate * self.proj_lora_pose(cond_attn_out) * self.lora_weight
        else:
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out)
        # MLP part with second modulation
        cond_img = cond_img + cond_mod2.gate * attn.img_mlp(
            (1 + cond_mod2.scale) * attn.img_norm2(cond_img) + cond_mod2.shift
        )


        # 10) Residual + gating for Pose (shares pose_lora)
        if self.pose_lora:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                                + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight
        else:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out)
        # MLP part with second modulation
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 11) Residual + gating for Real
        if self.real_lora:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out) \
                                + real_mod1.gate * self.proj_lora_real(real_attn_out) * self.lora_weight
        else:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out)
        # MLP part with second modulation
        real_img = real_img + real_mod2.gate * attn.img_mlp(
            (1 + real_mod2.scale) * attn.img_norm2(real_img) + real_mod2.shift
        )

        # 12) Residual + gating for Text
        # If text_lora=False, no offset; if text_lora=True, use self.proj_lora_text
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 13) Re-combine cond, pose and real into a single `img` again
        img = torch.cat([cond_img, pose_img, real_img], dim=1)  # [bs, cond_len + pose_len + real_len, dim]

        return img, txt


class DoubleStreamBlockPoseLoraProcessorAR_version2_crossattn(nn.Module):
    """
    A processor that handles:
      - Cond + Pose + Real image sequences (split from `img`)
      - Pose and Cond modulation have an additional LoRA offset
      - Separate LoRA for pose QKV/proj vs. real QKV/proj
      - Optional LoRA for text if text_lora=True
    """
    def __init__(self,
                 dim: int,
                 rank=4,
                 network_alpha=None,
                 lora_weight=1.0,
                 text_lora=False,
                 pose_lora=True,
                 real_lora=False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.use_attn_mask = False
        self.real_image_active = None

        # Separate LoRA for pose/cond QKV/proj (cond and pose share LoRA based on prompt)
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pose = LoRALinearLayer(dim, dim, rank, network_alpha)

        # LoRA for pose/cond modulation (cond and pose share LoRA based on prompt)
        self.pose_mod_lora = LoRALinearLayer(dim, 6 * dim, rank, network_alpha)

        # If text_lora=True, we also add LoRA to the text side
        self.text_lora = text_lora
        self.pose_lora = pose_lora # This now applies to both cond and pose image streams
        self.real_lora = real_lora

        self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)
        # if self.real_lora:
        #     # Separate LoRA for real QKV/proj
        #     self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        #     self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)
        # else:
        #     self.qkv_lora_real = None
        #     self.proj_lora_real = None

        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora_text = LoRALinearLayer(dim, dim, rank, network_alpha)
        else:
            # placeholders to keep code simpler; won't be used
            self.qkv_lora_text = None
            self.proj_lora_text = None

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        attn: the DoubleStreamBlock
          - attn.img_mod -> Modulation block for images
          - attn.txt_mod -> Modulation block for text
          - attn.img_attn -> SelfAttention for images
          - attn.txt_attn -> SelfAttention for text
        img: [bs, (cond_seqlen + pose_seqlen + real_seqlen), dim]
        txt: [bs, 512, dim]
        vec: the 'real' image embedding
        pe: positional embedding for attention (now 3-signal version)
        attention_kwargs includes:
          - cond_sequence_len: the length of the conditioning images
          - image_sequence_len: the length of the real images
          - vec_cond: the embedding for the conditioning images
          - vec_pose: the embedding for the pose images
          - (optionally) text_lora: if you want a dynamic override
        """
        # 1) Parse out the cond, pose and real lengths
        cond_seqlen = attention_kwargs["cond_sequence_len"]
        real_seqlen = attention_kwargs["image_sequence_len"]  # length of real image tokens
        pre_real_seqlen = real_seqlen
        pose_seqlen = img.shape[1] - cond_seqlen - real_seqlen - pre_real_seqlen # Total len - cond_len - real_len

        vec_cond = attention_kwargs["vec_cond"]
        vec_pose = attention_kwargs["vec_pose"]
        vec_pre_real = attention_kwargs["vec_pre_real"]
        assert vec_pre_real is not None, "vec_pre_real is required for version2"

        ar_attn_mask = attention_kwargs["ar_attn_mask"] # can be None

        bbox_id_embeddings = attention_kwargs.get("bbox_id_embeddings", None)
        mockpe = attention_kwargs.get("mockpe", None)
        pe_learnable = attention_kwargs.get("pe_learnable", False)
        batch_shift = attention_kwargs.get("batch_shift", None)
        batch_resize_scale = attention_kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = attention_kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = attention_kwargs.get("learned_shift_scale_ratio", None)
        ori_h = attention_kwargs.get("ori_h", None)
        ori_w = attention_kwargs.get("ori_w", None)


        # If you want to allow runtime override for text_lora, do:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) Split cond, pose tokens from real tokens
        cond_img = img[:, :cond_seqlen]                                   # [bs, cond_seqlen, dim]
        pose_img = img[:, cond_seqlen:cond_seqlen + pose_seqlen]          # [bs, pose_seqlen, dim]
        pre_real_img = img[:, cond_seqlen + pose_seqlen:cond_seqlen + pose_seqlen + pre_real_seqlen] # [bs, pre_real_seqlen, dim]
        real_img = img[:, cond_seqlen + pose_seqlen + pre_real_seqlen:]                     # [bs, real_seqlen, dim]

        # ---------------------------------------------------------------------
        # HELPER: compute ModulationOut with LoRA offset for Pose/Cond
        def pose_modulation_forward(mod_block: nn.Module, vector: Tensor):
            """
            This replicates mod_block(vec) but inserts a LoRA offset.
            - Normally: out = mod_block.lin(silu(vec))
            - We'll add pose_mod_lora(silu(vec)) * lora_weight
            Then chunk the result into two ModulationOut sets of (shift, scale, gate).
            This helper is used for both cond and pose streams.
            """
            base = mod_block.lin(nn.functional.silu(vector))  # [bs, 6*dim]
            offset = self.pose_mod_lora(nn.functional.silu(vector)) * self.lora_weight
            combined = base + offset  # [bs, 6*dim]
            # shape to [bs, 1, 6*dim], chunk into 2 sets of 3
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # (shift, scale, gate)
                ModulationOut(*chunked[3:])   # (shift, scale, gate)
            )

        # HELPER: compute ModulationOut for real (no LoRA offset)
        def real_modulation_forward(mod_block: nn.Module, real_vector: Tensor):
            return mod_block(real_vector)  # calls mod_block.forward()

        # HELPER: compute ModulationOut for text, optional LoRA
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor):
            # Normal path: mod_block(text_vector)
            return mod_block(text_vector)

        # 3) Gather the modulation parameters
        # Cond uses separate LoRA-based mod
        cond_mod1, cond_mod2 = pose_modulation_forward(attn.img_mod, vec_cond)
        # Pose uses separate LoRA-based mod
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)
        # Real uses normal mod
        real_mod1, real_mod2 = real_modulation_forward(attn.img_mod, vec)
        pre_real_mod1, pre_real_mod2 = real_modulation_forward(attn.img_mod, vec_pre_real)
        # Text uses normal mod (original code used vec_pose for txt_mod)
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose)

        # ---------------------------------------------------------------------
        # HELPER: apply QKV with LoRA offset
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool):
            """
            x: [bs, seq_len, dim] input tokens (after norm & shift/scale).
            attn_module.qkv: the base QKV linear
            lora_downup: the LoRA offset (qkv_lora_pose or qkv_lora_real or qkv_lora_text)
            """
            base_qkv = attn_module.qkv(x)  # [bs, seq_len, 3*dim]
            if do_lora:
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # shape [bs, seq_len, 3*dim]
            else:
                return base_qkv

        # 4) Pre-Attention Norm + Shift/Scale for each stream
        # Cond
        cond_modulated = attn.img_norm1(cond_img)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        # Pose
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift
        # Real
        real_modulated = attn.img_norm1(real_img)
        real_modulated = (1 + real_mod1.scale) * real_modulated + real_mod1.shift
        pre_real_modulated = attn.img_norm1(pre_real_img)
        pre_real_modulated = (1 + pre_real_mod1.scale) * pre_real_modulated + pre_real_mod1.shift
        # Text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # 5) QKV for each portion
        # Cond uses pose_lora (as per "cond and pose will gone through exactly same process")
        cond_qkv = apply_qkv_with_lora(cond_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        real_qkv = apply_qkv_with_lora(real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        pre_real_qkv = apply_qkv_with_lora(pre_real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)  # might have no LoRA if text_lora=False

        # 6) Rearrange and combine
        # shape for each qkv is: [bs, seq_len, 3*dim]
        bs, cond_len, _ = cond_qkv.shape
        _, pose_len, _ = pose_qkv.shape
        _, real_len, _ = real_qkv.shape
        _, pre_real_len, _ = pre_real_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # Split into q, k, v (B L (3 H D)) -> (3, B, H, L, D)
        # using attn.num_heads / attn.head_dim from the block
        def split_qkv(x, attn_module):
            # Pass attn_module to get num_heads and head_dim
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=attn_module.num_heads)
            return q, k, v

        # Split using the respective attention module for num_heads and head_dim
        # Assuming img_attn and txt_attn have these attributes.
        cond_q, cond_k, cond_v = split_qkv(cond_qkv, attn.img_attn)
        pose_q, pose_k, pose_v = split_qkv(pose_qkv, attn.img_attn)
        real_q, real_k, real_v = split_qkv(real_qkv, attn.img_attn)
        pre_real_q, pre_real_k, pre_real_v = split_qkv(pre_real_qkv, attn.img_attn)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv,  attn.txt_attn)

        # run the typical "attn.img_attn.norm" or "attn.txt_attn.norm" for each
        # (some implementations do a small pre-softmax normalization)
        cond_q, cond_k = attn.img_attn.norm(cond_q, cond_k, cond_v)
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        real_q, real_k = attn.img_attn.norm(real_q, real_k, real_v)
        pre_real_q, pre_real_k = attn.img_attn.norm(pre_real_q, pre_real_k, pre_real_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # Now we concatenate all streams: text first, then cond, then pose, then real
        q = torch.cat([txt_q, cond_q, pose_q, pre_real_q, real_q], dim=-2)  # along sequence dimension
        k = torch.cat([txt_k, cond_k, pose_k, pre_real_k, real_k], dim=-2)
        v = torch.cat([txt_v, cond_v, pose_v, pre_real_v, real_v], dim=-2)

        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert False, "pe_learnable is not used anymore"

        assert bbox_id_embeddings is None, "bbox_id_embeddings is not used anymore"

        attn_out = attention(q, k, v, pe=pe, attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings)  # shape [bs, (txt_len+cond_len+pose_len+real_len), dim]

        # 8) Split the attention output back to text, cond, pose, real
        txt_attn_out  = attn_out[:, :txt_len]
        cond_attn_out = attn_out[:, txt_len : txt_len + cond_len]
        pose_attn_out = attn_out[:, txt_len + cond_len : txt_len + cond_len + pose_len]
        pre_real_attn_out = attn_out[:, txt_len + cond_len + pose_len : txt_len + cond_len + pose_len + pre_real_len]
        real_attn_out = attn_out[:, txt_len + cond_len + pose_len + pre_real_len:]

        # 9) Residual + gating for Cond (shares pose_lora)
        if self.pose_lora: # This pose_lora flag now applies to cond too
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out) \
                                + cond_mod1.gate * self.proj_lora_pose(cond_attn_out) * self.lora_weight
        else:
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out)
        # MLP part with second modulation
        cond_img = cond_img + cond_mod2.gate * attn.img_mlp(
            (1 + cond_mod2.scale) * attn.img_norm2(cond_img) + cond_mod2.shift
        )


        # 10) Residual + gating for Pose (shares pose_lora)
        if self.pose_lora:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                                + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight
        else:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out)
        # MLP part with second modulation
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 11) Residual + gating for Real
        if self.real_lora:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out) \
                                + real_mod1.gate * self.proj_lora_real(real_attn_out) * self.lora_weight
            # pre_real_img = pre_real_img + pre_real_mod1.gate * attn.img_attn.proj(pre_real_attn_out) \
            #                     + pre_real_mod1.gate * self.proj_lora_real(pre_real_attn_out) * self.lora_weight
        else:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out)
            pre_real_img = pre_real_img + pre_real_mod1.gate * attn.img_attn.proj(pre_real_attn_out)
        # MLP part with second modulation
        real_img = real_img + real_mod2.gate * attn.img_mlp(
            (1 + real_mod2.scale) * attn.img_norm2(real_img) + real_mod2.shift
        )
        pre_real_img = pre_real_img + pre_real_mod2.gate * attn.img_mlp(
            (1 + pre_real_mod2.scale) * attn.img_norm2(pre_real_img) + pre_real_mod2.shift
        )

        # 12) Residual + gating for Text
        # If text_lora=False, no offset; if text_lora=True, use self.proj_lora_text
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 13) Re-combine cond, pose and real into a single `img` again
        img = torch.cat([cond_img, pose_img, pre_real_img, real_img], dim=1)  # [bs, cond_len + pose_len + pre_real_len + real_len, dim]

        return img, txt



    def forward_version1(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        attn: the DoubleStreamBlock
          - attn.img_mod -> Modulation block for images
          - attn.txt_mod -> Modulation block for text
          - attn.img_attn -> SelfAttention for images
          - attn.txt_attn -> SelfAttention for text
        img: [bs, (cond_seqlen + pose_seqlen + real_seqlen), dim]
        txt: [bs, 512, dim]
        vec: the 'real' image embedding
        pe: positional embedding for attention (now 3-signal version)
        attention_kwargs includes:
          - cond_sequence_len: the length of the conditioning images
          - image_sequence_len: the length of the real images
          - vec_cond: the embedding for the conditioning images
          - vec_pose: the embedding for the pose images
          - (optionally) text_lora: if you want a dynamic override
        """
        # 1) Parse out the cond, pose and real lengths
        cond_seqlen = attention_kwargs["cond_sequence_len"]
        real_seqlen = attention_kwargs["image_sequence_len"]  # length of real image tokens
        pose_seqlen = img.shape[1] - cond_seqlen - real_seqlen # Total len - cond_len - real_len

        vec_cond = attention_kwargs["vec_cond"]
        vec_pose = attention_kwargs["vec_pose"]

        ar_attn_mask = attention_kwargs["ar_attn_mask"] # can be None

        bbox_id_embeddings = attention_kwargs.get("bbox_id_embeddings", None)
        mockpe = attention_kwargs.get("mockpe", None)
        pe_learnable = attention_kwargs.get("pe_learnable", False)
        batch_shift = attention_kwargs.get("batch_shift", None)
        batch_resize_scale = attention_kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = attention_kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = attention_kwargs.get("learned_shift_scale_ratio", None)
        ori_h = attention_kwargs.get("ori_h", None)
        ori_w = attention_kwargs.get("ori_w", None)


        # If you want to allow runtime override for text_lora, do:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) Split cond, pose tokens from real tokens
        cond_img = img[:, :cond_seqlen]                                   # [bs, cond_seqlen, dim]
        pose_img = img[:, cond_seqlen:cond_seqlen + pose_seqlen]          # [bs, pose_seqlen, dim]
        real_img = img[:, cond_seqlen + pose_seqlen:]                     # [bs, real_seqlen, dim]

        # print("DEBUG -----------------------------------")
        # print("cond_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", cond_img.shape)
        # print("pose_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", pose_img.shape)
        # print("real_img.shape inside DoubleStreamBlockPoseLoraProcessorAR", real_img.shape)
        # print("DEBUG -----------------------------------")

        # ---------------------------------------------------------------------
        # HELPER: compute ModulationOut with LoRA offset for Pose/Cond
        def pose_modulation_forward(mod_block: nn.Module, vector: Tensor):
            """
            This replicates mod_block(vec) but inserts a LoRA offset.
            - Normally: out = mod_block.lin(silu(vec))
            - We'll add pose_mod_lora(silu(vec)) * lora_weight
            Then chunk the result into two ModulationOut sets of (shift, scale, gate).
            This helper is used for both cond and pose streams.
            """
            base = mod_block.lin(nn.functional.silu(vector))  # [bs, 6*dim]
            offset = self.pose_mod_lora(nn.functional.silu(vector)) * self.lora_weight
            combined = base + offset  # [bs, 6*dim]
            # shape to [bs, 1, 6*dim], chunk into 2 sets of 3
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # (shift, scale, gate)
                ModulationOut(*chunked[3:])   # (shift, scale, gate)
            )

        # HELPER: compute ModulationOut for real (no LoRA offset)
        def real_modulation_forward(mod_block: nn.Module, real_vector: Tensor):
            return mod_block(real_vector)  # calls mod_block.forward()

        # HELPER: compute ModulationOut for text, optional LoRA
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor):
            # Normal path: mod_block(text_vector)
            return mod_block(text_vector)

        # 3) Gather the modulation parameters
        # Cond uses separate LoRA-based mod
        cond_mod1, cond_mod2 = pose_modulation_forward(attn.img_mod, vec_cond)
        # Pose uses separate LoRA-based mod
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)
        # Real uses normal mod
        real_mod1, real_mod2 = real_modulation_forward(attn.img_mod, vec)
        # Text uses normal mod (original code used vec_pose for txt_mod)
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose)

        # ---------------------------------------------------------------------
        # HELPER: apply QKV with LoRA offset
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool):
            """
            x: [bs, seq_len, dim] input tokens (after norm & shift/scale).
            attn_module.qkv: the base QKV linear
            lora_downup: the LoRA offset (qkv_lora_pose or qkv_lora_real or qkv_lora_text)
            """
            base_qkv = attn_module.qkv(x)  # [bs, seq_len, 3*dim]
            if do_lora:
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # shape [bs, seq_len, 3*dim]
            else:
                return base_qkv

        # 4) Pre-Attention Norm + Shift/Scale for each stream
        # Cond
        cond_modulated = attn.img_norm1(cond_img)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        # Pose
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift
        # Real
        real_modulated = attn.img_norm1(real_img)
        real_modulated = (1 + real_mod1.scale) * real_modulated + real_mod1.shift
        # Text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # print("DEBUG -----------------------------------")
        # print("cond_modulated.shape", cond_modulated.shape)
        # print("pose_modulated.shape", pose_modulated.shape)
        # print("real_modulated.shape", real_modulated.shape)
        # print("txt_modulated.shape", txt_modulated.shape)
        # print("DEBUG -----------------------------------")

        # 5) QKV for each portion
        # Cond uses pose_lora (as per "cond and pose will gone through exactly same process")
        cond_qkv = apply_qkv_with_lora(cond_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        real_qkv = apply_qkv_with_lora(real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)  # might have no LoRA if text_lora=False

        # 6) Rearrange and combine
        # shape for each qkv is: [bs, seq_len, 3*dim]
        bs, cond_len, _ = cond_qkv.shape
        _, pose_len, _ = pose_qkv.shape
        _, real_len, _ = real_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # print("DEBUG -----------------------------------")
        # print("cond_qkv.shape", cond_qkv.shape)
        # print("pose_qkv.shape", pose_qkv.shape)
        # print("real_qkv.shape", real_qkv.shape)
        # print("txt_qkv.shape", txt_qkv.shape)
        # print("DEBUG -----------------------------------")

        # Split into q, k, v (B L (3 H D)) -> (3, B, H, L, D)
        # using attn.num_heads / attn.head_dim from the block
        def split_qkv(x, attn_module):
            # Pass attn_module to get num_heads and head_dim
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=attn_module.num_heads)
            return q, k, v

        # Split using the respective attention module for num_heads and head_dim
        # Assuming img_attn and txt_attn have these attributes.
        cond_q, cond_k, cond_v = split_qkv(cond_qkv, attn.img_attn)
        pose_q, pose_k, pose_v = split_qkv(pose_qkv, attn.img_attn)
        real_q, real_k, real_v = split_qkv(real_qkv, attn.img_attn)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv,  attn.txt_attn)

        # run the typical "attn.img_attn.norm" or "attn.txt_attn.norm" for each
        # (some implementations do a small pre-softmax normalization)
        cond_q, cond_k = attn.img_attn.norm(cond_q, cond_k, cond_v)
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        real_q, real_k = attn.img_attn.norm(real_q, real_k, real_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # print("DEBUG -----------------------------------")
        # print("cond_q.shape", cond_q.shape)
        # print("pose_q.shape", pose_q.shape)
        # print("real_q.shape", real_q.shape)
        # print("txt_q.shape", txt_q.shape)
        # print("DEBUG -----------------------------------")

        # Now we concatenate all streams: text first, then cond, then pose, then real
        q = torch.cat([txt_q, cond_q, pose_q, real_q], dim=-2)  # along sequence dimension
        k = torch.cat([txt_k, cond_k, pose_k, real_k], dim=-2)
        v = torch.cat([txt_v, cond_v, pose_v, real_v], dim=-2)

        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert mockpe is not None, "mockpe is required when pe_learnable is True"
            assert learned_shift_scale_ratio is not None, "learned_shift_scale_ratio is required when pe_learnable is True"
            mock_shift_scale = mockpe(k, v) # [bs, 3, 1]
            mock_shift_scale = mock_shift_scale.squeeze(-1) # [bs, 3]
            shift_x_learned, shift_y_learned, scale_learned = mock_shift_scale[:, 0], mock_shift_scale[:, 1], mock_shift_scale[:, 2] # [bs]
            shift_learned = torch.stack([shift_x_learned, shift_y_learned], dim=1) # (B, 2)

            use_learned_shift_scale = torch.rand(()) <= learned_shift_scale_ratio  # scalar boolean -- switch once for whole batch
            if use_learned_shift_scale:
                batch_shift_to_use = shift_learned
                batch_resize_scale_to_use = scale_learned
            else:
                batch_shift_to_use = batch_shift
                batch_resize_scale_to_use = batch_resize_scale

            pe = adjust_condition_ids_inplace(batch_ids=pe, batch_multi_person_last_indicator=batch_multi_person_last_indicator, batch_shift=batch_shift_to_use, batch_resize_scale=batch_resize_scale_to_use, ori_h=ori_h, ori_w=ori_w)
            pe = EmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(pe)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 7) Run attention
        # Update build_mask and build_mask_batch arguments
        # if self.real_image_active is None:
        #     if self.use_attn_mask: # when sampling, vec_pose is None
        #         # Assuming hardcoded lengths for mask are txt_len=512, cond_len=1024, pose_len=1024, img_len=1024
        #         # These should ideally be derived from actual sequence lengths if dynamic.
        #         attn_mask = build_mask(txt_len=txt_len, cond_len=cond_len, pose_len=pose_len, img_len=real_len, device=q.device)
        #     else:
        #         attn_mask = None
        # else:
        #     attn_mask = build_mask_batch(txt_len=txt_len, cond_len=cond_len, pose_len=pose_len, img_len=real_len, device=q.device, real_image_active=self.real_image_active)

        # print("DEBUG -----------------------------------")
        # print("q.shape", q.shape)
        # print("k.shape", k.shape)
        # print("v.shape", v.shape)
        # print("pe.shape", pe.shape)
        # print("DEBUG -----------------------------------")

        attn_out = attention(q, k, v, pe=pe, attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings)  # shape [bs, (txt_len+cond_len+pose_len+real_len), dim]

        # 8) Split the attention output back to text, cond, pose, real
        txt_attn_out  = attn_out[:, :txt_len]
        cond_attn_out = attn_out[:, txt_len : txt_len + cond_len]
        pose_attn_out = attn_out[:, txt_len + cond_len : txt_len + cond_len + pose_len]
        real_attn_out = attn_out[:, txt_len + cond_len + pose_len:]

        # 9) Residual + gating for Cond (shares pose_lora)
        if self.pose_lora: # This pose_lora flag now applies to cond too
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out) \
                                + cond_mod1.gate * self.proj_lora_pose(cond_attn_out) * self.lora_weight
        else:
            cond_img = cond_img + cond_mod1.gate * attn.img_attn.proj(cond_attn_out)
        # MLP part with second modulation
        cond_img = cond_img + cond_mod2.gate * attn.img_mlp(
            (1 + cond_mod2.scale) * attn.img_norm2(cond_img) + cond_mod2.shift
        )


        # 10) Residual + gating for Pose (shares pose_lora)
        if self.pose_lora:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                                + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight
        else:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out)
        # MLP part with second modulation
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 11) Residual + gating for Real
        if self.real_lora:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out) \
                                + real_mod1.gate * self.proj_lora_real(real_attn_out) * self.lora_weight
        else:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out)
        # MLP part with second modulation
        real_img = real_img + real_mod2.gate * attn.img_mlp(
            (1 + real_mod2.scale) * attn.img_norm2(real_img) + real_mod2.shift
        )

        # 12) Residual + gating for Text
        # If text_lora=False, no offset; if text_lora=True, use self.proj_lora_text
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 13) Re-combine cond, pose and real into a single `img` again
        img = torch.cat([cond_img, pose_img, real_img], dim=1)  # [bs, cond_len + pose_len + real_len, dim]

        return img, txt





class DoubleStreamBlockPoseLoraProcessor(nn.Module):
    """
    A processor that handles:
      - Pose + Real image sequences (split from `img`)
      - Pose modulation has an additional LoRA offset
      - Separate LoRA for pose QKV/proj vs. real QKV/proj
      - Optional LoRA for text if text_lora=True
    """
    def __init__(self,
                 dim: int,
                 rank=4,
                 network_alpha=None,
                 lora_weight=1.0,
                 text_lora=False,
                 pose_lora=False,
                 real_lora=False):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.use_attn_mask = False
        self.real_image_active = None

        # Separate LoRA for pose QKV/proj
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pose = LoRALinearLayer(dim, dim, rank, network_alpha)

        # Separate LoRA for real QKV/proj
        if real_lora:
            self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora_real = LoRALinearLayer(dim, dim, rank, network_alpha)
        else:
            self.qkv_lora_real = None
            self.proj_lora_real = None

        # LoRA for pose modulation
        # The "Modulation" block is normally a single nn.Linear(dim -> 6*dim)
        # We'll add a rank-R offset to it for the pose embedding
        self.pose_mod_lora = LoRALinearLayer(dim, 6 * dim, rank, network_alpha)

        # If text_lora=True, we also add LoRA to the text side
        self.text_lora = text_lora
        self.pose_lora = pose_lora
        self.real_lora = real_lora
        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
            self.proj_lora_text = LoRALinearLayer(dim, dim, rank, network_alpha)
        else:
            # placeholders to keep code simpler; won't be used
            self.qkv_lora_text = None
            self.proj_lora_text = None

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        """
        attn: the DoubleStreamBlock
          - attn.img_mod -> Modulation block for images
          - attn.txt_mod -> Modulation block for text
          - attn.img_attn -> SelfAttention for images
          - attn.txt_attn -> SelfAttention for text
        img: [bs, (pose_seqlen + real_seqlen), dim]
        txt: [bs, 512, dim]
        vec: the 'real' image embedding
        pe: positional embedding for attention
        attention_kwargs includes:
          - image_sequence_len: the length of the real images
          - vec_pose: the embedding for the pose images
          - (optionally) text_lora: if you want a dynamic override
        """
        # 1) Parse out the real and pose lengths
        real_seqlen = attention_kwargs["image_sequence_len"]  # length of real image tokens
        pose_seqlen = img.shape[1] - real_seqlen
        vec_pose = attention_kwargs["vec_pose"]

        # If you want to allow runtime override for text_lora, do:
        # self.text_lora = attention_kwargs.get("text_lora", self.text_lora)

        # 2) Split pose tokens from real tokens
        pose_img = img[:, :pose_seqlen]    # [bs, pose_seqlen, dim]
        real_img = img[:, pose_seqlen:]    # [bs, real_seqlen, dim]

        # ---------------------------------------------------------------------
        # HELPER: compute ModulationOut with LoRA offset for Pose
        def pose_modulation_forward(mod_block: nn.Module, pose_vector: Tensor):
            """
            This replicates mod_block(vec_pose) but inserts a LoRA offset.
            - Normally: out = mod_block.lin(silu(vec))
            - We'll add pose_mod_lora(silu(vec_pose)) * lora_weight
            Then chunk the result into two ModulationOut sets of (shift, scale, gate).
            """
            base = mod_block.lin(nn.functional.silu(pose_vector))  # [bs, 6*dim]
            offset = self.pose_mod_lora(nn.functional.silu(pose_vector)) * self.lora_weight
            combined = base + offset  # [bs, 6*dim]
            # shape to [bs, 1, 6*dim], chunk into 2 sets of 3
            chunked = combined[:, None, :].chunk(6, dim=-1)
            return (
                ModulationOut(*chunked[:3]),  # (shift, scale, gate)
                ModulationOut(*chunked[3:])   # (shift, scale, gate)
            )

        # HELPER: compute ModulationOut for real (no LoRA offset)
        def real_modulation_forward(mod_block: nn.Module, real_vector: Tensor):
            return mod_block(real_vector)  # calls mod_block.forward()

        # HELPER: compute ModulationOut for text, optional LoRA
        def text_modulation_forward(mod_block: nn.Module, text_vector: Tensor):
            # Normal path: mod_block(text_vector)
            return mod_block(text_vector)

        # 3) Gather the modulation parameters
        # Pose uses separate LoRA-based mod
        pose_mod1, pose_mod2 = pose_modulation_forward(attn.img_mod, vec_pose)
        # Real uses normal mod
        real_mod1, real_mod2 = real_modulation_forward(attn.img_mod, vec)
        # Text uses normal mod or no difference in the code
        txt_mod1, txt_mod2 = text_modulation_forward(attn.txt_mod, vec_pose) # TODO this is changed after the first 6 runs

        # ---------------------------------------------------------------------
        # HELPER: apply QKV with LoRA offset
        def apply_qkv_with_lora(x: Tensor, attn_module: SelfAttention, lora_downup: LoRALinearLayer, do_lora: bool):
            """
            x: [bs, seq_len, dim] input tokens (after norm & shift/scale).
            attn_module.qkv: the base QKV linear
            lora_downup: the LoRA offset (qkv_lora_pose or qkv_lora_real or qkv_lora_text)
            """
            base_qkv = attn_module.qkv(x)  # [bs, seq_len, 3*dim]
            if do_lora:
                lora_qkv = lora_downup(x) * self.lora_weight
                return base_qkv + lora_qkv  # shape [bs, seq_len, 3*dim]
            else:
                return base_qkv



        # 4) Pre-Attention Norm + Shift/Scale for each stream
        # Pose
        pose_modulated = attn.img_norm1(pose_img)
        pose_modulated = (1 + pose_mod1.scale) * pose_modulated + pose_mod1.shift
        # Real
        real_modulated = attn.img_norm1(real_img)
        real_modulated = (1 + real_mod1.scale) * real_modulated + real_mod1.shift
        # Text
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # 5) QKV for each portion
        pose_qkv = apply_qkv_with_lora(pose_modulated, attn.img_attn, self.qkv_lora_pose, self.pose_lora)
        real_qkv = apply_qkv_with_lora(real_modulated, attn.img_attn, self.qkv_lora_real, self.real_lora)
        txt_qkv  = apply_qkv_with_lora(txt_modulated, attn.txt_attn, self.qkv_lora_text, self.text_lora)  # might have no LoRA if text_lora=False

        # 6) Rearrange and combine
        # shape for each qkv is: [bs, seq_len, 3*dim]
        bs, pose_len, _ = pose_qkv.shape
        _, real_len, _ = real_qkv.shape
        _, txt_len,  _ = txt_qkv.shape

        # Split into q, k, v (B L (3 H D)) -> (3, B, H, L, D)
        # using attn.num_heads / attn.head_dim from the block
        def split_qkv(x):
            q, k, v = rearrange(x, "B L (three H D) -> three B H L D", three=3, H=attn.num_heads, D=attn.head_dim)
            # attn.img_attn.norm or attn.txt_attn.norm typically normalizes q, k
            return q, k, v

        pose_q, pose_k, pose_v = split_qkv(pose_qkv)
        real_q, real_k, real_v = split_qkv(real_qkv)
        txt_q,  txt_k,  txt_v  = split_qkv(txt_qkv)

        # run the typical "attn.img_attn.norm" or "attn.txt_attn.norm" for each
        # (some implementations do a small pre-softmax normalization)
        # We'll do the same approach as your original code:
        pose_q, pose_k = attn.img_attn.norm(pose_q, pose_k, pose_v)
        real_q, real_k = attn.img_attn.norm(real_q, real_k, real_v)
        txt_q,  txt_k  = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # Now we concatenate all streams: text first, then pose, then real
        # (You can do any ordering you want, but let's keep text first for consistency.)
        q = torch.cat([txt_q, pose_q, real_q], dim=-2)  # along sequence dimension
        k = torch.cat([txt_k, pose_k, real_k], dim=-2)
        v = torch.cat([txt_v, pose_v, real_v], dim=-2)

        # 7) Run attention
        if self.real_image_active is None:
            if self.use_attn_mask: # when sampling, vec_pose is None
                attn_mask = build_mask(txt_len=512, pose_len=1024, img_len=1024, device=q.device) # TODO, seqlen still hardcode here!!
            else:
                attn_mask = None
        else:
            attn_mask = build_mask_batch(txt_len=512, pose_len=1024, img_len=1024, device=q.device, real_image_active=self.real_image_active)

        assert attn_mask is None, "attn_mask should be None"

        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask)  # shape [bs, (txt_len+pose_len+real_len), dim]

        # 8) Split the attention output back to text, pose, real
        txt_attn_out  = attn_out[:, :txt_len]
        pose_attn_out = attn_out[:, txt_len: txt_len + pose_len]
        real_attn_out = attn_out[:, txt_len + pose_len:]

        # 9) Residual + gating for Pose
        if self.pose_lora:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out) \
                                + pose_mod1.gate * self.proj_lora_pose(pose_attn_out) * self.lora_weight
        else:
            pose_img = pose_img + pose_mod1.gate * attn.img_attn.proj(pose_attn_out)
        # MLP part with second modulation
        pose_img = pose_img + pose_mod2.gate * attn.img_mlp(
            (1 + pose_mod2.scale) * attn.img_norm2(pose_img) + pose_mod2.shift
        )

        # 10) Residual + gating for Real
        if self.real_lora:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out) \
                                + real_mod1.gate * self.proj_lora_real(real_attn_out) * self.lora_weight
        else:
            real_img = real_img + real_mod1.gate * attn.img_attn.proj(real_attn_out)
        # MLP part with second modulation
        real_img = real_img + real_mod2.gate * attn.img_mlp(
            (1 + real_mod2.scale) * attn.img_norm2(real_img) + real_mod2.shift
        )

        # 11) Residual + gating for Text
        # If text_lora=False, no offset; if text_lora=True, use self.proj_lora_text
        if self.text_lora:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out) \
                      + txt_mod1.gate * self.proj_lora_text(txt_attn_out) * self.lora_weight
        else:
            txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn_out)

        txt = txt + txt_mod2.gate * attn.txt_mlp(
            (1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift
        )

        # 12) Re-combine pose and real into a single `img` again
        img = torch.cat([pose_img, real_img], dim=1)  # [bs, pose_len + real_len, dim]

        return img, txt



class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
        vec_pose: Tensor = None,
        image_sequence_len: int = None,
        vec_cond: Tensor = None,
        vec_pre_real: Tensor = None,
        cond_sequence_len: int = None,
        ar_attn_mask: Tensor | None = None,
        bbox_id_embeddings: Tensor | None = None,
        single_stream_attention_mask: Tensor | None = None,
        layer_index: int = None,
        mockpe: nn.Module = None,
        ####&&&&&& add for learnable pe &&&&&&####
        pe_learnable: bool = False,
        batch_shift: Tensor | None = None,
        batch_resize_scale: Tensor | None = None,
        batch_multi_person_last_indicator: Tensor | None = None,
        ####&&&&&& add for learnable pe &&&&&&####
        learned_shift_scale_ratio: float = None,
        ori_h: Tensor | None = None,
        ori_w: Tensor | None = None,
        version1: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            if version1:
                return self.processor.forward_version1(self, img=img, txt=txt, vec=vec, pe=pe,vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )
            else:
                return self.processor(self, img=img, txt=txt, vec=vec, pe=pe,vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )
        else:
            if version1:
                return self.processor.forward_version1(self, img=img, txt=txt, vec=vec, pe=pe, image_proj=image_proj, ip_scale=ip_scale, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )
            else:
                return self.processor(self, img=img, txt=txt, vec=vec, pe=pe, image_proj=image_proj, ip_scale=ip_scale, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )

class IPSingleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0
    ) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing
        ip_query = q
        ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_single_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)


        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

        attn_out = attn_1 + ip_scale * ip_attention

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        return out

class SingleStreamBlockLoraProcessorDualAttnMod(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha=None, lora_weight: float = 1, text_lora=False):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight
        self.use_attn_mask = False
        self.text_lora = text_lora

    def forward(
        self,
        attn: nn.Module,
        x: Tensor,                  # [batch_size, 512 + pose_len + image_sequence_len, dim]
        vec: Tensor,               # main modulation vector
        pe: Tensor,                # positional embeddings
        vec_pose: Tensor = None,   # optional pose modulation vector
        image_sequence_len: int = None
    ) -> Tensor:
        """
        We now assume the sequence is: [text, pose, image].
          - text always 512 tokens
          - next "pose_len" tokens are the pose portion
          - final "image_sequence_len" tokens are the image
        """
        B, L, D = x.shape
        txt_len = 512

        # 1) Get the "main" modulation from vec
        mod_main, _ = attn.modulation(vec)

        # 2) If no vec_pose, keep original logic on the entire x
        if vec_pose is None:
            x_mod = (1 + mod_main.scale) * attn.pre_norm(x) + mod_main.shift
            x_mod_txt = x_mod[:, :txt_len]
            x_mod_visual = x_mod[:, txt_len:]
            qkv, mlp = torch.split(
                attn.linear1(x_mod),
                [3 * attn.hidden_size, attn.mlp_hidden_dim],
                dim=-1
            )
            qkv_txt = qkv[:, :txt_len]
            qkv_visual = qkv[:, txt_len:]
            qkv_visual = qkv_visual + self.qkv_lora(x_mod_visual) * self.lora_weight
            if self.text_lora:
                qkv_txt = qkv_txt + self.qkv_lora(x_mod_txt) * self.lora_weight
            qkv = torch.cat((qkv_txt, qkv_visual), dim=1)

            q, k, v = rearrange(
                qkv, "B L (K H D) -> K B H L D",
                K=3, H=attn.num_heads, D=attn.head_dim
            )
            q, k = attn.norm(q, k, v)

            # Attention
            if self.use_attn_mask: # when sampling, vec_pose is None
                attn_mask = build_mask(txt_len=512, pose_len=1024, img_len=1024, device=q.device)
            else:
                attn_mask = None
            attn_1 = attention(q, k, v, pe=pe, attn_mask=attn_mask)

            attn_1_txt = attn_1[:, :txt_len]
            attn_1_visual = attn_1[:, txt_len:]

            # MLP
            mlp_act = attn.mlp_act(mlp)
            out_full = attn.linear2(torch.cat((attn_1, mlp_act), dim=2))
            out_full_txt = out_full[:, :txt_len]
            out_full_visual = out_full[:, txt_len:]

            out_full_visual = out_full_visual + self.proj_lora(torch.cat((attn_1_visual, mlp_act[:, txt_len:]), dim=2)) * self.lora_weight
            if self.text_lora:
                out_full_txt = out_full_txt + self.proj_lora(torch.cat((attn_1_txt, mlp_act[:, :txt_len]), dim=2)) * self.lora_weight
            out_full = torch.cat((out_full_txt, out_full_visual), dim=1)

            # Residual
            output = x + mod_main.gate * out_full
            return output

        # 3) Otherwise, split x into [text, pose, image]
        pose_len = L - txt_len - image_sequence_len
        x_txt = x[:, :txt_len]
        x_pose = x[:, txt_len : txt_len + pose_len]
        x_img = x[:, txt_len + pose_len : ]

        # 4) Get modulations from vec_pose
        mod_pose, _ = attn.modulation(vec_pose)

        # 5) Combine text mod from both main + pose
        combined_scale_txt = mod_main.scale #+ mod_pose.scale
        combined_shift_txt = mod_main.shift #+ mod_pose.shift
        combined_gate_txt  = mod_main.gate  #+ mod_pose.gate

        # 6) Apply combined mod to text
        x_txt_mod = (1 + combined_scale_txt) * attn.pre_norm(x_txt) + combined_shift_txt
        #    Pose portion uses only mod_pose
        x_pose_mod = (1 + mod_pose.scale) * attn.pre_norm(x_pose) + mod_pose.shift
        #    Image portion uses only mod_main
        x_img_mod = (1 + mod_main.scale) * attn.pre_norm(x_img) + mod_main.shift

        # 7) Re-concat for attention [txt, pose, img]
        x_mod = torch.cat((x_txt_mod, x_pose_mod, x_img_mod), dim=1)

        # Proceed with QKV + MLP as usual
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )
        # TODO check !!!!!!!!!!!!!!!!
        qkv_txt = qkv[:, :txt_len]
        qkv_visual = qkv[:, txt_len:]
        qkv_visual = qkv_visual + self.qkv_lora(x_mod[:, txt_len:]) * self.lora_weight
        if self.text_lora:
            qkv_txt = qkv_txt + self.qkv_lora(x_txt_mod) * self.lora_weight
        qkv = torch.cat((qkv_txt, qkv_visual), dim=1)
        # TODO check !!!!!!!!!!!!!!!!

        q, k, v = rearrange(
            qkv, "B L (K H D) -> K B H L D",
            K=3, H=attn.num_heads, D=attn.head_dim
        )
        q, k = attn.norm(q, k, v)

        # Attention
        if self.use_attn_mask:
            attn_mask = build_mask(txt_len=512, pose_len=pose_len, img_len=image_sequence_len, device=q.device)
        else:
            attn_mask = None
        attn_1 = attention(q, k, v, pe=pe, attn_mask=attn_mask)

        attn_1_txt = attn_1[:, :txt_len]
        attn_1_visual = attn_1[:, txt_len:]

        # MLP
        mlp_act = attn.mlp_act(mlp)
        out_full = attn.linear2(torch.cat((attn_1, mlp_act), dim=2))
        out_full_visual = out_full[:, txt_len:]
        out_full_txt = out_full[:, :txt_len]
        out_full_visual_lora = self.proj_lora(torch.cat((attn_1_visual, mlp_act[:, txt_len:]), dim=2)) * self.lora_weight
        out_full_visual = out_full_visual + out_full_visual_lora
        if self.text_lora:
            out_full_txt_lora = self.proj_lora(torch.cat((attn_1_txt, mlp_act[:, :txt_len]), dim=2)) * self.lora_weight
            out_full_txt = out_full_txt + out_full_txt_lora

        out_full = torch.cat((out_full_txt, out_full_visual), dim=1)

        # Split back to apply gating
        out_txt = out_full[:, :txt_len]
        out_pose = out_full[:, txt_len : txt_len + pose_len]
        out_img  = out_full[:, txt_len + pose_len : ]

        # Residual gating per segment
        #  Text uses combined gate
        out_txt = x_txt + combined_gate_txt * out_txt
        #  Pose uses pose gate
        out_pose = x_pose + mod_pose.gate * out_pose
        #  Image uses main gate
        out_img = x_img + mod_main.gate * out_img

        # Re-concat final
        output = torch.cat((out_txt, out_pose, out_img), dim=1)
        return output


class SingleStreamBlockLoraProcessorDualLoRA(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int = 4,
        network_alpha=None,
        lora_weight: float = 1,
        use_attn_mask: bool = False
    ):
        super().__init__()
        # We'll keep ONE LoRA for text (like your original).
        # We'll define TWO LoRAs for the image chunk:
        #   - pose (p) and main (r)
        self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_text = LoRALinearLayer(15360, dim, rank, network_alpha)

        # Pose LoRA
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_pose = LoRALinearLayer(15360, dim, rank, network_alpha)

        # Main (real) image LoRA
        self.qkv_lora_main = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora_main = LoRALinearLayer(15360, dim, rank, network_alpha)

        self.lora_weight = lora_weight
        self.use_attn_mask = use_attn_mask

    def forward(
        self,
        attn: nn.Module,
        x: torch.Tensor,         # [batch_size, 512 + pose_len + image_sequence_len, dim]
        vec: torch.Tensor,       # main modulation vector
        pe: torch.Tensor,        # positional embeddings
        vec_pose: torch.Tensor = None,   # optional pose modulation vector
        image_sequence_len: int = None
    ) -> torch.Tensor:
        """
        We assume the sequence is: [text, pose, image].
          text always 512 tokens
          next pose_len tokens are the pose portion
          final image_sequence_len tokens are the image portion
        If vec_pose is None => old logic on entire x with single main LoRA.
        Otherwise => split & use separate LoRAs for pose vs. main.
        """
        B, L, D = x.shape
        txt_len = 512

        # 1) main modulation from vec
        mod_main, _ = attn.modulation(vec)

        # 2) If no vec_pose, keep original logic on entire x with main LoRA
        if vec_pose is None:
            x_mod = (1 + mod_main.scale) * attn.pre_norm(x) + mod_main.shift
            qkv, mlp = torch.split(
                attn.linear1(x_mod),
                [3 * attn.hidden_size, attn.mlp_hidden_dim],
                dim=-1
            )
            # single LoRA path for everything
            qkv = qkv + self.qkv_lora_main(x_mod) * self.lora_weight

            q, k, v = rearrange(
                qkv, "B L (K H D) -> K B H L D",
                K=3, H=attn.num_heads, D=attn.head_dim
            )
            q, k = attn.norm(q, k, v)

            # Attn
            if self.use_attn_mask:
                # you might pick some default pose_len, etc. for sampling
                attn_mask = build_mask(txt_len=512, pose_len=1024, img_len=1024, device=q.device)
            else:
                attn_mask = None
            attn_1 = attention(q, k, v, pe=pe, attn_mask=attn_mask)

            # MLP
            mlp_act = attn.mlp_act(mlp)
            out_full = attn.linear2(torch.cat((attn_1, mlp_act), dim=2))
            # single projection LoRA
            out_full = out_full + self.proj_lora_main(torch.cat((attn_1, mlp_act), dim=2)) * self.lora_weight

            # Residual
            output = x + mod_main.gate * out_full
            return output

        # 3) Otherwise, we have a pose vector => split x into [text, pose, image]
        pose_len = L - txt_len - image_sequence_len
        x_txt = x[:, :txt_len]
        x_pose = x[:, txt_len : txt_len + pose_len]
        x_img = x[:, txt_len + pose_len : ]

        # 4) get mod_pose
        mod_pose, _ = attn.modulation(vec_pose)

        # 5) combine text mod from both main + pose
        comb_scale_txt = mod_main.scale + mod_pose.scale
        comb_shift_txt = mod_main.shift + mod_pose.shift
        comb_gate_txt  = mod_main.gate  + mod_pose.gate

        # 6) apply combined mod for text
        x_txt_mod = (1 + comb_scale_txt) * attn.pre_norm(x_txt) + comb_shift_txt
        # pose portion uses only mod_pose
        x_pose_mod = (1 + mod_pose.scale) * attn.pre_norm(x_pose) + mod_pose.shift
        # image portion uses only mod_main
        x_img_mod = (1 + mod_main.scale) * attn.pre_norm(x_img) + mod_main.shift

        # 7) Now apply different LoRA to each chunk
        #    - text => text LoRA
        #    - pose => pose LoRA
        #    - image => main LoRA
        # We'll do QKV separately, then cat
        txt_qkv = attn.linear1(x_txt_mod) + self.qkv_lora_text(x_txt_mod) * self.lora_weight
        pose_qkv = attn.linear1(x_pose_mod) + self.qkv_lora_pose(x_pose_mod) * self.lora_weight
        img_qkv  = attn.linear1(x_img_mod)  + self.qkv_lora_main(x_img_mod) * self.lora_weight

        # 8) re-concat these QKV chunks along dim=1 (the sequence dimension)
        # shape for each: [B, L_chunk, something]
        qkv = torch.cat([txt_qkv, pose_qkv, img_qkv], dim=1)
        mlp_len = attn.mlp_hidden_dim
        # qkv has size [B, total_len, 3*heads*D + mlp_len], presumably
        # We do a split:
        qkv_part, mlp_part = torch.split(
            qkv,
            [3 * attn.hidden_size, mlp_len],
            dim=-1
        )

        # 9) rearrange qkv => [K=3, B, H, L, D]
        q, k, v = rearrange(
            qkv_part, "B L (K H D) -> K B H L D",
            K=3, H=attn.num_heads, D=attn.head_dim
        )
        q, k = attn.norm(q, k, v)

        # 10) attention call
        if self.use_attn_mask:
            attn_mask = build_mask(
                txt_len=txt_len,
                pose_len=pose_len,
                img_len=image_sequence_len,
                device=q.device
            )
        else:
            attn_mask = None
        attn_1 = attention(q, k, v, pe=pe, attn_mask=attn_mask)

        # 11) MLP
        mlp_act = attn.mlp_act(mlp_part)
        out_full = attn.linear2(torch.cat((attn_1, mlp_act), dim=2))
        # Now we do different projection LoRA chunk for text, pose, image
        # But we can't trivially do it on separate chunks unless we re-split
        # Alternatively, do one big LoRA that doesn't differentiate.
        # If we *do* want separate MLP LoRA, we have to re-split out_full
        # same as we do for the residual gating. Let's do that:

        out_txt = out_full[:, :txt_len]
        out_pose = out_full[:, txt_len: txt_len + pose_len]
        out_img  = out_full[:, txt_len + pose_len:]

        # apply separate LoRA for each chunk in projection
        out_txt = out_txt + self.proj_lora_text(out_txt) * self.lora_weight
        out_pose = out_pose + self.proj_lora_pose(out_pose) * self.lora_weight
        out_img  = out_img + self.proj_lora_main(out_img) * self.lora_weight

        # re-cat them
        out_full = torch.cat((out_txt, out_pose, out_img), dim=1)

        # 12) residual gating
        # re-split original x into [text, pose, img] so we can do segment gating
        x_txt_orig = x_txt
        x_pose_orig = x_pose
        x_img_orig = x_img

        # text gating => combined
        out_txt = out_full[:, :txt_len]
        out_pose = out_full[:, txt_len : txt_len + pose_len]
        out_img  = out_full[:, txt_len + pose_len : ]

        # text => combined gate
        out_txt = x_txt_orig + comb_gate_txt * out_txt
        # pose => pose gate
        out_pose = x_pose_orig + mod_pose.gate * out_pose
        # image => main gate
        out_img = x_img_orig + mod_main.gate * out_img

        # final re-cat
        output = torch.cat((out_txt, out_pose, out_img), dim=1)
        return output


class SingleStreamBlockTripleLoraProcessorOneStream(nn.Module):
    """
    A Single-Stream block that processes [txt, pose] in a single self-attention pass,
    with separate LoRA offsets for text and pose QKV/proj, plus a LoRA offset
    for the entire (shift, scale, gate) vector of the pose modulation.

    Key points:
      - text_lora: bool controlling whether text QKV/proj gets LoRA.
      - Pose modulation has a LoRA offset for the entire 3*dim dimension.
      - We slice x => [txt, pose], do separate mod/gating, then combine for the single pass.
    """
    def __init__(
        self,
        dim: int,
        rank: int = 4,
        network_alpha: float = None,
        lora_weight: float = 1.0,
        text_lora: bool = False,
    ):
        super().__init__()

        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.text_lora = text_lora
        self.use_attn_mask = False # Kept for consistency, but `real_image_active` removed

        # Separate LoRA for QKV (text, pose)
        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        else:
            self.qkv_lora_text = None

        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

        # Separate LoRA for the projection (text, pose)
        # Note: input dim for proj_lora is 'hidden_size + mlp_hidden_dim' from linear2_in, which is 15360 if dim=3072, mlp_ratio=4
        # Calculation: dim + dim * mlp_ratio = dim * (1 + mlp_ratio) = 3072 * (1+4) = 15360
        input_proj_lora_dim = dim + int(dim * 4.0) # Assuming mlp_ratio=4.0 from SingleStreamBlock
        if self.text_lora:
            self.proj_lora_text = LoRALinearLayer(input_proj_lora_dim, dim, rank, network_alpha)
        else:
            self.proj_lora_text = None

        self.proj_lora_pose = LoRALinearLayer(input_proj_lora_dim, dim, rank, network_alpha)

        # LoRA offset for the pose modulation (dim -> 3*dim)
        # because for single-stream modulation we produce (shift, scale, gate) => 3*dim
        self.pose_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: the SingleStreamBlock, which has:
          - attn.modulation -> Modulation(hidden_size, double=False) => out shape 3*dim
          - attn.linear1 -> projects from dim -> (3*dim + mlp_hidden_dim)
          - attn.linear2 -> projects from (dim + mlp_hidden_dim) -> dim
          - attn.pre_norm -> layernorm
          - attn.norm -> QKNorm or something similar
        x: [bs, total_seq_len, hidden_size], with total_seq_len = text_len + pose_len
        vec: the embedding for real image (still passed, but not used for modulation or stream)
        pe: positional embedding used in attention
        kwargs includes:
          - text_sequence_len
          - vec_pose: the embedding for pose
        """

        # 1) Retrieve sequence lengths
        text_len = kwargs.get("text_sequence_len", 512)
        vec_pose = vec

        bs, total_seq_len, _ = x.shape
        # Now, total_seq_len only consists of text and pose lengths
        pose_len = total_seq_len - text_len

        # 2) Slice x into [txt, pose]
        txt_x  = x[:, :text_len]                               # [bs, text_len, dim]
        pose_x = x[:, text_len : text_len + pose_len]          # [bs, pose_len, dim]
        # Removed real_x

        # 3) Get modulation for text and pose
        #    - text uses "vec_pose" as per original user code (TODO comment)
        #    - pose uses "vec_pose" + LoRA offset for the entire 3*dim
        # Removed mod_real
        mod_text, _ = attn.modulation(vec_pose)       # (shift, scale, gate) for text

        # For pose, replicate the "modulation" forward pass with a LoRA offset.
        # The 'modulation' does: lin(silu(vec_pose)) => shape [bs, 3*dim].
        # We'll do that manually so we can add the offset from self.pose_mod_lora.
        pose_base = attn.modulation.lin(F.silu(vec_pose))  # shape [bs, 3*dim]
        pose_offset = self.pose_mod_lora(F.silu(vec_pose)) * self.lora_weight  # shape [bs, 3*dim]
        pose_combined = pose_base + pose_offset  # [bs, 3*dim]
        # Now chunk into shift, scale, gate
        # This is exactly how attn.modulation.forward() does it for single: chunk(3, dim=-1)
        # Then reshape to [bs,1,dim] each
        shift_p, scale_p, gate_p = torch.chunk(pose_combined, 3, dim=-1)
        shift_p = shift_p[:, None, :]  # [bs,1,dim]
        scale_p = scale_p[:, None, :]
        gate_p  = gate_p[:, None, :]
        # Put in a ModulationOut
        pose_mod = ModulationOut(shift_p, scale_p, gate_p)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            # from SingleStreamBlockProcessor:
            #   x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x,  mod_text)
        pose_modded = apply_mod(pose_x, pose_mod)
        # Removed real_modded

        # 5) Re-combine so we do a single pass
        x_mod = torch.cat([txt_modded, pose_modded], dim=1)  # [bs, total_seq_len, dim]

        # 6) Split attn.linear1 => QKV + MLP
        #    shape => [bs, total_seq_len, 3*dim + mlp_hidden_dim]
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # We'll now slice qkv by segment so we can apply separate LoRA offsets:
        txt_qkv  = qkv[:, :text_len]                # [bs, text_len, 3*dim]
        pose_qkv = qkv[:, text_len : text_len + pose_len]
        # Removed real_qkv

        def apply_lora_qkv(x_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool=True):
            # We do 'x_seg + lora(x_seg)' if do_lora, else just x_seg
            if do_lora:
                return lora_layer(x_seg) * self.lora_weight
            else:
                return torch.zeros(x_seg.shape[0], x_seg.shape[1], x_seg.shape[2]*3).to(x_seg.device).to(x_seg.dtype)

        # text LoRA only if self.text_lora is True
        txt_qkv  = txt_qkv + apply_lora_qkv(txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        # pose always gets LoRA (do_lora=True)
        pose_qkv = pose_qkv + apply_lora_qkv(pose_modded, self.qkv_lora_pose,  do_lora=True)
        # Removed real_qkv line

        # Recombine
        qkv = torch.cat([txt_qkv, pose_qkv], dim=1)  # [bs, total_seq_len, 3*dim]

        # 7) Rearrange =>  q, k, v
        q, k, v = rearrange(
            qkv,
            "B L (three H D) -> three B H L D",
            three=3,
            H=attn.num_heads,
            D=attn.head_dim
        )

        # attn.norm => QKNorm or similar that modifies q, k
        q, k = attn.norm(q, k, v)

        # 8) attention
        attn_out = attention(q, k, v, pe=pe)  # [bs, total_seq_len, dim]

        # 9) MLP part
        mlp_act = attn.mlp_act(mlp)  # [bs, total_seq_len, mlp_hidden_dim]
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)  # shape [bs, total_seq_len, dim + mlp_hidden_dim]
        base_output = attn.linear2(linear2_in)  # [bs, total_seq_len, dim]

        # We'll do separate LoRA for this projection as well:
        txt_lin2_in  = linear2_in[:, :text_len]
        pose_lin2_in = linear2_in[:, text_len : text_len + pose_len]
        # Removed real_lin2_in

        txt_lin2_out  = base_output[:, :text_len]
        pose_lin2_out = base_output[:, text_len : text_len + pose_len]
        # Removed real_lin2_out

        def apply_lora_proj(x_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool=True):
            if do_lora:
                return lora_layer(x_seg) * self.lora_weight
            else:
                return torch.zeros(x_seg.shape[0], x_seg.shape[1], 3072).to(x_seg.device).to(x_seg.dtype)

        # text -> only if text_lora
        txt_lin2_out  = txt_lin2_out  + apply_lora_proj(txt_lin2_in,  self.proj_lora_text, do_lora=self.text_lora)
        # pose -> always (do_lora=True)
        pose_lin2_out = pose_lin2_out + apply_lora_proj(pose_lin2_in, self.proj_lora_pose,  do_lora=True)
        # Removed real_lin2_out line

        # Reassemble final output
        output = torch.cat([txt_lin2_out, pose_lin2_out], dim=1)

        # 10) Residual with gating
        # We have separate gating from (mod_text, pose_mod)
        # so we slice output again by text, pose
        txt_out  = output[:, :text_len]
        pose_out = output[:, text_len : text_len + pose_len]
        # Removed real_out

        txt_gate  = mod_text.gate
        pose_gate = pose_mod.gate
        # Removed real_gate

        # Each is shape [bs,1,dim], broadcast across seq_len
        # final = x_seg + gate * out_seg
        txt_final  = txt_x  + txt_gate  * txt_out
        pose_final = pose_x + pose_gate * pose_out
        # Removed real_final

        final_output = torch.cat([txt_final, pose_final], dim=1)
        return final_output


class SingleStreamBlockTripleLoraProcessorAR(nn.Module):
    """
    A Single-Stream block that processes [txt, cond, pose, img] in a single self-attention pass.
    The 'cond' stream's LoRA application is controlled by the 'pose_lora' flag, making it
    behave identically to the pose stream, but with its own weights and conditioning vector.

    Key points:
      - text_lora, pose_lora, real_lora: Bools controlling QKV/proj LoRA.
      - The 'pose_lora' flag now controls LoRA for BOTH the 'cond' and 'pose' streams.
      - Cond and Pose modulations each have their own unconditional LoRA offset.
      - We slice x => [txt, cond, pose, img], do separate mod/gating, then combine.
    """
    def __init__(
        self,
        dim: int,
        rank=4,
        network_alpha=None,
        lora_weight=1.0,
        text_lora=False,
        pose_lora=True, # This flag now controls both cond and pose
        real_lora=False,
    ):
        super().__init__()

        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.text_lora = text_lora
        self.pose_lora = pose_lora # Controls both cond and pose
        self.real_lora = real_lora
        self.use_attn_mask = False
        self.real_image_active = None

        # --- QKV LoRA Layers (cond now has its own layer) ---
        self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # if self.text_lora:
        #     self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # else:
        #     self.qkv_lora_text = None

        self.qkv_lora_cond = LoRALinearLayer(dim, dim * 3, rank, network_alpha) # New layer for cond
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

        self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # if self.real_lora:
        #     self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # else:
        #     self.qkv_lora_real = None

        # --- Projection LoRA Layers ---
        proj_in_dim = 15360 # Placeholder: dim + mlp_hidden_dim
        self.proj_lora_text = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # if self.text_lora:
        #     self.proj_lora_text = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # else:
        #     self.proj_lora_text = None

        self.proj_lora_cond = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha) # New layer for cond
        self.proj_lora_pose = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)

        self.proj_lora_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # if self.real_lora:
        #     self.proj_lora_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # else:
        #     self.proj_lora_real = None

        # --- Modulation LoRA Layers (unconditionally applied) ---
        self.cond_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha) # New layer for cond
        self.pose_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: The SingleStreamBlock module.
        x: [bs, total_seq_len, hidden_size], with total_seq_len = txt + cond + pose + img
        vec: The embedding for real image.
        pe: Positional embedding for attention.
        kwargs includes:
          - text_sequence_len
          - cond_sequence_len
          - image_sequence_len
          - vec_cond: The embedding for the conditioning signal
          - vec_pose: The embedding for pose.
        """

        # 1) Retrieve sequence lengths and conditioning vectors
        text_len = kwargs.get("text_sequence_len", 512)
        cond_len = kwargs["cond_sequence_len"]
        image_seq_len = kwargs["image_sequence_len"]
        vec_cond = kwargs["vec_cond"]
        vec_pose = kwargs["vec_pose"]
        ar_attn_mask = kwargs["ar_attn_mask"] # can be None
        bbox_id_embeddings = kwargs.get("bbox_id_embeddings", None)
        mockpe = kwargs.get("mockpe", None)
        pe_learnable = kwargs.get("pe_learnable", False)
        batch_shift = kwargs.get("batch_shift", None)
        batch_resize_scale = kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = kwargs.get("learned_shift_scale_ratio", None)
        ori_h = kwargs.get("ori_h", None)
        ori_w = kwargs.get("ori_w", None)

        bs, total_seq_len, dim = x.shape
        pose_len = total_seq_len - text_len - cond_len - image_seq_len

        # 2) Slice x into [txt, cond, pose, real]
        txt_x  = x[:, :text_len]
        cond_x = x[:, text_len : text_len + cond_len]
        pose_x = x[:, text_len + cond_len : text_len + cond_len + pose_len]
        real_x = x[:, text_len + cond_len + pose_len :]

        # 3) Get modulation for each stream
        mod_text, _ = attn.modulation(vec_pose)
        mod_real, _ = attn.modulation(vec)

        def get_lora_modulation(mod_vec: Tensor, lora_layer: LoRALinearLayer) -> ModulationOut:
            base = attn.modulation.lin(F.silu(mod_vec))
            offset = lora_layer(F.silu(mod_vec)) * self.lora_weight
            combined = base + offset
            shift, scale, gate = torch.chunk(combined, 3, dim=-1)
            return ModulationOut(shift[:, None, :], scale[:, None, :], gate[:, None, :])

        cond_mod = get_lora_modulation(vec_cond, self.cond_mod_lora)
        pose_mod = get_lora_modulation(vec_pose, self.pose_mod_lora)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x, mod_text)
        cond_modded = apply_mod(cond_x, cond_mod)
        pose_modded = apply_mod(pose_x, pose_mod)
        real_modded = apply_mod(real_x, mod_real)

        # 5) Re-combine for a single pass
        x_mod = torch.cat([txt_modded, cond_modded, pose_modded, real_modded], dim=1)

        # 6) Split attn.linear1 => QKV + MLP
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # Slice QKV to apply separate LoRA offsets
        txt_qkv  = qkv[:, :text_len]
        cond_qkv = qkv[:, text_len : text_len + cond_len]
        pose_qkv = qkv[:, text_len + cond_len : text_len + cond_len + pose_len]
        real_qkv = qkv[:, text_len + cond_len + pose_len :]

        def apply_lora_qkv(qkv_seg: Tensor, x_mod_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return qkv_seg + lora_layer(x_mod_seg) * self.lora_weight
            return qkv_seg

        # Apply LoRA based on corresponding flags. Note cond uses pose_lora.
        txt_qkv  = apply_lora_qkv(txt_qkv,  txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        cond_qkv = apply_lora_qkv(cond_qkv, cond_modded, self.qkv_lora_cond, do_lora=self.pose_lora)
        pose_qkv = apply_lora_qkv(pose_qkv, pose_modded, self.qkv_lora_pose, do_lora=self.pose_lora)
        real_qkv = apply_lora_qkv(real_qkv, real_modded, self.qkv_lora_real, do_lora=self.real_lora)

        # Recombine QKV
        qkv = torch.cat([txt_qkv, cond_qkv, pose_qkv, real_qkv], dim=1)

        # 7) Rearrange to q, k, v
        q, k, v = rearrange(qkv, "B L (t H D) -> t B H L D", t=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert mockpe is not None, "mockpe is required when pe_learnable is True"
            assert learned_shift_scale_ratio is not None, "learned_shift_scale_ratio is required when pe_learnable is True"
            mock_shift_scale = mockpe(k, v) # [bs, 3, 1]
            mock_shift_scale = mock_shift_scale.squeeze(-1) # [bs, 3]
            shift_x_learned, shift_y_learned, scale_learned = mock_shift_scale[:, 0], mock_shift_scale[:, 1], mock_shift_scale[:, 2] # [bs]
            shift_learned = torch.stack([shift_x_learned, shift_y_learned], dim=1) # (B, 2)

            use_learned_shift_scale = torch.rand(()) <= learned_shift_scale_ratio  # scalar boolean -- switch once for whole batch
            if use_learned_shift_scale:
                batch_shift_to_use = shift_learned
                batch_resize_scale_to_use = scale_learned
            else:
                batch_shift_to_use = batch_shift
                batch_resize_scale_to_use = batch_resize_scale

            pe = adjust_condition_ids_inplace(batch_ids=pe, batch_multi_person_last_indicator=batch_multi_person_last_indicator, batch_shift=batch_shift_to_use, batch_resize_scale=batch_resize_scale_to_use, ori_h=ori_h, ori_w=ori_w)
            pe = EmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(pe)
        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 8) Attention
        attn_mask = ar_attn_mask # Placeholder for your mask generation logic
        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask, bbox_id_embeddings=bbox_id_embeddings)

        # 9) MLP part with separate projection LoRA
        mlp_act = attn.mlp_act(mlp)
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)
        base_output = attn.linear2(linear2_in)

        # Slice inputs and outputs for LoRA projection
        txt_lin2_in, cond_lin2_in, pose_lin2_in, real_lin2_in = torch.split(
            linear2_in, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )
        txt_base_out, cond_base_out, pose_base_out, real_base_out = torch.split(
            base_output, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )

        def apply_lora_proj(base_out_seg: Tensor, lin2_in_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return base_out_seg + lora_layer(lin2_in_seg) * self.lora_weight
            return base_out_seg

        # Apply projection LoRA. Note cond uses pose_lora.
        txt_lin2_out  = apply_lora_proj(txt_base_out,  txt_lin2_in,  self.proj_lora_text, self.text_lora)
        cond_lin2_out = apply_lora_proj(cond_base_out, cond_lin2_in, self.proj_lora_cond, self.pose_lora)
        pose_lin2_out = apply_lora_proj(pose_base_out, pose_lin2_in, self.proj_lora_pose, self.pose_lora)
        real_lin2_out = apply_lora_proj(real_base_out, real_lin2_in, self.proj_lora_real, self.real_lora)

        # Reassemble final output before residual
        output = torch.cat([txt_lin2_out, cond_lin2_out, pose_lin2_out, real_lin2_out], dim=1)

        # 10) Residual with separate gating
        txt_out, cond_out, pose_out, real_out = torch.split(
            output, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )

        txt_final  = txt_x  + mod_text.gate * txt_out
        cond_final = cond_x + cond_mod.gate * cond_out
        pose_final = pose_x + pose_mod.gate * pose_out
        real_final = real_x + mod_real.gate * real_out

        final_output = torch.cat([txt_final, cond_final, pose_final, real_final], dim=1)
        return final_output


class SingleStreamBlockTripleLoraProcessorAR_version2(nn.Module):
    """
    A Single-Stream block that processes [txt, cond, pose, img] in a single self-attention pass.
    The 'cond' stream's LoRA application is controlled by the 'pose_lora' flag, making it
    behave identically to the pose stream, but with its own weights and conditioning vector.

    Key points:
      - text_lora, pose_lora, real_lora: Bools controlling QKV/proj LoRA.
      - The 'pose_lora' flag now controls LoRA for BOTH the 'cond' and 'pose' streams.
      - Cond and Pose modulations each have their own unconditional LoRA offset.
      - We slice x => [txt, cond, pose, img], do separate mod/gating, then combine.
    """
    def __init__(
        self,
        dim: int,
        rank=4,
        network_alpha=None,
        lora_weight=1.0,
        text_lora=False,
        pose_lora=True, # This flag now controls both cond and pose
        pre_real_lora=False,
        real_lora=False,
    ):
        super().__init__()

        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.text_lora = text_lora
        self.pose_lora = pose_lora # Controls both cond and pose
        self.pre_real_lora = pre_real_lora
        self.real_lora = real_lora
        self.use_attn_mask = False
        self.real_image_active = None

        # --- QKV LoRA Layers (cond now has its own layer) ---
        self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # if self.text_lora:
        #     self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # else:
        #     self.qkv_lora_text = None

        self.qkv_lora_cond = LoRALinearLayer(dim, dim * 3, rank, network_alpha) # New layer for cond
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.qkv_lora_pre_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # if self.real_lora:
        #     self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # else:
        #     self.qkv_lora_real = None

        # --- Projection LoRA Layers ---
        proj_in_dim = 15360 # Placeholder: dim + mlp_hidden_dim
        self.proj_lora_text = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # if self.text_lora:
        #     self.proj_lora_text = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # else:
        #     self.proj_lora_text = None

        self.proj_lora_cond = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha) # New layer for cond
        self.proj_lora_pose = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        self.proj_lora_pre_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        self.proj_lora_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # if self.real_lora:
        #     self.proj_lora_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # else:
        #     self.proj_lora_real = None

        # --- Modulation LoRA Layers (unconditionally applied) ---
        self.cond_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha) # New layer for cond
        self.pose_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: The SingleStreamBlock module.
        x: [bs, total_seq_len, hidden_size], with total_seq_len = txt + cond + pose + img
        vec: The embedding for real image.
        pe: Positional embedding for attention.
        kwargs includes:
          - text_sequence_len
          - cond_sequence_len
          - image_sequence_len
          - vec_cond: The embedding for the conditioning signal
          - vec_pose: The embedding for pose.
        """

        # 1) Retrieve sequence lengths and conditioning vectors
        text_len = kwargs.get("text_sequence_len", 512)
        cond_len = kwargs["cond_sequence_len"]
        image_seq_len = kwargs["image_sequence_len"]
        pre_real_seq_len = image_seq_len
        vec_cond = kwargs["vec_cond"]
        vec_pose = kwargs["vec_pose"]
        vec_pre_real = kwargs["vec_pre_real"]
        assert vec_pre_real is not None, "vec_pre_real is required for version2"
        ar_attn_mask = kwargs["ar_attn_mask"] # can be None
        bbox_id_embeddings = kwargs.get("bbox_id_embeddings", None)
        assert bbox_id_embeddings is None, "bbox_id_embeddings is not used anymore"
        mockpe = kwargs.get("mockpe", None)
        pe_learnable = kwargs.get("pe_learnable", False)
        batch_shift = kwargs.get("batch_shift", None)
        batch_resize_scale = kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = kwargs.get("learned_shift_scale_ratio", None)
        ori_h = kwargs.get("ori_h", None)
        ori_w = kwargs.get("ori_w", None)

        bs, total_seq_len, dim = x.shape
        pose_len = total_seq_len - text_len - cond_len - image_seq_len - pre_real_seq_len

        # 2) Slice x into [txt, cond, pose, real]
        txt_x  = x[:, :text_len]
        cond_x = x[:, text_len : text_len + cond_len]
        pose_x = x[:, text_len + cond_len : text_len + cond_len + pose_len]
        pre_real_x = x[:, text_len + cond_len + pose_len : text_len + cond_len + pose_len + pre_real_seq_len]
        real_x = x[:, text_len + cond_len + pose_len + pre_real_seq_len :]

        # 3) Get modulation for each stream
        mod_text, _ = attn.modulation(vec_pose)
        mod_real, _ = attn.modulation(vec)
        mod_pre_real, _ = attn.modulation(vec_pre_real)

        def get_lora_modulation(mod_vec: Tensor, lora_layer: LoRALinearLayer) -> ModulationOut:
            base = attn.modulation.lin(F.silu(mod_vec))
            offset = lora_layer(F.silu(mod_vec)) * self.lora_weight
            combined = base + offset
            shift, scale, gate = torch.chunk(combined, 3, dim=-1)
            return ModulationOut(shift[:, None, :], scale[:, None, :], gate[:, None, :])

        cond_mod = get_lora_modulation(vec_cond, self.cond_mod_lora)
        pose_mod = get_lora_modulation(vec_pose, self.pose_mod_lora)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x, mod_text)
        cond_modded = apply_mod(cond_x, cond_mod)
        pose_modded = apply_mod(pose_x, pose_mod)
        real_modded = apply_mod(real_x, mod_real)
        pre_real_modded = apply_mod(pre_real_x, mod_pre_real)

        # 5) Re-combine for a single pass
        x_mod = torch.cat([txt_modded, cond_modded, pose_modded, pre_real_modded, real_modded], dim=1)

        # 6) Split attn.linear1 => QKV + MLP
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # Slice QKV to apply separate LoRA offsets
        txt_qkv  = qkv[:, :text_len]
        cond_qkv = qkv[:, text_len : text_len + cond_len]
        pose_qkv = qkv[:, text_len + cond_len : text_len + cond_len + pose_len]
        pre_real_qkv = qkv[:, text_len + cond_len + pose_len : text_len + cond_len + pose_len + pre_real_seq_len]
        real_qkv = qkv[:, text_len + cond_len + pose_len + pre_real_seq_len :]

        def apply_lora_qkv(qkv_seg: Tensor, x_mod_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return qkv_seg + lora_layer(x_mod_seg) * self.lora_weight
            return qkv_seg

        # Apply LoRA based on corresponding flags. Note cond uses pose_lora.
        txt_qkv  = apply_lora_qkv(txt_qkv,  txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        cond_qkv = apply_lora_qkv(cond_qkv, cond_modded, self.qkv_lora_cond, do_lora=self.pose_lora)
        pose_qkv = apply_lora_qkv(pose_qkv, pose_modded, self.qkv_lora_pose, do_lora=self.pose_lora)
        real_qkv = apply_lora_qkv(real_qkv, real_modded, self.qkv_lora_real, do_lora=self.real_lora)
        pre_real_qkv = apply_lora_qkv(pre_real_qkv, pre_real_modded, self.qkv_lora_pre_real, do_lora=self.pre_real_lora)

        # Recombine QKV
        qkv = torch.cat([txt_qkv, cond_qkv, pose_qkv, pre_real_qkv, real_qkv], dim=1)

        # 7) Rearrange to q, k, v
        q, k, v = rearrange(qkv, "B L (t H D) -> t B H L D", t=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert False, "pe_learnable is not used anymore"
        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 8) Attention
        attn_mask = ar_attn_mask # Placeholder for your mask generation logic
        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask, bbox_id_embeddings=bbox_id_embeddings)

        # 9) MLP part with separate projection LoRA
        mlp_act = attn.mlp_act(mlp)
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)
        base_output = attn.linear2(linear2_in)

        # Slice inputs and outputs for LoRA projection
        txt_lin2_in, cond_lin2_in, pose_lin2_in, pre_real_lin2_in, real_lin2_in = torch.split(
            linear2_in, [text_len, cond_len, pose_len, pre_real_seq_len, image_seq_len], dim=1
        )
        txt_base_out, cond_base_out, pose_base_out, pre_real_base_out, real_base_out = torch.split(
            base_output, [text_len, cond_len, pose_len, pre_real_seq_len, image_seq_len], dim=1
        )

        def apply_lora_proj(base_out_seg: Tensor, lin2_in_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return base_out_seg + lora_layer(lin2_in_seg) * self.lora_weight
            return base_out_seg

        # Apply projection LoRA. Note cond uses pose_lora.
        txt_lin2_out  = apply_lora_proj(txt_base_out,  txt_lin2_in,  self.proj_lora_text, self.text_lora)
        cond_lin2_out = apply_lora_proj(cond_base_out, cond_lin2_in, self.proj_lora_cond, self.pose_lora)
        pose_lin2_out = apply_lora_proj(pose_base_out, pose_lin2_in, self.proj_lora_pose, self.pose_lora)
        real_lin2_out = apply_lora_proj(real_base_out, real_lin2_in, self.proj_lora_real, self.real_lora)
        pre_real_lin2_out = apply_lora_proj(pre_real_base_out, pre_real_lin2_in, self.proj_lora_pre_real, self.pre_real_lora)

        # Reassemble final output before residual
        output = torch.cat([txt_lin2_out, cond_lin2_out, pose_lin2_out, pre_real_lin2_out, real_lin2_out], dim=1)

        # 10) Residual with separate gating
        txt_out, cond_out, pose_out, pre_real_out, real_out = torch.split(
            output, [text_len, cond_len, pose_len, pre_real_seq_len, image_seq_len], dim=1
        )

        txt_final  = txt_x  + mod_text.gate * txt_out
        cond_final = cond_x + cond_mod.gate * cond_out
        pose_final = pose_x + pose_mod.gate * pose_out
        real_final = real_x + mod_real.gate * real_out
        pre_real_final = pre_real_x + mod_pre_real.gate * pre_real_out

        final_output = torch.cat([txt_final, cond_final, pose_final, pre_real_final, real_final], dim=1)
        return final_output

    def forward_version1(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: The SingleStreamBlock module.
        x: [bs, total_seq_len, hidden_size], with total_seq_len = txt + cond + pose + img
        vec: The embedding for real image.
        pe: Positional embedding for attention.
        kwargs includes:
          - text_sequence_len
          - cond_sequence_len
          - image_sequence_len
          - vec_cond: The embedding for the conditioning signal
          - vec_pose: The embedding for pose.
        """

        # 1) Retrieve sequence lengths and conditioning vectors
        text_len = kwargs.get("text_sequence_len", 512)
        cond_len = kwargs["cond_sequence_len"]
        image_seq_len = kwargs["image_sequence_len"]
        vec_cond = kwargs["vec_cond"]
        vec_pose = kwargs["vec_pose"]
        ar_attn_mask = kwargs["ar_attn_mask"] # can be None
        bbox_id_embeddings = kwargs.get("bbox_id_embeddings", None)
        mockpe = kwargs.get("mockpe", None)
        pe_learnable = kwargs.get("pe_learnable", False)
        batch_shift = kwargs.get("batch_shift", None)
        batch_resize_scale = kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = kwargs.get("learned_shift_scale_ratio", None)
        ori_h = kwargs.get("ori_h", None)
        ori_w = kwargs.get("ori_w", None)

        bs, total_seq_len, dim = x.shape
        pose_len = total_seq_len - text_len - cond_len - image_seq_len

        # 2) Slice x into [txt, cond, pose, real]
        txt_x  = x[:, :text_len]
        cond_x = x[:, text_len : text_len + cond_len]
        pose_x = x[:, text_len + cond_len : text_len + cond_len + pose_len]
        real_x = x[:, text_len + cond_len + pose_len :]

        # 3) Get modulation for each stream
        mod_text, _ = attn.modulation(vec_pose)
        mod_real, _ = attn.modulation(vec)

        def get_lora_modulation(mod_vec: Tensor, lora_layer: LoRALinearLayer) -> ModulationOut:
            base = attn.modulation.lin(F.silu(mod_vec))
            offset = lora_layer(F.silu(mod_vec)) * self.lora_weight
            combined = base + offset
            shift, scale, gate = torch.chunk(combined, 3, dim=-1)
            return ModulationOut(shift[:, None, :], scale[:, None, :], gate[:, None, :])

        cond_mod = get_lora_modulation(vec_cond, self.cond_mod_lora)
        pose_mod = get_lora_modulation(vec_pose, self.pose_mod_lora)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x, mod_text)
        cond_modded = apply_mod(cond_x, cond_mod)
        pose_modded = apply_mod(pose_x, pose_mod)
        real_modded = apply_mod(real_x, mod_real)

        # 5) Re-combine for a single pass
        x_mod = torch.cat([txt_modded, cond_modded, pose_modded, real_modded], dim=1)

        # 6) Split attn.linear1 => QKV + MLP
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # Slice QKV to apply separate LoRA offsets
        txt_qkv  = qkv[:, :text_len]
        cond_qkv = qkv[:, text_len : text_len + cond_len]
        pose_qkv = qkv[:, text_len + cond_len : text_len + cond_len + pose_len]
        real_qkv = qkv[:, text_len + cond_len + pose_len :]

        def apply_lora_qkv(qkv_seg: Tensor, x_mod_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return qkv_seg + lora_layer(x_mod_seg) * self.lora_weight
            return qkv_seg

        # Apply LoRA based on corresponding flags. Note cond uses pose_lora.
        txt_qkv  = apply_lora_qkv(txt_qkv,  txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        cond_qkv = apply_lora_qkv(cond_qkv, cond_modded, self.qkv_lora_cond, do_lora=self.pose_lora)
        pose_qkv = apply_lora_qkv(pose_qkv, pose_modded, self.qkv_lora_pose, do_lora=self.pose_lora)
        real_qkv = apply_lora_qkv(real_qkv, real_modded, self.qkv_lora_real, do_lora=self.real_lora)

        # Recombine QKV
        qkv = torch.cat([txt_qkv, cond_qkv, pose_qkv, real_qkv], dim=1)

        # 7) Rearrange to q, k, v
        q, k, v = rearrange(qkv, "B L (t H D) -> t B H L D", t=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert mockpe is not None, "mockpe is required when pe_learnable is True"
            assert learned_shift_scale_ratio is not None, "learned_shift_scale_ratio is required when pe_learnable is True"
            mock_shift_scale = mockpe(k, v) # [bs, 3, 1]
            mock_shift_scale = mock_shift_scale.squeeze(-1) # [bs, 3]
            shift_x_learned, shift_y_learned, scale_learned = mock_shift_scale[:, 0], mock_shift_scale[:, 1], mock_shift_scale[:, 2] # [bs]
            shift_learned = torch.stack([shift_x_learned, shift_y_learned], dim=1) # (B, 2)

            use_learned_shift_scale = torch.rand(()) <= learned_shift_scale_ratio  # scalar boolean -- switch once for whole batch
            if use_learned_shift_scale:
                batch_shift_to_use = shift_learned
                batch_resize_scale_to_use = scale_learned
            else:
                batch_shift_to_use = batch_shift
                batch_resize_scale_to_use = batch_resize_scale

            pe = adjust_condition_ids_inplace(batch_ids=pe, batch_multi_person_last_indicator=batch_multi_person_last_indicator, batch_shift=batch_shift_to_use, batch_resize_scale=batch_resize_scale_to_use, ori_h=ori_h, ori_w=ori_w)
            pe = EmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(pe)
        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 8) Attention
        attn_mask = ar_attn_mask # Placeholder for your mask generation logic
        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask, bbox_id_embeddings=bbox_id_embeddings)

        # 9) MLP part with separate projection LoRA
        mlp_act = attn.mlp_act(mlp)
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)
        base_output = attn.linear2(linear2_in)

        # Slice inputs and outputs for LoRA projection
        txt_lin2_in, cond_lin2_in, pose_lin2_in, real_lin2_in = torch.split(
            linear2_in, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )
        txt_base_out, cond_base_out, pose_base_out, real_base_out = torch.split(
            base_output, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )

        def apply_lora_proj(base_out_seg: Tensor, lin2_in_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return base_out_seg + lora_layer(lin2_in_seg) * self.lora_weight
            return base_out_seg

        # Apply projection LoRA. Note cond uses pose_lora.
        txt_lin2_out  = apply_lora_proj(txt_base_out,  txt_lin2_in,  self.proj_lora_text, self.text_lora)
        cond_lin2_out = apply_lora_proj(cond_base_out, cond_lin2_in, self.proj_lora_cond, self.pose_lora)
        pose_lin2_out = apply_lora_proj(pose_base_out, pose_lin2_in, self.proj_lora_pose, self.pose_lora)
        real_lin2_out = apply_lora_proj(real_base_out, real_lin2_in, self.proj_lora_real, self.real_lora)

        # Reassemble final output before residual
        output = torch.cat([txt_lin2_out, cond_lin2_out, pose_lin2_out, real_lin2_out], dim=1)

        # 10) Residual with separate gating
        txt_out, cond_out, pose_out, real_out = torch.split(
            output, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )

        txt_final  = txt_x  + mod_text.gate * txt_out
        cond_final = cond_x + cond_mod.gate * cond_out
        pose_final = pose_x + pose_mod.gate * pose_out
        real_final = real_x + mod_real.gate * real_out

        final_output = torch.cat([txt_final, cond_final, pose_final, real_final], dim=1)
        return final_output


class SingleStreamBlockTripleLoraProcessorAR_version2_crossattn(nn.Module):
    """
    A Single-Stream block that processes [txt, cond, pose, img] in a single self-attention pass.
    The 'cond' stream's LoRA application is controlled by the 'pose_lora' flag, making it
    behave identically to the pose stream, but with its own weights and conditioning vector.

    Key points:
      - text_lora, pose_lora, real_lora: Bools controlling QKV/proj LoRA.
      - The 'pose_lora' flag now controls LoRA for BOTH the 'cond' and 'pose' streams.
      - Cond and Pose modulations each have their own unconditional LoRA offset.
      - We slice x => [txt, cond, pose, img], do separate mod/gating, then combine.
    """
    def __init__(
        self,
        dim: int,
        rank=4,
        network_alpha=None,
        lora_weight=1.0,
        text_lora=False,
        pose_lora=True, # This flag now controls both cond and pose
        real_lora=False,
    ):
        super().__init__()

        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.text_lora = text_lora
        self.pose_lora = pose_lora # Controls both cond and pose
        self.real_lora = real_lora
        self.use_attn_mask = False
        self.real_image_active = None

        # --- QKV LoRA Layers (cond now has its own layer) ---
        self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # if self.text_lora:
        #     self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # else:
        #     self.qkv_lora_text = None

        self.qkv_lora_cond = LoRALinearLayer(dim, dim * 3, rank, network_alpha) # New layer for cond
        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

        self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # if self.real_lora:
        #     self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # else:
        #     self.qkv_lora_real = None

        # --- Projection LoRA Layers ---
        proj_in_dim = 15360 # Placeholder: dim + mlp_hidden_dim
        self.proj_lora_text = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # if self.text_lora:
        #     self.proj_lora_text = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # else:
        #     self.proj_lora_text = None

        self.proj_lora_cond = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha) # New layer for cond
        self.proj_lora_pose = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)

        self.proj_lora_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # if self.real_lora:
        #     self.proj_lora_real = LoRALinearLayer(proj_in_dim, dim, rank, network_alpha)
        # else:
        #     self.proj_lora_real = None

        # --- Modulation LoRA Layers (unconditionally applied) ---
        self.cond_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha) # New layer for cond
        self.pose_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: The SingleStreamBlock module.
        x: [bs, total_seq_len, hidden_size], with total_seq_len = txt + cond + pose + img
        vec: The embedding for real image.
        pe: Positional embedding for attention.
        kwargs includes:
          - text_sequence_len
          - cond_sequence_len
          - image_sequence_len
          - vec_cond: The embedding for the conditioning signal
          - vec_pose: The embedding for pose.
        """

        # 1) Retrieve sequence lengths and conditioning vectors
        text_len = kwargs.get("text_sequence_len", 512)
        cond_len = kwargs["cond_sequence_len"]
        image_seq_len = kwargs["image_sequence_len"]
        pre_real_seq_len = image_seq_len
        vec_cond = kwargs["vec_cond"]
        vec_pose = kwargs["vec_pose"]
        vec_pre_real = kwargs["vec_pre_real"]
        assert vec_pre_real is not None, "vec_pre_real is required for version2"
        ar_attn_mask = kwargs["ar_attn_mask"] # can be None
        bbox_id_embeddings = kwargs.get("bbox_id_embeddings", None)
        assert bbox_id_embeddings is None, "bbox_id_embeddings is not used anymore"
        mockpe = kwargs.get("mockpe", None)
        pe_learnable = kwargs.get("pe_learnable", False)
        batch_shift = kwargs.get("batch_shift", None)
        batch_resize_scale = kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = kwargs.get("learned_shift_scale_ratio", None)
        ori_h = kwargs.get("ori_h", None)
        ori_w = kwargs.get("ori_w", None)

        bs, total_seq_len, dim = x.shape
        pose_len = total_seq_len - text_len - cond_len - image_seq_len - pre_real_seq_len

        # 2) Slice x into [txt, cond, pose, real]
        txt_x  = x[:, :text_len]
        cond_x = x[:, text_len : text_len + cond_len]
        pose_x = x[:, text_len + cond_len : text_len + cond_len + pose_len]
        pre_real_x = x[:, text_len + cond_len + pose_len : text_len + cond_len + pose_len + pre_real_seq_len]
        real_x = x[:, text_len + cond_len + pose_len + pre_real_seq_len :]

        # 3) Get modulation for each stream
        mod_text, _ = attn.modulation(vec_pose)
        mod_real, _ = attn.modulation(vec)
        mod_pre_real, _ = attn.modulation(vec_pre_real)

        def get_lora_modulation(mod_vec: Tensor, lora_layer: LoRALinearLayer) -> ModulationOut:
            base = attn.modulation.lin(F.silu(mod_vec))
            offset = lora_layer(F.silu(mod_vec)) * self.lora_weight
            combined = base + offset
            shift, scale, gate = torch.chunk(combined, 3, dim=-1)
            return ModulationOut(shift[:, None, :], scale[:, None, :], gate[:, None, :])

        cond_mod = get_lora_modulation(vec_cond, self.cond_mod_lora)
        pose_mod = get_lora_modulation(vec_pose, self.pose_mod_lora)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x, mod_text)
        cond_modded = apply_mod(cond_x, cond_mod)
        pose_modded = apply_mod(pose_x, pose_mod)
        real_modded = apply_mod(real_x, mod_real)
        pre_real_modded = apply_mod(pre_real_x, mod_pre_real)

        # 5) Re-combine for a single pass
        x_mod = torch.cat([txt_modded, cond_modded, pose_modded, pre_real_modded, real_modded], dim=1)

        # 6) Split attn.linear1 => QKV + MLP
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # Slice QKV to apply separate LoRA offsets
        txt_qkv  = qkv[:, :text_len]
        cond_qkv = qkv[:, text_len : text_len + cond_len]
        pose_qkv = qkv[:, text_len + cond_len : text_len + cond_len + pose_len]
        pre_real_qkv = qkv[:, text_len + cond_len + pose_len : text_len + cond_len + pose_len + pre_real_seq_len]
        real_qkv = qkv[:, text_len + cond_len + pose_len + pre_real_seq_len :]

        def apply_lora_qkv(qkv_seg: Tensor, x_mod_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return qkv_seg + lora_layer(x_mod_seg) * self.lora_weight
            return qkv_seg

        # Apply LoRA based on corresponding flags. Note cond uses pose_lora.
        txt_qkv  = apply_lora_qkv(txt_qkv,  txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        cond_qkv = apply_lora_qkv(cond_qkv, cond_modded, self.qkv_lora_cond, do_lora=self.pose_lora)
        pose_qkv = apply_lora_qkv(pose_qkv, pose_modded, self.qkv_lora_pose, do_lora=self.pose_lora)
        real_qkv = apply_lora_qkv(real_qkv, real_modded, self.qkv_lora_real, do_lora=self.real_lora)
        pre_real_qkv = apply_lora_qkv(pre_real_qkv, pre_real_modded, self.qkv_lora_real, do_lora=self.real_lora)

        # Recombine QKV
        qkv = torch.cat([txt_qkv, cond_qkv, pose_qkv, pre_real_qkv, real_qkv], dim=1)

        # 7) Rearrange to q, k, v
        q, k, v = rearrange(qkv, "B L (t H D) -> t B H L D", t=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert False, "pe_learnable is not used anymore"
        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 8) Attention
        attn_mask = ar_attn_mask # Placeholder for your mask generation logic
        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask, bbox_id_embeddings=bbox_id_embeddings)

        # 9) MLP part with separate projection LoRA
        mlp_act = attn.mlp_act(mlp)
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)
        base_output = attn.linear2(linear2_in)

        # Slice inputs and outputs for LoRA projection
        txt_lin2_in, cond_lin2_in, pose_lin2_in, pre_real_lin2_in, real_lin2_in = torch.split(
            linear2_in, [text_len, cond_len, pose_len, pre_real_seq_len, image_seq_len], dim=1
        )
        txt_base_out, cond_base_out, pose_base_out, pre_real_base_out, real_base_out = torch.split(
            base_output, [text_len, cond_len, pose_len, pre_real_seq_len, image_seq_len], dim=1
        )

        def apply_lora_proj(base_out_seg: Tensor, lin2_in_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return base_out_seg + lora_layer(lin2_in_seg) * self.lora_weight
            return base_out_seg

        # Apply projection LoRA. Note cond uses pose_lora.
        txt_lin2_out  = apply_lora_proj(txt_base_out,  txt_lin2_in,  self.proj_lora_text, self.text_lora)
        cond_lin2_out = apply_lora_proj(cond_base_out, cond_lin2_in, self.proj_lora_cond, self.pose_lora)
        pose_lin2_out = apply_lora_proj(pose_base_out, pose_lin2_in, self.proj_lora_pose, self.pose_lora)
        real_lin2_out = apply_lora_proj(real_base_out, real_lin2_in, self.proj_lora_real, self.real_lora)
        pre_real_lin2_out = apply_lora_proj(pre_real_base_out, pre_real_lin2_in, self.proj_lora_real, self.real_lora)

        # Reassemble final output before residual
        output = torch.cat([txt_lin2_out, cond_lin2_out, pose_lin2_out, pre_real_lin2_out, real_lin2_out], dim=1)

        # 10) Residual with separate gating
        txt_out, cond_out, pose_out, pre_real_out, real_out = torch.split(
            output, [text_len, cond_len, pose_len, pre_real_seq_len, image_seq_len], dim=1
        )

        txt_final  = txt_x  + mod_text.gate * txt_out
        cond_final = cond_x + cond_mod.gate * cond_out
        pose_final = pose_x + pose_mod.gate * pose_out
        real_final = real_x + mod_real.gate * real_out
        pre_real_final = pre_real_x  + mod_pre_real.gate * pre_real_out

        final_output = torch.cat([txt_final, cond_final, pose_final, pre_real_final, real_final], dim=1)
        return final_output


    def forward_version1(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: The SingleStreamBlock module.
        x: [bs, total_seq_len, hidden_size], with total_seq_len = txt + cond + pose + img
        vec: The embedding for real image.
        pe: Positional embedding for attention.
        kwargs includes:
          - text_sequence_len
          - cond_sequence_len
          - image_sequence_len
          - vec_cond: The embedding for the conditioning signal
          - vec_pose: The embedding for pose.
        """

        # 1) Retrieve sequence lengths and conditioning vectors
        text_len = kwargs.get("text_sequence_len", 512)
        cond_len = kwargs["cond_sequence_len"]
        image_seq_len = kwargs["image_sequence_len"]
        vec_cond = kwargs["vec_cond"]
        vec_pose = kwargs["vec_pose"]
        ar_attn_mask = kwargs["ar_attn_mask"] # can be None
        bbox_id_embeddings = kwargs.get("bbox_id_embeddings", None)
        mockpe = kwargs.get("mockpe", None)
        pe_learnable = kwargs.get("pe_learnable", False)
        batch_shift = kwargs.get("batch_shift", None)
        batch_resize_scale = kwargs.get("batch_resize_scale", None)
        batch_multi_person_last_indicator = kwargs.get("batch_multi_person_last_indicator", None)
        learned_shift_scale_ratio = kwargs.get("learned_shift_scale_ratio", None)
        ori_h = kwargs.get("ori_h", None)
        ori_w = kwargs.get("ori_w", None)

        bs, total_seq_len, dim = x.shape
        pose_len = total_seq_len - text_len - cond_len - image_seq_len

        # 2) Slice x into [txt, cond, pose, real]
        txt_x  = x[:, :text_len]
        cond_x = x[:, text_len : text_len + cond_len]
        pose_x = x[:, text_len + cond_len : text_len + cond_len + pose_len]
        real_x = x[:, text_len + cond_len + pose_len :]

        # 3) Get modulation for each stream
        mod_text, _ = attn.modulation(vec_pose)
        mod_real, _ = attn.modulation(vec)

        def get_lora_modulation(mod_vec: Tensor, lora_layer: LoRALinearLayer) -> ModulationOut:
            base = attn.modulation.lin(F.silu(mod_vec))
            offset = lora_layer(F.silu(mod_vec)) * self.lora_weight
            combined = base + offset
            shift, scale, gate = torch.chunk(combined, 3, dim=-1)
            return ModulationOut(shift[:, None, :], scale[:, None, :], gate[:, None, :])

        cond_mod = get_lora_modulation(vec_cond, self.cond_mod_lora)
        pose_mod = get_lora_modulation(vec_pose, self.pose_mod_lora)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x, mod_text)
        cond_modded = apply_mod(cond_x, cond_mod)
        pose_modded = apply_mod(pose_x, pose_mod)
        real_modded = apply_mod(real_x, mod_real)

        # 5) Re-combine for a single pass
        x_mod = torch.cat([txt_modded, cond_modded, pose_modded, real_modded], dim=1)

        # 6) Split attn.linear1 => QKV + MLP
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # Slice QKV to apply separate LoRA offsets
        txt_qkv  = qkv[:, :text_len]
        cond_qkv = qkv[:, text_len : text_len + cond_len]
        pose_qkv = qkv[:, text_len + cond_len : text_len + cond_len + pose_len]
        real_qkv = qkv[:, text_len + cond_len + pose_len :]

        def apply_lora_qkv(qkv_seg: Tensor, x_mod_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return qkv_seg + lora_layer(x_mod_seg) * self.lora_weight
            return qkv_seg

        # Apply LoRA based on corresponding flags. Note cond uses pose_lora.
        txt_qkv  = apply_lora_qkv(txt_qkv,  txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        cond_qkv = apply_lora_qkv(cond_qkv, cond_modded, self.qkv_lora_cond, do_lora=self.pose_lora)
        pose_qkv = apply_lora_qkv(pose_qkv, pose_modded, self.qkv_lora_pose, do_lora=self.pose_lora)
        real_qkv = apply_lora_qkv(real_qkv, real_modded, self.qkv_lora_real, do_lora=self.real_lora)

        # Recombine QKV
        qkv = torch.cat([txt_qkv, cond_qkv, pose_qkv, real_qkv], dim=1)

        # 7) Rearrange to q, k, v
        q, k, v = rearrange(qkv, "B L (t H D) -> t B H L D", t=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)


        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####
        if pe_learnable:
            assert mockpe is not None, "mockpe is required when pe_learnable is True"
            assert learned_shift_scale_ratio is not None, "learned_shift_scale_ratio is required when pe_learnable is True"
            mock_shift_scale = mockpe(k, v) # [bs, 3, 1]
            mock_shift_scale = mock_shift_scale.squeeze(-1) # [bs, 3]
            shift_x_learned, shift_y_learned, scale_learned = mock_shift_scale[:, 0], mock_shift_scale[:, 1], mock_shift_scale[:, 2] # [bs]
            shift_learned = torch.stack([shift_x_learned, shift_y_learned], dim=1) # (B, 2)

            use_learned_shift_scale = torch.rand(()) <= learned_shift_scale_ratio  # scalar boolean -- switch once for whole batch
            if use_learned_shift_scale:
                batch_shift_to_use = shift_learned
                batch_resize_scale_to_use = scale_learned
            else:
                batch_shift_to_use = batch_shift
                batch_resize_scale_to_use = batch_resize_scale

            pe = adjust_condition_ids_inplace(batch_ids=pe, batch_multi_person_last_indicator=batch_multi_person_last_indicator, batch_shift=batch_shift_to_use, batch_resize_scale=batch_resize_scale_to_use, ori_h=ori_h, ori_w=ori_w)
            pe = EmbedND(dim=128, theta=10_000, axes_dim=[16, 56, 56])(pe)
        ####&&&&&& add for learnable pe -- after getting k and v from generating sequence &&&&&&####

        # 8) Attention
        attn_mask = ar_attn_mask # Placeholder for your mask generation logic
        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask, bbox_id_embeddings=bbox_id_embeddings)

        # 9) MLP part with separate projection LoRA
        mlp_act = attn.mlp_act(mlp)
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)
        base_output = attn.linear2(linear2_in)

        # Slice inputs and outputs for LoRA projection
        txt_lin2_in, cond_lin2_in, pose_lin2_in, real_lin2_in = torch.split(
            linear2_in, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )
        txt_base_out, cond_base_out, pose_base_out, real_base_out = torch.split(
            base_output, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )

        def apply_lora_proj(base_out_seg: Tensor, lin2_in_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool):
            if do_lora:
                return base_out_seg + lora_layer(lin2_in_seg) * self.lora_weight
            return base_out_seg

        # Apply projection LoRA. Note cond uses pose_lora.
        txt_lin2_out  = apply_lora_proj(txt_base_out,  txt_lin2_in,  self.proj_lora_text, self.text_lora)
        cond_lin2_out = apply_lora_proj(cond_base_out, cond_lin2_in, self.proj_lora_cond, self.pose_lora)
        pose_lin2_out = apply_lora_proj(pose_base_out, pose_lin2_in, self.proj_lora_pose, self.pose_lora)
        real_lin2_out = apply_lora_proj(real_base_out, real_lin2_in, self.proj_lora_real, self.real_lora)

        # Reassemble final output before residual
        output = torch.cat([txt_lin2_out, cond_lin2_out, pose_lin2_out, real_lin2_out], dim=1)

        # 10) Residual with separate gating
        txt_out, cond_out, pose_out, real_out = torch.split(
            output, [text_len, cond_len, pose_len, image_seq_len], dim=1
        )

        txt_final  = txt_x  + mod_text.gate * txt_out
        cond_final = cond_x + cond_mod.gate * cond_out
        pose_final = pose_x + pose_mod.gate * pose_out
        real_final = real_x + mod_real.gate * real_out

        final_output = torch.cat([txt_final, cond_final, pose_final, real_final], dim=1)
        return final_output


class SingleStreamBlockTripleLoraProcessor(nn.Module):
    """
    A Single-Stream block that processes [txt, pose, img] in a single self-attention pass,
    with separate LoRA offsets for text, pose, and image QKV/proj, plus a LoRA offset
    for the entire (shift, scale, gate) vector of the pose modulation.

    Key points:
      - text_lora: bool controlling whether text QKV/proj gets LoRA.
      - Pose modulation has a LoRA offset for the entire 3*hidden_size dimension.
      - Real image modulation is unchanged (no LoRA).
      - We slice x => [txt, pose, img], do separate mod/gating, then combine for the single pass.
    """
    def __init__(
        self,
        dim: int, # TODO remember the arg name is changed
        rank=4,
        network_alpha=None,
        lora_weight=1.0,
        text_lora=False,
        pose_lora=False,
        real_lora=False,
    ):
        super().__init__()

        self.rank = rank
        self.network_alpha = network_alpha
        self.lora_weight = lora_weight
        self.text_lora = text_lora
        self.pose_lora = pose_lora
        self.real_lora = real_lora
        self.use_attn_mask = False
        self.real_image_active = None
        # Separate LoRA for QKV (text, pose, real)
        if self.text_lora:
            self.qkv_lora_text = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        else:
            self.qkv_lora_text = None

        self.qkv_lora_pose = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

        if self.real_lora:
            self.qkv_lora_real = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        else:
            self.qkv_lora_real = None

        # Separate LoRA for the projection (text, pose, real)
        if self.text_lora:
            self.proj_lora_text = LoRALinearLayer(15360, dim, rank, network_alpha)
        else:
            self.proj_lora_text = None

        self.proj_lora_pose = LoRALinearLayer(15360, dim, rank, network_alpha)

        if self.real_lora:
            self.proj_lora_real = LoRALinearLayer(15360, dim, rank, network_alpha) # 15360 = 5x3072
        else:
            self.proj_lora_real = None


        # LoRA offset for the pose modulation (dim -> 3*dim)
        # because for single-stream modulation we produce (shift, scale, gate) => 3*dim
        self.pose_mod_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:
        """
        attn: the SingleStreamBlock, which has:
          - attn.modulation -> Modulation(hidden_size, double=False) => out shape 3*dim
          - attn.linear1 -> projects from dim -> (3*dim + mlp_hidden_dim)
          - attn.linear2 -> projects from (dim + mlp_hidden_dim) -> dim
          - attn.pre_norm -> layernorm
          - attn.norm -> QKNorm or something similar
        x: [bs, total_seq_len, hidden_size], with total_seq_len = text_len + pose_len + image_len
        vec: the embedding for real image
        pe: positional embedding used in attention
        kwargs includes:
          - text_sequence_len
          - image_sequence_len
          - vec_pose: the embedding for pose
        """

        # 1) Retrieve sequence lengths
        text_len = kwargs.get("text_sequence_len", 512)
        image_seq_len = kwargs["image_sequence_len"]
        vec_pose = kwargs["vec_pose"]

        bs, total_seq_len, _ = x.shape
        pose_len = total_seq_len - text_len - image_seq_len

        # 2) Slice x into [txt, pose, real]
        txt_x  = x[:, :text_len]                               # [bs, text_len, dim]
        pose_x = x[:, text_len : text_len + pose_len]          # [bs, pose_len, dim]
        real_x = x[:, text_len + pose_len : text_len + pose_len + image_seq_len]

        # 3) Get modulation for text, pose, real
        #    - text uses "vec" (or possibly a text vec?), here we'll just do the same as real_x
        #    - pose uses "vec_pose" + LoRA offset for the entire 3*dim
        #    - real uses normal "vec"
        mod_text, _ = attn.modulation(vec_pose)       # (shift, scale, gate) for text TODO this is changed after the first 6 runs
        mod_real, _ = attn.modulation(vec)       # (shift, scale, gate) for real

        # For pose, replicate the "modulation" forward pass with a LoRA offset.
        # The 'modulation' does: lin(silu(vec_pose)) => shape [bs, 3*dim].
        # We'll do that manually so we can add the offset from self.pose_mod_lora.
        pose_base = attn.modulation.lin(F.silu(vec_pose))  # shape [bs, 3*dim]
        pose_offset = self.pose_mod_lora(F.silu(vec_pose)) * self.lora_weight  # shape [bs, 3*dim]
        pose_combined = pose_base + pose_offset  # [bs, 3*dim]
        # Now chunk into shift, scale, gate
        # This is exactly how attn.modulation.forward() does it for single: chunk(3, dim=-1)
        # Then reshape to [bs,1,dim] each
        shift_p, scale_p, gate_p = torch.chunk(pose_combined, 3, dim=-1)
        shift_p = shift_p[:, None, :]  # [bs,1,dim]
        scale_p = scale_p[:, None, :]
        gate_p  = gate_p[:, None, :]
        # Put in a ModulationOut
        pose_mod = ModulationOut(shift_p, scale_p, gate_p)

        # 4) Pre-norm + scale/shift for each segment
        def apply_mod(x_seg: Tensor, mod_seg: ModulationOut) -> Tensor:
            # from SingleStreamBlockProcessor:
            #   x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
            return (1 + mod_seg.scale) * attn.pre_norm(x_seg) + mod_seg.shift

        txt_modded  = apply_mod(txt_x,  mod_text)
        pose_modded = apply_mod(pose_x, pose_mod)
        real_modded = apply_mod(real_x, mod_real)

        # 5) Re-combine so we do a single pass
        x_mod = torch.cat([txt_modded, pose_modded, real_modded], dim=1)  # [bs, total_seq_len, dim]

        # 6) Split attn.linear1 => QKV + MLP
        #    shape => [bs, total_seq_len, 3*dim + mlp_hidden_dim]
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        # We'll now slice qkv by segment so we can apply separate LoRA offsets:
        txt_qkv  = qkv[:, :text_len]                # [bs, text_len, 3*dim]
        pose_qkv = qkv[:, text_len : text_len + pose_len]
        real_qkv = qkv[:, text_len + pose_len :]

        def apply_lora_qkv(x_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool=True):
            # We do 'x_seg + lora(x_seg)' if do_lora, else just x_seg
            if do_lora:
                return lora_layer(x_seg) * self.lora_weight
            else:
                return torch.zeros(x_seg.shape[0], x_seg.shape[1], x_seg.shape[2]*3).to(x_seg.device).to(x_seg.dtype)

        # text LoRA only if self.text_lora is True
        txt_qkv  = txt_qkv + apply_lora_qkv(txt_modded,  self.qkv_lora_text, do_lora=self.text_lora)
        # pose always gets LoRA
        pose_qkv = pose_qkv + apply_lora_qkv(pose_modded, self.qkv_lora_pose,  do_lora=self.pose_lora)
        # real always gets LoRA
        real_qkv = real_qkv + apply_lora_qkv(real_modded, self.qkv_lora_real,  do_lora=self.real_lora)

        # Recombine
        qkv = torch.cat([txt_qkv, pose_qkv, real_qkv], dim=1)  # [bs, total_seq_len, 3*dim]

        # 7) Rearrange =>  q, k, v
        q, k, v = rearrange(
            qkv,
            "B L (three H D) -> three B H L D",
            three=3,
            H=attn.num_heads,
            D=attn.head_dim
        )

        # attn.norm => QKNorm or similar that modifies q, k
        q, k = attn.norm(q, k, v)

        # 8) attention
        if self.real_image_active is None:
            if self.use_attn_mask: # when sampling, vec_pose is None
                attn_mask = build_mask(txt_len=512, pose_len=1024, img_len=1024, device=q.device) # TODO, seqlen still hardcode here!!
            else:
                attn_mask = None
        else:
            attn_mask = build_mask_batch(txt_len=512, pose_len=1024, img_len=1024, device=q.device, real_image_active=self.real_image_active)


        assert attn_mask is None, "attn_mask should be None"

        attn_out = attention(q, k, v, pe=pe, attn_mask=attn_mask)  # [bs, total_seq_len, dim]

        # 9) MLP part
        mlp_act = attn.mlp_act(mlp)  # [bs, total_seq_len, mlp_hidden_dim]
        linear2_in = torch.cat((attn_out, mlp_act), dim=2)  # shape [bs, total_seq_len, dim + mlp_hidden_dim]
        base_output = attn.linear2(linear2_in)  # [bs, total_seq_len, dim]

        # We'll do separate LoRA for this projection as well:
        txt_lin2_in  = linear2_in[:, :text_len]
        pose_lin2_in = linear2_in[:, text_len : text_len + pose_len]
        real_lin2_in = linear2_in[:, text_len + pose_len :]

        txt_lin2_out  = base_output[:, :text_len]
        pose_lin2_out = base_output[:, text_len : text_len + pose_len]
        real_lin2_out = base_output[:, text_len + pose_len :]

        def apply_lora_proj(x_seg: Tensor, lora_layer: LoRALinearLayer, do_lora: bool=True):
            if do_lora:
                return lora_layer(x_seg) * self.lora_weight
            else:
                return torch.zeros(x_seg.shape[0], x_seg.shape[1], 3072).to(x_seg.device).to(x_seg.dtype)

        # text -> only if text_lora
        txt_lin2_out  = txt_lin2_out  + apply_lora_proj(txt_lin2_in,  self.proj_lora_text, do_lora=self.text_lora)
        # pose -> always
        pose_lin2_out = pose_lin2_out + apply_lora_proj(pose_lin2_in, self.proj_lora_pose,  do_lora=self.pose_lora)
        # real -> always
        real_lin2_out = real_lin2_out + apply_lora_proj(real_lin2_in, self.proj_lora_real,  do_lora=self.real_lora)

        # Reassemble final
        output = torch.cat([txt_lin2_out, pose_lin2_out, real_lin2_out], dim=1)

        # 10) Residual with gating
        # We have separate gating from (mod_text, pose_mod, mod_real)
        # so we slice output again by text, pose, real
        txt_out  = output[:, :text_len]
        pose_out = output[:, text_len : text_len + pose_len]
        real_out = output[:, text_len + pose_len :]

        txt_gate  = mod_text.gate
        pose_gate = pose_mod.gate
        real_gate = mod_real.gate

        # Each is shape [bs,1,dim], broadcast across seq_len
        # final = x_seg + gate * out_seg
        txt_final  = txt_x  + txt_gate  * txt_out
        pose_final = pose_x + pose_gate * pose_out
        real_final = real_x + real_gate * real_out

        final_output = torch.cat([txt_final, pose_final, real_final], dim=1)
        return final_output


class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, vec_pose: Tensor = None, image_sequence_len: int = None) -> Tensor:
        # print("x shape in single stream block lora processor", x.shape)

        # mod, _ = attn.modulation(vec)
        # x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        # qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        # qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        # q, k = attn.norm(q, k, v)

        # # compute attention
        # attn_1 = attention(q, k, v, pe=pe)

        # # compute activation in mlp stream, cat again and run second linear layer
        # output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        # output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        # output = x + mod.gate * output
        # return output
        B, L, D = x.shape
        txt_len = 512

        # -- 1) Always get the "main" modulation from vec
        mod_main, _ = attn.modulation(vec)

        # -- 2) If vec_pose is None, keep the old logic exactly
        if vec_pose is None:
            # Original un-splitted approach on the entire sequence x
            x_mod = (1 + mod_main.scale) * attn.pre_norm(x) + mod_main.shift

            # Split into QKV vs MLP chunk
            qkv, mlp = torch.split(
                attn.linear1(x_mod),
                [3 * attn.hidden_size, attn.mlp_hidden_dim],
                dim=-1
            )

            # LoRA contribution
            qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

            # Reshape into Q, K, V
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                                K=3, H=attn.num_heads, D=attn.head_dim)
            q, k = attn.norm(q, k, v)

            # Attention
            attn_1 = attention(q, k, v, pe=pe)

            # MLP path
            mlp_act = attn.mlp_act(mlp)  # activation on MLP chunk
            out_full = attn.linear2(torch.cat((attn_1, mlp_act), dim=2))
            out_full = out_full + self.proj_lora(torch.cat((attn_1, mlp_act), dim=2)) * self.lora_weight

            # Residual: single gate across entire sequence
            output = x + mod_main.gate * out_full
            return output
        # -- 3) Otherwise, we do the "splitted" approach (text + image + pose)

        # a) Split x into three parts
        x_txt = x[:, :txt_len]  # text portion
        x_img = x[:, txt_len : txt_len + image_sequence_len]  # image
        x_pose = x[:, txt_len + image_sequence_len :]         # pose portion

        # b) Get the separate modulations from vec_pose
        mod_pose, _ = attn.modulation(vec_pose)

        # c) Combine text mod from both main + pose
        combined_scale_txt = mod_main.scale + mod_pose.scale
        combined_shift_txt = mod_main.shift + mod_pose.shift
        combined_gate_txt  = mod_main.gate  + mod_pose.gate

        # d) Apply to each segment
        #    Text portion uses combined (scale, shift), etc.
        x_txt_mod = (1 + combined_scale_txt) * attn.pre_norm(x_txt) + combined_shift_txt

        #    Image portion uses only mod_main
        x_img_mod = (1 + mod_main.scale) * attn.pre_norm(x_img) + mod_main.shift

        #    Pose portion uses only mod_pose
        x_pose_mod = (1 + mod_pose.scale) * attn.pre_norm(x_pose) + mod_pose.shift

        # e) Concatenate back into a single modded sequence
        x_mod = torch.cat((x_txt_mod, x_img_mod, x_pose_mod), dim=1)

        # -- 4) Proceed with QKV + MLP as usual
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                            K=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)

        # Attention
        attn_1 = attention(q, k, v, pe=pe)

        # MLP path
        mlp_act = attn.mlp_act(mlp)
        out_full = attn.linear2(torch.cat((attn_1, mlp_act), dim=2))
        out_full = out_full + self.proj_lora(torch.cat((attn_1, mlp_act), dim=2)) * self.lora_weight

        # Now we have out_full for the entire sequence [txt + img + pose]
        # -- 5) Split it back to apply gating portion-by-portion
        out_txt = out_full[:, :txt_len]
        out_img = out_full[:, txt_len : txt_len + image_sequence_len]
        out_pose = out_full[:, txt_len + image_sequence_len :]

        # The original x was also in three parts
        x_txt_orig = x_txt
        x_img_orig = x_img
        x_pose_orig = x_pose

        # f) Apply separate gating on each segment for the residual
        #    - text uses the combined gate
        out_txt = x_txt_orig + combined_gate_txt * out_txt
        #    - image uses main mod gate
        out_img = x_img_orig + mod_main.gate * out_img
        #    - pose uses pose mod gate
        out_pose = x_pose_orig + mod_pose.gate * out_pose

        # g) Re-concatenate all back
        output = torch.cat((out_txt, out_img, out_pose), dim=1)
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, vec_pose: Tensor = None, image_sequence_len: int = None, vec_cond: Tensor = None, vec_pre_real: Tensor = None, cond_sequence_len: int = None, single_stream_attention_mask: Tensor = None, ar_attn_mask: Tensor = None, bbox_id_embeddings: Tensor = None, **kwargs) -> Tensor:
        # print("x shape in single stream block processor", x.shape)

        # mod, _ = attn.modulation(vec)
        # x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        # qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        # q, k = attn.norm(q, k, v)

        # # compute attention
        # attn_1 = attention(q, k, v, pe=pe)

        # # compute activation in mlp stream, cat again and run second linear layer
        # output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        # output = x + mod.gate * output
        # return output
        # print("x shape in single stream block processor", x.shape)
        B, L, D = x.shape
        txt_len = 512

        # 1) If vec_pose is None, keep the *original* logic.
        if vec_pose is None:
            mod, _ = attn.modulation(vec)
            x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
            qkv, mlp = torch.split(
                attn.linear1(x_mod),
                [3 * attn.hidden_size, attn.mlp_hidden_dim],
                dim=-1
            )

            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                                K=3, H=attn.num_heads)
            q, k = attn.norm(q, k, v)

            # compute attention
            attn_1 = attention(q, k, v, pe=pe, attn_mask=single_stream_attention_mask)

            # compute activation in mlp stream, cat again, and run second linear layer
            output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), dim=2))

            # residual
            output = x + mod.gate * output
            return output

        # 2) Otherwise, we split the input into (text, image, pose).
        x_txt = x[:, :txt_len]  # text portion
        x_img = x[:, txt_len : txt_len + image_sequence_len]
        x_pose = x[:, txt_len + image_sequence_len : ]

        # Get modulations from vec (main) and vec_pose
        mod_main, _ = attn.modulation(vec)
        mod_pose, _ = attn.modulation(vec_pose)

        # Combine text mod parameters from both vec & vec_pose
        combined_scale_txt = mod_main.scale + mod_pose.scale
        combined_shift_txt = mod_main.shift + mod_pose.shift
        combined_gate_txt  = mod_main.gate  + mod_pose.gate

        # Apply combined mod for text
        x_txt_mod = (1 + combined_scale_txt) * attn.pre_norm(x_txt) + combined_shift_txt
        # Apply main mod for image
        x_img_mod = (1 + mod_main.scale) * attn.pre_norm(x_img) + mod_main.shift
        # Apply pose mod for pose
        x_pose_mod = (1 + mod_pose.scale) * attn.pre_norm(x_pose) + mod_pose.shift

        # Re-concatenate
        x_mod = torch.cat((x_txt_mod, x_img_mod, x_pose_mod), dim=1)

        # Now proceed with QKV and MLP as in the original code
        qkv, mlp = torch.split(
            attn.linear1(x_mod),
            [3 * attn.hidden_size, attn.mlp_hidden_dim],
            dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D",
                            K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # Multi-head attention
        attn_1 = attention(q, k, v, pe=pe)

        # MLP path
        out_linear2 = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), dim=2))

        # Split again to apply gating
        out_txt = out_linear2[:, :txt_len]
        out_img = out_linear2[:, txt_len : txt_len + image_sequence_len]
        out_pose = out_linear2[:, txt_len + image_sequence_len : ]

        # For the final residual, each segment has its own gate:
        # Text uses combined gate
        out_txt = x_txt + combined_gate_txt * out_txt
        # Image uses main gate
        out_img = x_img + mod_main.gate * out_img
        # Pose uses pose gate
        out_pose = x_pose + mod_pose.gate * out_pose

        # Re-concatenate final output
        output = torch.cat((out_txt, out_img, out_pose), dim=1)
        return output

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        vec_pose: Tensor = None,
        image_sequence_len: int = None,
        vec_cond: Tensor = None,
        vec_pre_real: Tensor = None,
        cond_sequence_len: int = None,
        ar_attn_mask: Tensor | None = None,
        bbox_id_embeddings: Tensor | None = None,
        single_stream_attention_mask: Tensor | None = None,
        mockpe: nn.Module = None,
        ####&&&&&& add for learnable pe &&&&&&####
        pe_learnable: bool = False,
        batch_shift: Tensor | None = None,
        batch_resize_scale: Tensor | None = None,
        batch_multi_person_last_indicator: Tensor | None = None,
        learned_shift_scale_ratio: float = None,
        ####&&&&&& add for learnable pe &&&&&&####
        ori_h: Tensor | None = None,
        ori_w: Tensor | None = None,
        version1: bool = False,
    ) -> Tensor:
        if image_proj is None:
            if version1:
                return self.processor.forward_version1(self, x=x, vec=vec, pe=pe, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )
            else:
                return self.processor(self, x=x, vec=vec, pe=pe, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )
        else:
            if version1:
                return self.processor.forward_version1(self, x=x, vec=vec, pe=pe, image_proj=image_proj, ip_scale=ip_scale, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )
            else:
                return self.processor(self, x=x, vec=vec, pe=pe, image_proj=image_proj, ip_scale=ip_scale, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    ####&&&&&& add for learnable pe &&&&&&####
                )

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor, return_embeddings: bool = False) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        if return_embeddings:
            return x
        # check_tensor("after final layer's norm", x)
        x = self.linear(x)
        return x

class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens
