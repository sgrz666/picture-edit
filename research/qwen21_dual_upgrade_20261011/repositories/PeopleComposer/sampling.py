import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder
import random
from tqdm import tqdm

def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str], pose_id_value: float = 0.0) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.full((h // 2, w // 2, 3), pose_id_value)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def prepare_split_resolution(t5: HFEmbedder, clip: HFEmbedder, img1: Tensor, img2: Tensor, prompt: str | list[str], pose_id_value: float = 0.0, change_pe: bool = False, smaller_scale: int = 1, black_latents: Tensor = None) -> dict[str, Tensor]:
    bs, c, h1, w1 = img1.shape
    bs, c, h2, w2 = img2.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img1 = rearrange(img1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img1.shape[0] == 1 and bs > 1:
        img1 = repeat(img1, "1 ... -> bs ...", bs=bs)

    img2 = rearrange(img2, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img2.shape[0] == 1 and bs > 1:
        img2 = repeat(img2, "1 ... -> bs ...", bs=bs)

    if black_latents is not None:
        black_latents = black_latents.unsqueeze(0)
        img = torch.cat([black_latents, img1, img2], dim=1)
    else:
        img = torch.cat([img1, img2], dim=1)

    if change_pe:
        img_ids1 = torch.full((h1 // 2, w1 // 2, 3), (smaller_scale // 2 - 0.5))
        img_ids1[..., 1] = img_ids1[..., 1] + (torch.arange(h1 // 2)[:, None]) * smaller_scale
        img_ids1[..., 2] = img_ids1[..., 2] + (torch.arange(w1 // 2)[None, :]) * smaller_scale
        img_ids1[..., 0] = pose_id_value
        img_ids1 = repeat(img_ids1, "h w c -> b (h w) c", b=bs)
    else:
        img_ids1 = torch.zeros((h1 // 2, w1 // 2, 3))
        img_ids1[..., 1] = img_ids1[..., 1] + torch.arange(h1 // 2)[:, None]
        img_ids1[..., 2] = img_ids1[..., 2] + torch.arange(w1 // 2)[None, :]
        img_ids1[..., 0] = pose_id_value
        img_ids1 = repeat(img_ids1, "h w c -> b (h w) c", b=bs)


    img_ids2 = torch.zeros((h2 // 2, w2 // 2, 3))
    img_ids2[..., 1] = img_ids2[..., 1] + torch.arange(h2 // 2)[:, None]
    img_ids2[..., 2] = img_ids2[..., 2] + torch.arange(w2 // 2)[None, :]
    img_ids2 = repeat(img_ids2, "h w c -> b (h w) c", b=bs)

    if black_latents is not None:
        img_ids0 = torch.clone(img_ids1)
        img_ids = torch.cat([img_ids0, img_ids1, img_ids2], dim=1)
    else:
        img_ids = torch.cat([img_ids1, img_ids2], dim=1)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def prepare_single(t5: HFEmbedder,
                clip: HFEmbedder,
                bs,
                c,
                h,
                w,
                device,
                prompt: str | list[str],
                prompt_vec: Tensor = None,
                prompt_t5: Tensor = None,
                pose_id_value: float = 0.0,
                general_prompt: str = None, # general_prompt
                prompt_type: str = "general") -> dict[str, Tensor]:

    if prompt_type == "detailed":
        prompt_to_use = prompt
    elif prompt_type == "general":
        prompt_to_use = general_prompt
    elif prompt_type == "both":
        k = random.randint(0, 1)
        if k == 0:
            prompt_to_use = prompt
        else:
            prompt_to_use = general_prompt
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    if isinstance(prompt_to_use, str):
        prompt_to_use = [prompt_to_use]

    # -------- Positional encodings for pose and image --------
    img_ids_list = []

    for i in range(bs):

        h_i = int(h[i])
        w_i = int(w[i])

        # Pose positional ids
        pose_ids = torch.full((h_i // 2, w_i // 2, 3), pose_id_value)
        pose_ids[..., 1] += torch.arange(h_i // 2)[:, None]
        pose_ids[..., 2] += torch.arange(w_i // 2)[None, :]
        pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
        img_ids_list.append(pose_ids)


    img_ids = torch.stack(img_ids_list, dim=0).to(device)            # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if prompt_t5 is not None:
        txt = prompt_t5
    else:
        txt = t5(prompt_to_use)
    #### pre-save prompt_t5 #####
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    if prompt_vec is not None:
        vec = prompt_vec
    else:
        vec = clip(prompt_to_use)

    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids": img_ids.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
    }


def prepare_dual(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str], img_pose: Tensor, prompt_vec: Tensor = None, prompt_t5: Tensor = None, pose_id_value: float = 0.0, add_h: bool = True, add_pose_clip: bool = False) -> dict[str, Tensor]:
    bs, c, h, w = img.shape # h, w = H/8, W/8 -- autoencoder downsampling -- not patchified yet!
    bs_pose, c_pose, h_pose, w_pose = img_pose.shape # h, w = H/8, W/8 -- autoencoder downsampling -- not patchified yet!
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_pose = rearrange(img_pose, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_pose.shape[0] == 1 and bs > 1:
        img_pose = repeat(img_pose, "1 ... -> bs ...", bs=bs)

    img_ids_pose = torch.full((h_pose // 2, w_pose // 2, 3), pose_id_value) # h//2, w//2 = H/16, W/16 -- after encode + patchified # TODO design choice, positional encoding of pose image
    img_ids_pose[..., 1] = img_ids_pose[..., 1] + torch.arange(h_pose // 2)[:, None]
    img_ids_pose[..., 2] = img_ids_pose[..., 2] + torch.arange(w_pose // 2)[None, :]
    img_ids_pose = repeat(img_ids_pose, "h w c -> b (h w) c", b=bs_pose) # eg. [bs, 1024, 3] every 3 will be [0, i, j], positional encoding


    img_ids = torch.zeros(h // 2, w // 2, 3) # h//2, w//2 = H/16, W/16 -- after encode + patchified
    if add_h:
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None] + h_pose // 2
    else:
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs) # eg. [bs, 1024, 3] every 3 will be [0, i, j], positional encoding


    if isinstance(prompt, str):
        prompt = [prompt]
    #### pre-save prompt_t5 #####
    if prompt_t5 is not None:
        txt = prompt_t5
    else:
        txt = t5(prompt)
    #### pre-save prompt_t5 #####
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    if prompt_vec is not None:
        vec = prompt_vec
    else:
        vec = clip(prompt)
        if add_pose_clip:
            vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_pose": img_pose,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
        "img_ids_pose": img_ids_pose.to(img.device),
        "vec_pose": vec_pose.to(img.device) if add_pose_clip else None
    }

def prepare_dual_train(t5: HFEmbedder,
                       clip: HFEmbedder,
                       bs,
                       c,
                       h,
                       w,
                       bs_pose,
                       c_pose,
                       h_pose,
                       w_pose,
                       device,
                       prompt: str | list[str], # detailed_prompt
                       prompt_vec: Tensor = None,
                       prompt_t5: Tensor = None,
                       pose_id_value: float = 0.0,
                       add_h: bool = True,
                       add_pose_clip: bool = False,
                       use_general_prompt: bool = False, # dummy now
                       general_prompt: str = None, # general_prompt
                       prompt_type: str = "detailed") -> dict[str, Tensor]:
    if prompt_type == "detailed":
        prompt_to_use = prompt
    elif prompt_type == "general":
        prompt_to_use = general_prompt
    elif prompt_type == "both":
        k = random.randint(0, 1)
        if k == 0:
            prompt_to_use = prompt
        else:
            prompt_to_use = general_prompt
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    if isinstance(prompt_to_use, str):
        prompt_to_use = [prompt_to_use]
    # if use_general_prompt:
    #     if isinstance(general_prompt, str):
    #         general_prompt = [general_prompt]
    #     prompt = general_prompt



    # -------- Positional encodings for pose and image --------
    img_ids_pose_list = []
    img_ids_list = []

    for i in range(bs):
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        h_i = int(h[i])
        w_i = int(w[i])

        # Pose positional ids
        pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
        pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
        pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
        pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
        img_ids_pose_list.append(pose_ids)

        # Image positional ids
        img_ids = torch.zeros((h_i // 2, w_i // 2, 3))
        if add_h:
            img_ids[..., 1] += torch.arange(h_i // 2)[:, None] + h_p // 2
        else:
            img_ids[..., 1] += torch.arange(h_i // 2)[:, None]
        img_ids[..., 2] += torch.arange(w_i // 2)[None, :]
        img_ids = img_ids.view(-1, 3)  # (h_i//2 * w_i//2, 3)
        img_ids_list.append(img_ids)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
    img_ids = torch.stack(img_ids_list, dim=0).to(device)            # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if prompt_t5 is not None:
        txt = prompt_t5
    else:
        txt = t5(prompt_to_use)
    #### pre-save prompt_t5 #####
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    if prompt_vec is not None:
        vec = prompt_vec
    else:
        vec = clip(prompt_to_use)
        if add_pose_clip:
            vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in prompt_to_use])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids": img_ids.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }

def process_prompt_with_split_ids_flux_w_rewrite( # TODO-BUG CHECK
    t5_embedder,
    prompt_segments_list: list[list[str]], # kept for signature compatibility, not used
    rewrite_full_caption: list[str],
    rewrite_segment_tag: list[list[int]],
    rewrite_segment_text: list[list[str]],
    bs: int,
    debug: bool = True,
):
    batch_full_prompts = rewrite_full_caption # list[str]
    # Basic sanity checks on outer batch sizes
    assert len(batch_full_prompts) == len(rewrite_segment_tag) == len(rewrite_segment_text), \
        "Batch size mismatch among rewrite_full_caption / rewrite_segment_tag / rewrite_segment_text"


    # 1. Encode the full concatenated prompts using your t5_embedder's forward method
    txt = t5_embedder(batch_full_prompts)

    # Handle batch size repetition if txt has a batch size of 1
    if txt.shape[0] == 1 and bs > 1:
        assert False, "should not happen"
        txt = torch.repeat_interleave(txt, repeats=bs, dim=0)

    current_bs = txt.shape[0]
    seq_len = txt.shape[1] # This is t5_embedder.max_length due to padding="max_length"
    assert seq_len == 512, f"seq_len is {seq_len}, but should be 512"

    # ------------------------------------------------------------
    # 2) Initialize txt_ids (we only fill txt_ids[..., 0])
    # ------------------------------------------------------------
    txt_ids = torch.zeros(current_bs, seq_len, 3, dtype=torch.long, device=txt.device) # Ensure device matches txt

    # ------------------------------------------------------------
    # 3) For each sample: retokenize to get offset_mapping, then label tokens by segment tags
    # ------------------------------------------------------------
    for i in range(current_bs):
        full_prompt = batch_full_prompts[i]
        seg_tags = rewrite_segment_tag[i]   # List[int]
        seg_texts = rewrite_segment_text[i] # List[str]
        assert len(seg_tags) == len(seg_texts), \
            f"len(tag) != len(text) for sample {i}: {len(seg_tags)} vs {len(seg_texts)}"

        # --- Critical alignment check ---
        # Your rule: "".join(seg_texts) should equal full_prompt exactly.
        reconstructed = "".join(seg_texts)
        if reconstructed != full_prompt:
            # Print a helpful diff hint (prefix) for debugging
            if debug:
                print(f"[DEBUG] sample {i}: reconstructed != full_prompt")
                # show first mismatch position if possible
                min_len = min(len(reconstructed), len(full_prompt))
                mismatch = None
                for k in range(min_len):
                    if reconstructed[k] != full_prompt[k]:
                        mismatch = k
                        break
                print("full_prompt[:200]      =", repr(full_prompt[:200]))
                print("reconstructed[:200]    =", repr(reconstructed[:200]))
                if mismatch is not None:
                    print("mismatch at char idx   =", mismatch)
                    print("full_prompt snippet    =", repr(full_prompt[max(0, mismatch-20):mismatch+20]))
                    print("reconstructed snippet  =", repr(reconstructed[max(0, mismatch-20):mismatch+20]))
                else:
                    print("One is a prefix of the other.",
                          "len(full_prompt)=", len(full_prompt),
                          "len(reconstructed)=", len(reconstructed))
            # I don't want to raise error, I want the train to continue first
            print("[Error, should fall back]mismatched! reconstructed != full_prompt!!")
            txt_ids[i].zero_()   # 保守：别喂错的 id
            continue

        # Re-tokenize the full prompt for this batch item to get token IDs and offsets
        # Use the tokenizer directly from the t5_embedder object
        encoded_full_prompt = t5_embedder.tokenizer(
            full_prompt,
            truncation=True,
            max_length=t5_embedder.max_length,
            padding="max_length", # Ensure consistency with how HFEmbedder tokenizes for forward pass
            return_offsets_mapping=True, # Crucial for mapping characters to tokens
            return_tensors="pt"
        )
        input_ids = encoded_full_prompt.input_ids[0]
        offset_mapping = encoded_full_prompt.offset_mapping[0]
        attention_mask = encoded_full_prompt.attention_mask[0] # To handle padding tokens

        eos_id = t5_embedder.tokenizer.eos_token_id

        # We compute char ranges for each segment by walking through seg_texts in order.
        # IMPORTANT: because we require reconstructed == full_prompt,
        # the char index is deterministic without needing to search substrings.
        char_cursor = 0

        # Iterate through the original prompt segments
        for seg_idx, (tag, seg_str) in enumerate(zip(seg_tags, seg_texts)):
            seg_start_char = char_cursor
            seg_end_char = seg_start_char + len(seg_str)
            char_cursor = seg_end_char  # move cursor forward

            # Map tag -> desired id:
            #   tag == 0 -> id = 0
            #   tag > 0  -> id = tag + 1
            if tag == 0:
                seg_id_val = 0
            else:
                seg_id_val = int(tag) + 1  # e.g. 1->2, 2->3, ...


            # Assign seg_id_val to any token whose char span overlaps [seg_start_char, seg_end_char)
            # Skip padding tokens and eos.
            for token_idx, (tok_start, tok_end) in enumerate(offset_mapping.tolist()):
                if token_idx >= seq_len:
                    break

                is_padding = attention_mask[token_idx].item() == 0
                is_eos = input_ids[token_idx].item() == eos_id
                if is_padding or is_eos:
                    continue

                # Offsets can be (0,0) for special tokens in some tokenizers.
                # Overlap test still safe; if seg_id_val==0 it's fine either way,
                # but we keep the overlap condition for correctness.
                if tok_start < seg_end_char and tok_end > seg_start_char:
                    txt_ids[i, token_idx, 0] = seg_id_val

        # Optional extra sanity: ensure we consumed all chars
        if char_cursor != len(full_prompt):
            if debug:
                print(
                    f"[DEBUG] sample {i}: char_cursor ({char_cursor}) "
                    f"!= len(full_prompt) ({len(full_prompt)})"
                )
            print("[Error]!!! char_cursor != len(full_prompt)")
            txt_ids[i].zero_()   # 同样保守

    return txt, txt_ids

def process_prompt_with_split_ids_flux( # TODO-BUG CHECK
    t5_embedder,
    prompt_segments_list: list[list[str]],
    bs: int
):
    """
    Processes a list of prompt segments to create T5 embeddings and custom txt_ids
    using the provided FLUX t5 (HFEmbedder) object.

    Args:
        t5_embedder (HFEmbedder): The loaded HFEmbedder object (your `t5` instance).
        prompt_segments_list (list of list of str): A list where each inner list
                                                    represents the segments for one
                                                    batch item. E.g., [["xxx1", "xxx2", "xxx3"], ...]
        bs (int): Batch size.

    Returns:
        tuple: (txt, txt_ids)
            txt (torch.Tensor): The T5 embeddings for the concatenated prompts.
            txt_ids (torch.Tensor): The custom positional IDs.
    """
    batch_full_prompts = []

    for prompt_segments in prompt_segments_list:
        full_prompt = " ".join(prompt_segments)
        batch_full_prompts.append(full_prompt)

    # 1. Encode the full concatenated prompts using your t5_embedder's forward method
    txt = t5_embedder(batch_full_prompts)

    # Handle batch size repetition if txt has a batch size of 1
    if txt.shape[0] == 1 and bs > 1:
        assert False, "should not happen"
        txt = torch.repeat_interleave(txt, repeats=bs, dim=0)

    current_bs = txt.shape[0]
    seq_len = txt.shape[1] # This is t5_embedder.max_length due to padding="max_length"
    assert seq_len == 512, f"seq_len is {seq_len}, but should be 512"

    # 2. Create txt_ids with custom incrementing IDs in the last dimension
    txt_ids = torch.zeros(current_bs, seq_len, 3, dtype=torch.long, device=txt.device) # Ensure device matches txt

    for i in range(current_bs):
        # Re-tokenize the full prompt for this batch item to get token IDs and offsets
        # Use the tokenizer directly from the t5_embedder object
        encoded_full_prompt = t5_embedder.tokenizer(
            batch_full_prompts[i],
            truncation=True,
            max_length=t5_embedder.max_length,
            padding="max_length", # Ensure consistency with how HFEmbedder tokenizes for forward pass
            return_offsets_mapping=True, # Crucial for mapping characters to tokens
            return_tensors="pt"
        )
        input_ids = encoded_full_prompt.input_ids[0]
        offset_mapping = encoded_full_prompt.offset_mapping[0]
        attention_mask = encoded_full_prompt.attention_mask[0] # To handle padding tokens

        current_id_val = 2 # Start incrementing from 2
        char_idx_in_full_prompt = 0

        # Iterate through the original prompt segments
        for seg_idx, segment in enumerate(prompt_segments_list[i]):
            if seg_idx > 0:
                char_idx_in_full_prompt += 1 # Account for the space added by " ".join()

            segment_start_char = char_idx_in_full_prompt
            segment_end_char = segment_start_char + len(segment)
            char_idx_in_full_prompt = segment_end_char

            # Find the token indices corresponding to this character range
            for token_idx, (start_offset, end_offset) in enumerate(offset_mapping):
                if token_idx >= seq_len: # Should not happen with padding="max_length" but for safety
                    break

                # Check if the token is an actual content token (not padding and not a special token that shouldn't get an ID)
                # T5's special tokens: 0 (pad), 1 (eos). You might need to check other special token IDs if they exist.
                is_padding_token = attention_mask[token_idx] == 0
                is_eos_token = input_ids[token_idx] == t5_embedder.tokenizer.eos_token_id

                if not is_padding_token and not is_eos_token: # Apply ID only to valid content tokens
                    # If the token's character span overlaps with the segment's character span
                    if start_offset < segment_end_char and end_offset > segment_start_char:
                        txt_ids[i, token_idx, 0] = current_id_val

            current_id_val += 1

    return txt, txt_ids


def process_prompt_with_split_ids_flux_changeall( # TODO-BUG CHECK
    t5_embedder,
    prompt_segments_list: list[list[str]],
    bs: int
):
    """
    Processes a list of prompt segments to create T5 embeddings and custom txt_ids
    using the provided FLUX t5 (HFEmbedder) object.

    Args:
        t5_embedder (HFEmbedder): The loaded HFEmbedder object (your `t5` instance).
        prompt_segments_list (list of list of str): A list where each inner list
                                                    represents the segments for one
                                                    batch item. E.g., [["xxx1", "xxx2", "xxx3"], ...]
        bs (int): Batch size.

    Returns:
        tuple: (txt, txt_ids)
            txt (torch.Tensor): The T5 embeddings for the concatenated prompts.
            txt_ids (torch.Tensor): The custom positional IDs.
    """
    batch_full_prompts = []

    for prompt_segments in prompt_segments_list:
        full_prompt = " ".join(prompt_segments)
        batch_full_prompts.append(full_prompt)

    # 1. Encode the full concatenated prompts using your t5_embedder's forward method
    txt = t5_embedder(batch_full_prompts)

    # Handle batch size repetition if txt has a batch size of 1
    if txt.shape[0] == 1 and bs > 1:
        assert False, "should not happen"
        txt = torch.repeat_interleave(txt, repeats=bs, dim=0)

    current_bs = txt.shape[0]
    seq_len = txt.shape[1] # This is t5_embedder.max_length due to padding="max_length"
    assert seq_len == 512, f"seq_len is {seq_len}, but should be 512"

    # 2. Create txt_ids with custom incrementing IDs in the last dimension
    txt_ids = torch.zeros(current_bs, seq_len, 3, dtype=torch.long, device=txt.device) # Ensure device matches txt

    for i in range(current_bs):
        # Re-tokenize the full prompt for this batch item to get token IDs and offsets
        # Use the tokenizer directly from the t5_embedder object
        encoded_full_prompt = t5_embedder.tokenizer(
            batch_full_prompts[i],
            truncation=True,
            max_length=t5_embedder.max_length,
            padding="max_length", # Ensure consistency with how HFEmbedder tokenizes for forward pass
            return_offsets_mapping=True, # Crucial for mapping characters to tokens
            return_tensors="pt"
        )
        input_ids = encoded_full_prompt.input_ids[0]
        offset_mapping = encoded_full_prompt.offset_mapping[0]
        attention_mask = encoded_full_prompt.attention_mask[0] # To handle padding tokens

        current_id_val = 2 # Start incrementing from 2
        char_idx_in_full_prompt = 0

        # Iterate through the original prompt segments
        for seg_idx, segment in enumerate(prompt_segments_list[i]):
            if seg_idx > 0:
                char_idx_in_full_prompt += 1 # Account for the space added by " ".join()

            segment_start_char = char_idx_in_full_prompt
            segment_end_char = segment_start_char + len(segment)
            char_idx_in_full_prompt = segment_end_char

            # Find the token indices corresponding to this character range
            for token_idx, (start_offset, end_offset) in enumerate(offset_mapping):
                if token_idx >= seq_len: # Should not happen with padding="max_length" but for safety
                    break

                # Check if the token is an actual content token (not padding and not a special token that shouldn't get an ID)
                # T5's special tokens: 0 (pad), 1 (eos). You might need to check other special token IDs if they exist.
                is_padding_token = attention_mask[token_idx] == 0
                is_eos_token = input_ids[token_idx] == t5_embedder.tokenizer.eos_token_id

                if not is_padding_token and not is_eos_token: # Apply ID only to valid content tokens
                    # If the token's character span overlaps with the segment's character span
                    if start_offset < segment_end_char and end_offset > segment_start_char:
                        txt_ids[i, token_idx] = current_id_val ### change here!!!!!
                        # #### TRAIL need to be deleted #####
                        # if current_id_val == 2:
                        #     txt_ids[i, token_idx, 0] = 3
                        # elif current_id_val == 3:
                        #     txt_ids[i, token_idx, 0] = 2
                        # else:
                        #     raise ValueError(f"current_id_val is {current_id_val}, but should be 2 or 3")
                        # #### TRAIL need to be deleted #####

            current_id_val += 1

    return txt, txt_ids



def get_prompt_mask_for_prompt_list( # TODO-BUG CHECK
    t5_embedder,
    prompt_list: list[str], # ["xxx1", "xxx2"]
    bs: int = 1
):
    """
    Processes a list of prompt segments to create T5 embeddings and custom txt_ids
    using the provided FLUX t5 (HFEmbedder) object.

    Args:
        t5_embedder (HFEmbedder): The loaded HFEmbedder object (your `t5` instance).
        prompt_segments_list (list of list of str): A list where each inner list
                                                    represents the segments for one
                                                    batch item. E.g., [["xxx1", "xxx2", "xxx3"], ...]
        bs (int): Batch size.

    Returns:
        tuple: (txt, txt_ids)
            txt (torch.Tensor): The T5 embeddings for the concatenated prompts.
            txt_ids (torch.Tensor): The custom positional IDs.
    """
    full_prompt = [" ".join(prompt_list)] # ["xxx1 xxx2"]

    # 1. Encode the full concatenated prompts using your t5_embedder's forward method
    txt = t5_embedder(full_prompt) # [1, 512]

    # Handle batch size repetition if txt has a batch size of 1
    if txt.shape[0] == 1 and bs > 1:
        assert False, "should not happen"
        txt = torch.repeat_interleave(txt, repeats=bs, dim=0)

    current_bs = txt.shape[0] # 1
    seq_len = txt.shape[1] # This is t5_embedder.max_length due to padding="max_length"
    assert seq_len == 512, f"seq_len is {seq_len}, but should be 512"

    # 2. Create txt_ids with custom incrementing IDs in the last dimension
    txt_ids = torch.zeros(current_bs, seq_len, dtype=torch.long, device=txt.device) # [1, 512]


    # Re-tokenize the full prompt for this batch item to get token IDs and offsets
    # Use the tokenizer directly from the t5_embedder object
    encoded_full_prompt = t5_embedder.tokenizer(
        full_prompt[0],
        truncation=True,
        max_length=t5_embedder.max_length,
        padding="max_length", # Ensure consistency with how HFEmbedder tokenizes for forward pass
        return_offsets_mapping=True, # Crucial for mapping characters to tokens
        return_tensors="pt"
    )
    input_ids = encoded_full_prompt.input_ids[0]
    offset_mapping = encoded_full_prompt.offset_mapping[0]
    attention_mask = encoded_full_prompt.attention_mask[0] # To handle padding tokens

    current_id_val = 0 # Start incrementing from 2
    char_idx_in_full_prompt = 0

    # Iterate through the original prompt segments
    for seg_idx, segment in enumerate(prompt_list): # ["xxx1", "xxx2"]
        if seg_idx > 0:
            char_idx_in_full_prompt += 1 # Account for the space added by " ".join()

        segment_start_char = char_idx_in_full_prompt
        segment_end_char = segment_start_char + len(segment)
        char_idx_in_full_prompt = segment_end_char

        # Find the token indices corresponding to this character range
        for token_idx, (start_offset, end_offset) in enumerate(offset_mapping):
            if token_idx >= seq_len: # Should not happen with padding="max_length" but for safety
                break

            # Check if the token is an actual content token (not padding and not a special token that shouldn't get an ID)
            # T5's special tokens: 0 (pad), 1 (eos). You might need to check other special token IDs if they exist.
            is_padding_token = attention_mask[token_idx] == 0
            is_eos_token = input_ids[token_idx] == t5_embedder.tokenizer.eos_token_id

            if not is_padding_token and not is_eos_token: # Apply ID only to valid content tokens
                # If the token's character span overlaps with the segment's character span
                if start_offset < segment_end_char and end_offset > segment_start_char:
                    txt_ids[0, token_idx] = current_id_val

        current_id_val += 1

    return txt, txt_ids # [1, 512], [1, 512]

def prepare_w_prompt_mask(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt_list: list[str], clip_prompt: str,pose_id_value: float = 0.0) -> dict[str, Tensor]:
    bs, c, h, w = img.shape # 1, 16, 128, 128
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.full((h // 2, w // 2, 3), pose_id_value)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    txt, txt_mask_ids = get_prompt_mask_for_prompt_list(t5, prompt_list)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(clip_prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "txt_mask_ids": txt_mask_ids.to(img.device),
        "vec": vec.to(img.device),
    }

def generate_img_ids_for_embed_id(
    bs: int,
    h_condition: torch.Tensor, # Batch of h for condition
    w_condition: torch.Tensor, # Batch of w for condition
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
):
    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []
    bbox_ids_pose_list = []
    bbox_ids_condition_list = []
    bbox_ids_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            bbox_ids_grid = torch.full((height_grid // 2, width_grid // 2), -1, dtype=torch.long)
            img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))

            img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
            img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 0

            for bbox_coords in current_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                bbox_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3), bbox_ids_grid.view(-1) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_condition, bbox_ids_condition = apply_bbox_ids_to_grid(h_c, w_c)
        img_ids_pose, bbox_ids_pose = apply_bbox_ids_to_grid(h_p, w_p)
        img_ids_real, bbox_ids_real = apply_bbox_ids_to_grid(h_r, w_r)

        # Append to lists
        img_ids_condition_list.append(img_ids_condition)
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)
        bbox_ids_pose_list.append(bbox_ids_pose)
        bbox_ids_condition_list.append(bbox_ids_condition)
        bbox_ids_real_list.append(bbox_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)
    bbox_ids_pose = torch.stack(bbox_ids_pose_list, dim=0).to(device)
    bbox_ids_condition = torch.stack(bbox_ids_condition_list, dim=0).to(device)
    bbox_ids_real = torch.stack(bbox_ids_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_condition, img_ids_real, bbox_ids_pose, bbox_ids_condition, bbox_ids_real


def generate_img_ids_for_embed_id_realnotadded(
    bs: int,
    h_condition: torch.Tensor, # Batch of h for condition
    w_condition: torch.Tensor, # Batch of w for condition
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
):
    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []
    bbox_ids_pose_list = []
    bbox_ids_condition_list = []
    bbox_ids_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, is_pose):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            bbox_ids_grid = torch.full((height_grid // 2, width_grid // 2), -1, dtype=torch.long) # is_real: all -1, nothing will be added to q, k in the embed_id_features!
            img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))

            img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
            img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            if is_pose:
                current_bbox_id = 0

                for bbox_coords in current_bboxes:
                    x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords

                    # Scale and convert to patch coordinates using integer division
                    x1_patch = x1_orig // scale_factor
                    y1_patch = y1_orig // scale_factor
                    x2_patch = x2_orig // scale_factor
                    y2_patch = y2_orig // scale_factor

                    # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                    # Note: `width_grid // 2 - 1` is the max index for width
                    x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                    y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                    x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                    y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                    # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                    if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                    if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                    # If the box is effectively empty after scaling/clamping, skip it.
                    if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                        continue

                    # Assign the current_bbox_id to the first channel (index 0) of the grid
                    bbox_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1] = current_bbox_id

                    current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3), bbox_ids_grid.view(-1) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_condition, bbox_ids_condition = apply_bbox_ids_to_grid(h_c, w_c, is_pose=True)
        img_ids_pose, bbox_ids_pose = apply_bbox_ids_to_grid(h_p, w_p, is_pose=True)
        img_ids_real, bbox_ids_real = apply_bbox_ids_to_grid(h_r, w_r, is_pose=False)

        # Append to lists
        img_ids_condition_list.append(img_ids_condition)
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)
        bbox_ids_pose_list.append(bbox_ids_pose)
        bbox_ids_condition_list.append(bbox_ids_condition)
        bbox_ids_real_list.append(bbox_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)
    bbox_ids_pose = torch.stack(bbox_ids_pose_list, dim=0).to(device)
    bbox_ids_condition = torch.stack(bbox_ids_condition_list, dim=0).to(device)
    bbox_ids_real = torch.stack(bbox_ids_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_condition, img_ids_real, bbox_ids_pose, bbox_ids_condition, bbox_ids_real

def generate_img_ids_for_layout_prediction(
    bs: int,
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
):
    img_ids_real_list = []
    bbox_ids_real_list = []

    for i in range(bs):
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            bbox_ids_grid = torch.full((height_grid // 2, width_grid // 2), 0, dtype=torch.long)
            img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))

            img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
            img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 1

            for bbox_coords in current_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                bbox_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3), bbox_ids_grid.view(-1) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_real, bbox_ids_real = apply_bbox_ids_to_grid(h_r, w_r)

        # Append to lists
        img_ids_real_list.append(img_ids_real)
        bbox_ids_real_list.append(bbox_ids_real)

    # Stack everything
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)
    bbox_ids_real = torch.stack(bbox_ids_real_list, dim=0).to(device)

    return img_ids_real, bbox_ids_real


from typing import List, Tuple
import torch

def prepare_text_for_layout_prediction(  # TODO-BUG CHECK
    t5_embedder,
    prompt_segments_list: List[List[str]],
    bs: int
):
    """
    Processes a list of prompt segments to create T5 embeddings and segment token spans
    using the provided FLUX t5 (HFEmbedder) object.

    Args:
        t5_embedder (HFEmbedder): The loaded HFEmbedder object (your `t5` instance).
        prompt_segments_list (list of list of str): A list where each inner list
                                                    represents the segments for one
                                                    batch item. E.g., [["xxx1", "xxx2", "xxx3"], ...]
        bs (int): Batch size.

    Returns:
        tuple: (txt, segment_token_spans)
            txt (torch.Tensor): The T5 embeddings for the concatenated prompts.
                                Shape typically (B, 512, D).
            segment_token_spans (torch.LongTensor): Per-batch, per-segment (start_idx, end_idx),
                                inclusive indices into the tokenized full prompt.
                                Shape (B, max_num_segments, 2). Unused/trimmed segments are (-1, -1).
    """
    # ---- 0) Join segments into full prompts (same as your flow) ----
    batch_full_prompts = []
    for prompt_segments in prompt_segments_list:
        full_prompt = " ".join(prompt_segments)
        batch_full_prompts.append(full_prompt)

    # ---- 1) Encode full prompts to get embeddings (same as your flow) ----
    txt = t5_embedder(batch_full_prompts)

    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    # Handle batch size repetition if txt has a batch size of 1 (same as your flow)
    if txt.shape[0] == 1 and bs > 1:
        assert False, "should not happen"
        txt = torch.repeat_interleave(txt, repeats=bs, dim=0)

    current_bs = txt.shape[0]
    seq_len = txt.shape[1]  # This is t5_embedder.max_length due to padding="max_length"
    assert seq_len == 512, f"seq_len is {seq_len}, but should be 512"

    # ---- 2) Prepare container for segment spans (NEW) ----
    # Determine max number of segments across batch so we can return a dense tensor
    max_num_segments = max(len(segments) for segments in prompt_segments_list) if prompt_segments_list else 0
    # Create tensor (B, max_num_segments, 2), fill with -1 to denote "not set / truncated"
    segment_token_spans = torch.full(
        (current_bs, max_num_segments, 2),
        fill_value=-1,
        dtype=torch.long,
        device=txt.device
    )

    # ---- 3) For each item, retokenize with offsets and compute spans (mirrors your loop/logic) ----
    eos_id = t5_embedder.tokenizer.eos_token_id

    for i in range(current_bs):
        # Re-tokenize the full prompt for this batch item to get token IDs and offsets
        encoded_full_prompt = t5_embedder.tokenizer(
            batch_full_prompts[i],
            truncation=True,
            max_length=t5_embedder.max_length,
            padding="max_length",  # ensure consistency with forward pass
            return_offsets_mapping=True,  # map characters <-> tokens
            return_tensors="pt"
        )
        input_ids = encoded_full_prompt.input_ids[0]           # (L,)
        offset_mapping = encoded_full_prompt.offset_mapping[0] # (L, 2)
        attention_mask = encoded_full_prompt.attention_mask[0] # (L,)

        # Compute the valid token range (stop at EOS or padding)
        # This mirrors your special-token filtering but makes span search cheaper
        valid_len = 0
        for token_idx in range(seq_len):
            if attention_mask[token_idx].item() == 0:
                break
            if eos_id is not None and input_ids[token_idx].item() == eos_id:
                break
            valid_len += 1

        # Track character index in the joined string (space-aware), same as your flow
        char_idx_in_full_prompt = 0

        # Iterate through the original prompt segments (same order)
        for seg_idx, segment in enumerate(prompt_segments_list[i]):
            if seg_idx > 0:
                char_idx_in_full_prompt += 1  # account for the single space in " ".join()

            segment_start_char = char_idx_in_full_prompt
            segment_end_char = segment_start_char + len(segment)
            char_idx_in_full_prompt = segment_end_char

            # ---- Find first/last token overlapping this segment's char span (NEW in place of txt_ids writes) ----
            tok_start = None
            tok_end = None

            # Scan only valid tokens
            for token_idx in range(valid_len):
                start_offset = offset_mapping[token_idx][0].item()
                end_offset = offset_mapping[token_idx][1].item()

                # Some special tokens may have zero-length offsets; skip them
                if end_offset <= start_offset:
                    continue

                # Overlap test between (token span) and (segment span)
                if (start_offset < segment_end_char) and (end_offset > segment_start_char):
                    if tok_start is None:
                        tok_start = token_idx
                    tok_end = token_idx  # keep updating until last overlapping token

            # Write into the fixed-size (B, max_num_segments, 2) tensor
            if seg_idx < max_num_segments:
                if tok_start is not None and tok_end is not None:
                    segment_token_spans[i, seg_idx, 0] = tok_start
                    segment_token_spans[i, seg_idx, 1] = tok_end
                else:
                    # Segment is truncated or produced no overlapping tokens -> stays as (-1, -1)
                    pass

    return txt, txt_ids, segment_token_spans


def get_base_pose_id(apply_smaller_pose: int):
    base_number = apply_smaller_pose / 2 - 0.5
    unit_distance = apply_smaller_pose
    return base_number, unit_distance



def generate_img_ids(
    bs: int,
    h_condition: torch.Tensor, # Batch of h for condition
    w_condition: torch.Tensor, # Batch of w for condition
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    pose_id_value: float,        # Value for pose/condition IDs (e.g., 1). Now only a default.
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
    apply_smaller_pose: int = 1,
):
    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, is_pose=False):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            if is_pose:
                base_number, unit_distance = get_base_pose_id(apply_smaller_pose) # base = -0.5 + apply_smaller_pose/2, unit_distance = apply_smaller_pose
                img_ids_grid = torch.full((height_grid // 2, width_grid // 2, 3), base_number)
                img_ids_grid[..., 1] += (torch.arange(height_grid // 2, dtype=torch.long)[:, None]) * unit_distance # H indices
                img_ids_grid[..., 2] += (torch.arange(width_grid // 2, dtype=torch.long)[None, :]) * unit_distance # W indices
                img_ids_grid[..., 0] = pose_id_value
            else:
                img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))
                img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
                img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 2 # Start incrementing from 2 for bounding boxes

            for bbox_coords in current_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords
                if is_pose:
                    x1_orig = int(x1_orig / apply_smaller_pose)
                    y1_orig = int(y1_orig / apply_smaller_pose)
                    x2_orig = int(x2_orig / apply_smaller_pose)
                    y2_orig = int(y2_orig / apply_smaller_pose)

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                img_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1, 0] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_condition = apply_bbox_ids_to_grid(h_c, w_c, is_pose=True)
        img_ids_pose = apply_bbox_ids_to_grid(h_p, w_p, is_pose=True)
        img_ids_real = apply_bbox_ids_to_grid(h_r, w_r, is_pose=False)

        # Append to lists
        img_ids_condition_list.append(img_ids_condition)
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_condition, img_ids_real

def generate_img_ids_oneshot(
    bs: int,
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    pose_id_value: float,        # Value for pose/condition IDs (e.g., 1). Now only a default.
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
    apply_smaller_pose: int = 1,
):
    img_ids_pose_list = []
    img_ids_real_list = []

    for i in range(bs):
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, is_pose=False):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            if is_pose:
                base_number, unit_distance = get_base_pose_id(apply_smaller_pose) # base = 0, unit_distance = 1
                img_ids_grid = torch.full((height_grid // 2, width_grid // 2, 3), base_number)
                img_ids_grid[..., 1] += (torch.arange(height_grid // 2, dtype=torch.long)[:, None]) * unit_distance # H indices
                img_ids_grid[..., 2] += (torch.arange(width_grid // 2, dtype=torch.long)[None, :]) * unit_distance # W indices
                img_ids_grid[..., 0] = pose_id_value
            else:
                img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))
                img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
                img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 2 # Start incrementing from 2 for bounding boxes

            for bbox_coords in current_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords
                if is_pose:
                    x1_orig = int(x1_orig / apply_smaller_pose)
                    y1_orig = int(y1_orig / apply_smaller_pose)
                    x2_orig = int(x2_orig / apply_smaller_pose)
                    y2_orig = int(y2_orig / apply_smaller_pose)

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                img_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1, 0] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_pose = apply_bbox_ids_to_grid(h_p, w_p, is_pose=True)
        img_ids_real = apply_bbox_ids_to_grid(h_r, w_r, is_pose=False)

        # Append to lists
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_real


def generate_img_ids_vanilla_t2i(
    bs: int,
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
):
    img_ids_real_list = []

    for i in range(bs):
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, is_pose=False):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.

            img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))
            img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
            img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 2 # Start incrementing from 2 for bounding boxes

            for bbox_coords in current_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                img_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1, 0] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3) # Flatten to sequence

        img_ids_real = apply_bbox_ids_to_grid(h_r, w_r, is_pose=False)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)

    return img_ids_real

def generate_img_ids_version2(
    bs: int,
    h_condition: torch.Tensor, # Batch of h for condition
    w_condition: torch.Tensor, # Batch of w for condition
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    h_pre_real: torch.Tensor,      # Batch of h for pre_real
    w_pre_real: torch.Tensor,      # Batch of w for pre_real
    pose_id_value: float,        # Value for pose/condition IDs (e.g., 1). Now only a default.
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
    apply_smaller_pose: int = 1,
):
    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []
    img_ids_pre_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid
        # pre_real
        h_pr = int(h_pre_real[i]) # height of the patched latent grid
        w_pr = int(w_pre_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, is_pose=False):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            if is_pose:
                base_number, unit_distance = get_base_pose_id(apply_smaller_pose) # base = -0.5 + apply_smaller_pose/2, unit_distance = apply_smaller_pose
                img_ids_grid = torch.full((height_grid // 2, width_grid // 2, 3), base_number)
                img_ids_grid[..., 1] += (torch.arange(height_grid // 2, dtype=torch.long)[:, None]) * unit_distance # H indices
                img_ids_grid[..., 2] += (torch.arange(width_grid // 2, dtype=torch.long)[None, :]) * unit_distance # W indices
                img_ids_grid[..., 0] = pose_id_value
            else:
                img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))
                img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
                img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 2 # Start incrementing from 2 for bounding boxes

            for bbox_coords in current_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords
                if is_pose:
                    x1_orig = int(x1_orig / apply_smaller_pose)
                    y1_orig = int(y1_orig / apply_smaller_pose)
                    x2_orig = int(x2_orig / apply_smaller_pose)
                    y2_orig = int(y2_orig / apply_smaller_pose)

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                img_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1, 0] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_condition = apply_bbox_ids_to_grid(h_c, w_c, is_pose=True)
        img_ids_pose = apply_bbox_ids_to_grid(h_p, w_p, is_pose=True)
        img_ids_real = apply_bbox_ids_to_grid(h_r, w_r, is_pose=False)
        img_ids_pre_real = apply_bbox_ids_to_grid(h_pr, w_pr, is_pose=False)

        # Append to lists
        img_ids_condition_list.append(img_ids_condition)
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)
        img_ids_pre_real_list.append(img_ids_pre_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)
    img_ids_pre_real = torch.stack(img_ids_pre_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_condition, img_ids_real, img_ids_pre_real



def generate_img_ids2(
    bs: int,
    h_condition: torch.Tensor, # Batch of h for condition
    w_condition: torch.Tensor, # Batch of w for condition
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    pose_id_value: float,        # Value for pose/condition IDs (e.g., 1). Now only a default.
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
    apply_smaller_pose: int = 1,
):
    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, signal_type):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            if signal_type == "condition" or signal_type == "pose":
                img_ids_grid = torch.full((height_grid // 2, width_grid // 2, 3), pose_id_value)
            else:
                img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))
            img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
            img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 2 # Start incrementing from 2 for bounding boxes

            if signal_type == "condition":
                use_bboxes = current_bboxes[:-1]
                # print(f"condition use_bboxes: {use_bboxes}")
            else:
                use_bboxes = current_bboxes
                # print(f"real use_bboxes: {use_bboxes}")

            for bbox_coords in use_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                img_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1, 0] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_condition = apply_bbox_ids_to_grid(h_c, w_c, signal_type="condition")
        img_ids_pose = apply_bbox_ids_to_grid(h_p, w_p, signal_type="pose")
        img_ids_real = apply_bbox_ids_to_grid(h_r, w_r, signal_type="real")

        # Append to lists
        img_ids_condition_list.append(img_ids_condition)
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_condition, img_ids_real

def generate_img_ids2_version2(
    bs: int,
    h_condition: torch.Tensor, # Batch of h for condition
    w_condition: torch.Tensor, # Batch of w for condition
    h_pose: torch.Tensor,      # Batch of h for pose
    w_pose: torch.Tensor,      # Batch of w for pose
    h_real: torch.Tensor,      # Batch of h for real
    w_real: torch.Tensor,      # Batch of w for real
    h_pre_real: torch.Tensor,      # Batch of h for pre_real
    w_pre_real: torch.Tensor,      # Batch of w for pre_real
    pose_id_value: float,        # Value for pose/condition IDs (e.g., 1). Now only a default.
    bounding_boxes_list: list[list[list[int]]], # List of lists of bboxes, for each batch item
    device: torch.device,
    apply_smaller_pose: int = 1,
):
    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []
    img_ids_pre_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid

        # pre_real
        h_pr = int(h_pre_real[i]) # height of the patched latent grid
        w_pr = int(w_pre_real[i]) # width of the patched latent grid

        # Hardcoded scaling factor (1/16 from original image pixels to latent patches)
        scale_factor = 16

        # Get bounding boxes for the current batch item
        # print(f"bounding_boxes_list: {bounding_boxes_list}")
        # print(f"bounding_boxes_list[i]: {bounding_boxes_list[i]}")
        current_bboxes = bounding_boxes_list[i]

        # --- Helper function to apply bbox IDs to a grid ---
        def apply_bbox_ids_to_grid(height_grid, width_grid, signal_type):
            # Initialize with 0 in the first channel, which will be overwritten by bbox IDs
            # or remain 0 if no bbox covers it.
            if signal_type == "condition" or signal_type == "pose":
                img_ids_grid = torch.full((height_grid // 2, width_grid // 2, 3), pose_id_value)
            else: # real, pre_real
                img_ids_grid = torch.zeros((height_grid // 2, width_grid // 2, 3))
            img_ids_grid[..., 1] += torch.arange(height_grid // 2, dtype=torch.long)[:, None] # H indices
            img_ids_grid[..., 2] += torch.arange(width_grid // 2, dtype=torch.long)[None, :] # W indices

            current_bbox_id = 2 # Start incrementing from 2 for bounding boxes

            if signal_type == "condition":
                use_bboxes = current_bboxes[:-1]
                # print(f"condition use_bboxes: {use_bboxes}")
            else:
                use_bboxes = current_bboxes
                # print(f"real use_bboxes: {use_bboxes}")

            for bbox_coords in use_bboxes:
                x1_orig, y1_orig, x2_orig, y2_orig = bbox_coords

                # Scale and convert to patch coordinates using integer division
                x1_patch = x1_orig // scale_factor
                y1_patch = y1_orig // scale_factor
                x2_patch = x2_orig // scale_factor
                y2_patch = y2_orig // scale_factor

                # Clamp to ensure coordinates stay within bounds [0, H_patches-1] or [0, W_patches-1]
                # Note: `width_grid // 2 - 1` is the max index for width
                x1_patch = max(0, min(x1_patch, width_grid // 2 - 1))
                y1_patch = max(0, min(y1_patch, height_grid // 2 - 1))
                x2_patch = max(0, min(x2_patch, width_grid // 2 - 1))
                y2_patch = max(0, min(y2_patch, height_grid // 2 - 1))

                # Ensure x1 <= x2 and y1 <= y2, and handle empty ranges
                if x1_patch > x2_patch: x1_patch, x2_patch = x2_patch, x1_patch
                if y1_patch > y2_patch: y1_patch, y2_patch = y2_patch, y1_patch

                # If the box is effectively empty after scaling/clamping, skip it.
                if x1_patch >= x2_patch or y1_patch >= y2_patch: # TODO-BUG CHECK add= to handle [-1, -1, -1, -1]
                    continue

                # Assign the current_bbox_id to the first channel (index 0) of the grid
                img_ids_grid[y1_patch : y2_patch + 1, x1_patch : x2_patch + 1, 0] = current_bbox_id

                current_bbox_id += 1 # Increment ID for the next bounding box

            return img_ids_grid.view(-1, 3) # Flatten to sequence

        # Apply the bounding box logic to all three ID types
        img_ids_condition = apply_bbox_ids_to_grid(h_c, w_c, signal_type="condition")
        img_ids_pose = apply_bbox_ids_to_grid(h_p, w_p, signal_type="pose")
        img_ids_real = apply_bbox_ids_to_grid(h_r, w_r, signal_type="real")
        img_ids_pre_real = apply_bbox_ids_to_grid(h_pr, w_pr, signal_type="pre_real")

        # Append to lists
        img_ids_condition_list.append(img_ids_condition)
        img_ids_pose_list.append(img_ids_pose)
        img_ids_real_list.append(img_ids_real)
        img_ids_pre_real_list.append(img_ids_pre_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)
    img_ids_pre_real = torch.stack(img_ids_pre_real_list, dim=0).to(device)

    return img_ids_pose, img_ids_condition, img_ids_real, img_ids_pre_real

def prepare_layout_prediction(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same


    # -------- Positional encodings for pose and image --------
    assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
    img_ids_real, bbox_ids_real = generate_img_ids_for_layout_prediction(
        bs,
        h_real,
        w_real,
        bounding_boxes_in_order,
        device
    )

    #### pre-save prompt_t5 #####
    txt, txt_ids, segment_token_spans = prepare_text_for_layout_prediction(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim
    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_real": img_ids_real.to(device),
        "bbox_ids_real": bbox_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "segment_token_spans": segment_token_spans.to(device),
        "vec": vec.to(device),
    }

def prepare_layout_prediction_infer(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    print("text_caption", prompt)

    #assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same

    img_ids_real_list = []
    for i in range(bs):
        # real
        h_r = int(h_real[i]) # height of the patched latent grid
        w_r = int(w_real[i]) # width of the patched latent grid
        img_ids_grid = torch.zeros((h_r // 2, w_r // 2, 3))
        img_ids_grid[..., 1] += torch.arange(h_r // 2, dtype=torch.long)[:, None] # H indices
        img_ids_grid[..., 2] += torch.arange(w_r // 2, dtype=torch.long)[None, :] # W indices
        img_ids_real = img_ids_grid.view(-1, 3) # Flatten to sequence
        img_ids_real_list.append(img_ids_real)

    img_ids_real = torch.stack(img_ids_real_list, dim=0).to("cuda")

    #### pre-save prompt_t5 #####
    txt, txt_ids, segment_token_spans = prepare_text_for_layout_prediction(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim
    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "segment_token_spans": segment_token_spans.to(device),
        "vec": vec.to(device),
    }



def prepare_dual_train_ar_w_rewrite(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # condition
                        h_condition,
                        w_condition,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        rewrite_full_caption: list[str],
                        rewrite_segment_tag: list[list[int]],
                        rewrite_segment_text: list[list[str]],
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        customize_img_ids: bool = False,
                        customize_txt_ids: bool = False,
                        generate_img_ids_type: int = 1,
                        apply_smaller_pose: int = 1,
                        ) -> dict[str, Tensor]:
    # print("H_condition", h_condition)
    # print("W_condition", w_condition)
    # print("H_pose", h_pose)
    # print("W_pose", w_pose)
    # print("H_real", h_real)
    # print("W_real", w_real)

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    if customize_txt_ids:
        assert isinstance(prompt[0], list), "prompt must be a list of lists" # outer list is batch dim
        # clip the same


    # -------- Positional encodings for pose and image --------
    if customize_img_ids:
        assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
        generate_img_ids_func_dict = {1: generate_img_ids, 2: generate_img_ids2} # TODO 3 not implemented
        img_ids_pose, img_ids_condition, img_ids_real = generate_img_ids_func_dict[generate_img_ids_type](
            bs,
            h_condition,
            w_condition,
            h_pose,
            w_pose,
            h_real,
            w_real,
            pose_id_value,
            bounding_boxes_in_order,
            device,
            apply_smaller_pose
        )
    else:
        img_ids_pose_list = []
        img_ids_condition_list = []
        img_ids_real_list = []

        for i in range(bs):
            # condition
            h_c = int(h_condition[i])
            w_c = int(w_condition[i])
            # pose
            h_p = int(h_pose[i])
            w_p = int(w_pose[i])
            # real
            h_r = int(h_real[i])
            w_r = int(w_real[i])

            # condition
            img_ids_condition = torch.full((h_c // 2, w_c // 2, 3), pose_id_value)
            img_ids_condition[..., 1] += torch.arange(h_c // 2)[:, None]
            img_ids_condition[..., 2] += torch.arange(w_c // 2)[None, :]
            img_ids_condition = img_ids_condition.view(-1, 3)  # (h_c//2 * w_c//2, 3) to sequence
            img_ids_condition_list.append(img_ids_condition)


            # Pose positional ids
            pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
            pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
            pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
            pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
            img_ids_pose_list.append(pose_ids)

            # Image positional ids
            img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
            img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
            img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
            img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
            img_ids_real_list.append(img_ids_real)

        # Stack everything
        img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
        img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)            # [bs, H_patches, 3]
        img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if customize_txt_ids: # True
        txt, txt_ids = process_prompt_with_split_ids_flux_w_rewrite(t5, prompt, rewrite_full_caption,rewrite_segment_tag, rewrite_segment_text,bs) # prompt is list of lists of strings, outer list is batch dim

    else:
        txt = t5(prompt) # prompt is list of strings, list is batch dim
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    # print("img_ids_condition.shape", img_ids_condition.shape)
    # print("img_ids_pose.shape", img_ids_pose.shape)
    # print("img_ids_real.shape", img_ids_real.shape)
    # print("txt.shape", txt.shape)
    # print("txt_ids.shape", txt_ids.shape)
    # print("vec.shape", vec.shape)
    # print("vec_pose.shape", vec_pose.shape)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }


def prepare_dual_train_ar(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # condition
                        h_condition,
                        w_condition,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        customize_img_ids: bool = False,
                        customize_txt_ids: bool = False,
                        generate_img_ids_type: int = 1,
                        apply_smaller_pose: int = 1,
                        ) -> dict[str, Tensor]:
    # print("H_condition", h_condition)
    # print("W_condition", w_condition)
    # print("H_pose", h_pose)
    # print("W_pose", w_pose)
    # print("H_real", h_real)
    # print("W_real", w_real)

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    if customize_txt_ids:
        assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same


    # -------- Positional encodings for pose and image --------
    if customize_img_ids:
        assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
        generate_img_ids_func_dict = {1: generate_img_ids, 2: generate_img_ids2} # TODO 3 not implemented
        img_ids_pose, img_ids_condition, img_ids_real = generate_img_ids_func_dict[generate_img_ids_type](
            bs,
            h_condition,
            w_condition,
            h_pose,
            w_pose,
            h_real,
            w_real,
            pose_id_value,
            bounding_boxes_in_order,
            device,
            apply_smaller_pose
        )
    else:
        img_ids_pose_list = []
        img_ids_condition_list = []
        img_ids_real_list = []

        for i in range(bs):
            # condition
            h_c = int(h_condition[i])
            w_c = int(w_condition[i])
            # pose
            h_p = int(h_pose[i])
            w_p = int(w_pose[i])
            # real
            h_r = int(h_real[i])
            w_r = int(w_real[i])

            # condition
            img_ids_condition = torch.full((h_c // 2, w_c // 2, 3), pose_id_value)
            img_ids_condition[..., 1] += torch.arange(h_c // 2)[:, None]
            img_ids_condition[..., 2] += torch.arange(w_c // 2)[None, :]
            img_ids_condition = img_ids_condition.view(-1, 3)  # (h_c//2 * w_c//2, 3) to sequence
            img_ids_condition_list.append(img_ids_condition)


            # Pose positional ids
            pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
            pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
            pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
            pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
            img_ids_pose_list.append(pose_ids)

            # Image positional ids
            img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
            img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
            img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
            img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
            img_ids_real_list.append(img_ids_real)

        # Stack everything
        img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
        img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)            # [bs, H_patches, 3]
        img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if customize_txt_ids:
        txt, txt_ids = process_prompt_with_split_ids_flux(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim
    else:
        txt = t5(prompt) # prompt is list of strings, list is batch dim
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    # print("img_ids_condition.shape", img_ids_condition.shape)
    # print("img_ids_pose.shape", img_ids_pose.shape)
    # print("img_ids_real.shape", img_ids_real.shape)
    # print("txt.shape", txt.shape)
    # print("txt_ids.shape", txt_ids.shape)
    # print("vec.shape", vec.shape)
    # print("vec_pose.shape", vec_pose.shape)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }



def prepare_dual_train_oneshot(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        customize_img_ids: bool = False,
                        customize_txt_ids: bool = False,
                        generate_img_ids_type: int = 1,
                        apply_smaller_pose: int = 1,
                        ) -> dict[str, Tensor]:
    # print("H_condition", h_condition)
    # print("W_condition", w_condition)
    # print("H_pose", h_pose)
    # print("W_pose", w_pose)
    # print("H_real", h_real)
    # print("W_real", w_real)

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    if customize_txt_ids:
        assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same


    # -------- Positional encodings for pose and image --------
    if customize_img_ids:
        assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
        img_ids_pose, img_ids_real = generate_img_ids_oneshot(
            bs,
            h_pose,
            w_pose,
            h_real,
            w_real,
            pose_id_value,
            bounding_boxes_in_order,
            device,
            apply_smaller_pose
        )

    else:
        img_ids_pose_list = []
        img_ids_real_list = []

        for i in range(bs):
            # pose
            h_p = int(h_pose[i])
            w_p = int(w_pose[i])
            # real
            h_r = int(h_real[i])
            w_r = int(w_real[i])

            # Pose positional ids
            pose_ids = torch.zeros((h_r // 2, w_r // 2, 3))
            pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
            pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
            pose_ids[..., 0] = pose_id_value
            pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
            img_ids_pose_list.append(pose_ids)

            # Image positional ids
            img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
            img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
            img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
            img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
            img_ids_real_list.append(img_ids_real)

        # Stack everything
        img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
        img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if customize_txt_ids:
        txt, txt_ids = process_prompt_with_split_ids_flux_changeall(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim
    else:
        txt = t5(prompt) # prompt is list of strings, list is batch dim
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    # print("img_ids_condition.shape", img_ids_condition.shape)
    # print("img_ids_pose.shape", img_ids_pose.shape)
    # print("img_ids_real.shape", img_ids_real.shape)
    # print("txt.shape", txt.shape)
    # print("txt_ids.shape", txt_ids.shape)
    # print("vec.shape", vec.shape)
    # print("vec_pose.shape", vec_pose.shape)

    return {
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }


def prepare_vanilla_t2i(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        customize_img_ids: bool = False,
                        customize_txt_ids: bool = False,
                        ) -> dict[str, Tensor]:
    # print("H_condition", h_condition)
    # print("W_condition", w_condition)
    # print("H_pose", h_pose)
    # print("W_pose", w_pose)
    # print("H_real", h_real)
    # print("W_real", w_real)

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    if customize_txt_ids:
        assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same


    # -------- Positional encodings for pose and image --------
    if customize_img_ids:
        assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
        img_ids_real = generate_img_ids_vanilla_t2i(
            bs,
            h_real,
            w_real,
            bounding_boxes_in_order,
            device,
        )
    else:
        img_ids_real_list = []

        for i in range(bs):
            # real
            h_r = int(h_real[i])
            w_r = int(w_real[i])
            # Image positional ids
            img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
            img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
            img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
            img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
            img_ids_real_list.append(img_ids_real)

        # Stack everything
        img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if customize_txt_ids:
        txt, txt_ids = process_prompt_with_split_ids_flux_changeall(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim
    else:
        txt = t5(prompt) # prompt is list of strings, list is batch dim
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)


    return {
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
    }



def prepare_dual_train_ar_version2(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # condition
                        h_condition,
                        w_condition,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        # pre_real
                        h_pre_real,
                        w_pre_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        customize_img_ids: bool = False,
                        customize_txt_ids: bool = False,
                        generate_img_ids_type: int = 1,
                        apply_smaller_pose: int = 1,
                        ) -> dict[str, Tensor]:
    # print("H_condition", h_condition)
    # print("W_condition", w_condition)
    # print("H_pose", h_pose)
    # print("W_pose", w_pose)
    # print("H_real", h_real)
    # print("W_real", w_real)

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    if customize_txt_ids:
        assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same


    # -------- Positional encodings for pose and image --------
    if customize_img_ids:
        assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
        generate_img_ids_func_dict = {1: generate_img_ids_version2, 2: generate_img_ids2_version2} # TODO 3 not implemented
        img_ids_pose, img_ids_condition, img_ids_real, img_ids_pre_real = generate_img_ids_func_dict[generate_img_ids_type](
            bs,
            h_condition,
            w_condition,
            h_pose,
            w_pose,
            h_real,
            w_real,
            h_pre_real,
            w_pre_real,
            pose_id_value,
            bounding_boxes_in_order,
            device,
            apply_smaller_pose
        )
    else:
        img_ids_pose_list = []
        img_ids_condition_list = []
        img_ids_real_list = []
        img_ids_pre_real_list = []

        for i in range(bs):
            # condition
            h_c = int(h_condition[i])
            w_c = int(w_condition[i])
            # pose
            h_p = int(h_pose[i])
            w_p = int(w_pose[i])
            # real
            h_r = int(h_real[i])
            w_r = int(w_real[i])
            # pre_real
            h_pr = int(h_pre_real[i])
            w_pr = int(w_pre_real[i])

            # condition
            img_ids_condition = torch.full((h_c // 2, w_c // 2, 3), pose_id_value)
            img_ids_condition[..., 1] += torch.arange(h_c // 2)[:, None]
            img_ids_condition[..., 2] += torch.arange(w_c // 2)[None, :]
            img_ids_condition = img_ids_condition.view(-1, 3)  # (h_c//2 * w_c//2, 3) to sequence
            img_ids_condition_list.append(img_ids_condition)


            # Pose positional ids
            pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
            pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
            pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
            pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
            img_ids_pose_list.append(pose_ids)

            # Image positional ids
            img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
            img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
            img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
            img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
            img_ids_real_list.append(img_ids_real)

            # Pre_real positional ids
            img_ids_pre_real = torch.zeros((h_pr // 2, w_pr // 2, 3))
            img_ids_pre_real[..., 1] += torch.arange(h_pr // 2)[:, None]
            img_ids_pre_real[..., 2] += torch.arange(w_pr // 2)[None, :]
            img_ids_pre_real = img_ids_pre_real.view(-1, 3)  # (h_pr//2 * w_pr//2, 3) to sequence
            img_ids_pre_real_list.append(img_ids_pre_real)

        # Stack everything
        img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
        img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)            # [bs, H_patches, 3]
        img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]
        img_ids_pre_real = torch.stack(img_ids_pre_real_list, dim=0).to(device)            # [bs, H_patches, 3]

    #### pre-save prompt_t5 #####
    if customize_txt_ids:
        txt, txt_ids = process_prompt_with_split_ids_flux(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim
    else:
        txt = t5(prompt) # prompt is list of strings, list is batch dim
        if txt.shape[0] == 1 and bs > 1:
            txt = repeat(txt, "1 ... -> bs ...", bs=bs)
        txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    # print("img_ids_condition.shape", img_ids_condition.shape)
    # print("img_ids_pose.shape", img_ids_pose.shape)
    # print("img_ids_real.shape", img_ids_real.shape)
    # print("txt.shape", txt.shape)
    # print("txt_ids.shape", txt_ids.shape)
    # print("vec.shape", vec.shape)
    # print("vec_pose.shape", vec_pose.shape)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "img_ids_pre_real": img_ids_pre_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }


def prepare_dual_train_layout(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        # customize_img_ids: bool = False,
                        # customize_txt_ids: bool = True,
                        # generate_img_ids_type: int = 1,
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]

    assert isinstance(prompt[0], list), "prompt must be a list of lists"
        # clip the same


    # -------- Positional encodings for pose and image --------

    img_ids_pose_list = []
    img_ids_real_list = []

    for i in range(bs):
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i])
        w_r = int(w_real[i])


        # Pose positional ids
        pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
        pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
        pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
        pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
        img_ids_pose_list.append(pose_ids)

        # Image positional ids
        img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
        img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
        img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
        img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]        # [bs, H_patches, 3]
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    txt, txt_ids = process_prompt_with_split_ids_flux(t5, prompt, bs) # prompt is list of lists of strings, outer list is batch dim

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }



def compute_cell_center_vectors(bbox_center, width: int, height: int):
    """
    For each grid cell (i,j), compute vector (dx, dy) from the cell center to the bbox center.
    - Coordinates in unit cells.
    - Grid has width x height cells; origin top-left; x→right, y→down.
    - Cell (i,j) center is at (i+0.5, j+0.5).
    - Bbox center is ((xmin+xmax)/2, (ymin+ymax)/2).

    Returns:
        dx: (H, W) array where dx[j, i] = bbox_cx - (i + 0.5)
        dy: (H, W) array where dy[j, i] = bbox_cy - (j + 0.5)
    """
    cx, cy = bbox_center

    # Build centers for all cells
    xs = torch.arange(width, dtype=torch.float32) + 0.5   # centers along x
    ys = torch.arange(height, dtype=torch.float32) + 0.5  # centers along y

    # Meshgrid with image-style indexing: i=x (cols), j=y (rows).
    # We'll produce arrays in shape (H, W) indexed dy[j,i], dx[j,i].
    # Meshgrid with image-style indexing: i=x (cols), j=y (rows).
    Xc, Yc = torch.meshgrid(xs, ys, indexing="xy")

    dx = cx - Xc
    dy = cy - Yc
    offsets = torch.stack([dy, dx], axis=-1)  # shape (H, W, 2)
    return offsets


def compute_cell_centers_from_offsets(bbox_center, offsets):
    """
    Given a bbox center (cx, cy) and an offset tensor (H, W, 2) where
    offsets[j, i] = (dx, dy) = (cx - cell_x, cy - cell_y),
    return a tensor (H, W, 2) of each cell center coordinate (x, y).

    Args:
        bbox_center: (cx, cy) tuple
        offsets: np.ndarray of shape (H, W, 2)

    Returns:
        centers: np.ndarray of shape (H, W, 2)
    """
    cx, cy = bbox_center
    dy = offsets[..., 0]
    #print("dy", dy)
    dx = offsets[..., 1]
    #print("dx", dx)

    cell_x = cx - dx
    cell_y = cy - dy

    centers = torch.stack([cell_y, cell_x], axis=-1)

    return centers


def compute_condition_img_ids(w, h, ori_center_encoded, new_center_encoded, resize_scale):
    # offsets of the distance of the bbox center to all cell centers in the new image
    offsets_new = compute_cell_center_vectors(bbox_center=new_center_encoded, width=w, height=h)
    # these offsets in the original image will be rescaled by 1/resize_scale, and we will move the bbox center to the old center
    centers = compute_cell_centers_from_offsets(bbox_center=ori_center_encoded, offsets=offsets_new/resize_scale)
    # keep two decimal places
    centers = torch.round(centers * 100) / 100

    h, w, _ = centers.shape

    # Create a zero channel
    zero_channel = torch.full((h, w, 1), 1, dtype=centers.dtype, device=centers.device)

    # Concatenate along the last dimension
    img_ids = torch.cat([zero_channel, centers], dim=-1)

    return img_ids

def prepare_dual_train_ar_pretend_pe(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        multi_person_last_indicator, # a list ofTrue or False
                        # condition
                        h_condition,
                        w_condition,
                        ori_center,
                        new_center,
                        resize_scale,
                        bounding_boxes_in_order, # a list of [x1, y1, x2, y2], use the first one for the first person
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        sample_mode: bool = False, # if True, directly pass in the s, t_x, t_y
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]


    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        multi_person_last_indicator_i = multi_person_last_indicator[i]
        # ori_center_i = ori_center[i]
        # new_center_i = new_center[i]
        # resize_scale_i = resize_scale[i]
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i])
        w_r = int(w_real[i])

        # condition
        if multi_person_last_indicator_i:
            if not sample_mode:
                ori_center_i = ori_center[i]
                new_center_i = new_center[i]
                resize_scale_i = resize_scale[i]
                ori_center_encoded = (ori_center_i[0] / 16, ori_center_i[1] / 16)
                new_center_encoded = (new_center_i[0] / 16, new_center_i[1] / 16)
                img_ids_condition = compute_condition_img_ids(w_c // 2, h_c // 2, ori_center_encoded, new_center_encoded, resize_scale_i)
            else:
                img_ids_condition = compute_condition_img_ids(w_c // 2, h_c // 2, (16, 32), (32, 32), 1.5)
            img_ids_condition = img_ids_condition.view(-1, 3)
            img_ids_condition_list.append(img_ids_condition)
        else:
            img_ids_condition = torch.full((h_c // 2, w_c // 2, 3), pose_id_value)
            img_ids_condition[..., 1] += torch.arange(h_c // 2)[:, None]
            img_ids_condition[..., 2] += torch.arange(w_c // 2)[None, :]
            img_ids_condition = img_ids_condition.view(-1, 3)  # (h_c//2 * w_c//2, 3) to sequence
            img_ids_condition_list.append(img_ids_condition)


        # Pose positional ids
        pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
        pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
        pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
        pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
        img_ids_pose_list.append(pose_ids)

        # Image positional ids
        img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
        img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
        img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
        img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)            # [bs, H_patches, 3]
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    txt = t5(prompt) # prompt is list of strings, list is batch dim
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }


def prepare_dual_train_ar_pretend_pe_learnable(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # condition
                        h_condition,
                        w_condition,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        pose_id_value: float = 1.0,
                        add_pose_clip: bool = True,
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]


    img_ids_pose_list = []
    img_ids_condition_list = []
    img_ids_real_list = []

    for i in range(bs):
        # condition
        h_c = int(h_condition[i])
        w_c = int(w_condition[i])
        # pose
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])
        # real
        h_r = int(h_real[i])
        w_r = int(w_real[i])

        # condition
        img_ids_condition = torch.full((h_c // 2, w_c // 2, 3), pose_id_value)
        img_ids_condition[..., 1] += torch.arange(h_c // 2)[:, None]
        img_ids_condition[..., 2] += torch.arange(w_c // 2)[None, :]
        img_ids_condition = img_ids_condition.view(-1, 3)  # (h_c//2 * w_c//2, 3) to sequence
        img_ids_condition_list.append(img_ids_condition)


        # Pose positional ids
        pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
        pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
        pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
        pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
        img_ids_pose_list.append(pose_ids)

        # Image positional ids
        img_ids_real = torch.zeros((h_r // 2, w_r // 2, 3))
        img_ids_real[..., 1] += torch.arange(h_r // 2)[:, None]
        img_ids_real[..., 2] += torch.arange(w_r // 2)[None, :]
        img_ids_real = img_ids_real.view(-1, 3)  # (h_r//2 * w_r//2, 3)
        img_ids_real_list.append(img_ids_real)

    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]
    img_ids_condition = torch.stack(img_ids_condition_list, dim=0).to(device)            # [bs, H_patches, 3]
    img_ids_real = torch.stack(img_ids_real_list, dim=0).to(device)            # [bs, H_patches, 3]


    txt = t5(prompt) # prompt is list of strings, list is batch dim
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None
    }


def prepare_dual_train_ar_embed_id(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # condition
                        h_condition,
                        w_condition,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        add_pose_clip: bool = True,
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]


    # -------- Positional encodings for pose and image --------
    assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
    img_ids_pose, img_ids_condition, img_ids_real, bbox_ids_pose, bbox_ids_condition, bbox_ids_real = generate_img_ids_for_embed_id(
        bs,
        h_condition,
        w_condition,
        h_pose,
        w_pose,
        h_real,
        w_real,
        bounding_boxes_in_order,
        device
    )

    #### pre-save prompt_t5 #####
    txt = t5(prompt) # prompt is list of strings, list is batch dim
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None,
        "bbox_ids_pose": bbox_ids_pose.to(device),
        "bbox_ids_condition": bbox_ids_condition.to(device),
        "bbox_ids_real": bbox_ids_real.to(device)
    }


def prepare_dual_train_ar_embed_id_realnotadded(t5: HFEmbedder,
                        clip: HFEmbedder,
                        bs,
                        # condition
                        h_condition,
                        w_condition,
                        # pose
                        h_pose,
                        w_pose,
                        # real
                        h_real,
                        w_real,
                        device,
                        prompt: str | list[str] | list[list[str]], # detailed_prompt
                        clip_prompt: str | list[str], # general_prompt
                        add_pose_clip: bool = True,
                        bounding_boxes_in_order: list[list[int]] | list[list[list[int]]] = None,
                        ) -> dict[str, Tensor]:

    if isinstance(prompt, str):
        prompt = [prompt]
    if isinstance(clip_prompt, str):
        clip_prompt = [clip_prompt]


    # -------- Positional encodings for pose and image --------
    assert bounding_boxes_in_order is not None, "bounding_boxes_in_order must be provided when customize_img_ids is True"
    img_ids_pose, img_ids_condition, img_ids_real, bbox_ids_pose, bbox_ids_condition, bbox_ids_real = generate_img_ids_for_embed_id_realnotadded(
        bs,
        h_condition,
        w_condition,
        h_pose,
        w_pose,
        h_real,
        w_real,
        bounding_boxes_in_order,
        device
    )

    #### pre-save prompt_t5 #####
    txt = t5(prompt) # prompt is list of strings, list is batch dim
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    vec = clip(clip_prompt)
    if add_pose_clip: # true
        vec_pose = clip(["Openpose image of " + prompt_i for prompt_i in clip_prompt])
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)
        if add_pose_clip:
            vec_pose = repeat(vec_pose, "1 ... -> bs ...", bs=bs)

    return {
        "img_ids_condition": img_ids_condition.to(device),
        "img_ids_pose": img_ids_pose.to(device),
        "img_ids_real": img_ids_real.to(device),
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "vec_pose": vec_pose.to(device) if add_pose_clip else None,
        "bbox_ids_pose": bbox_ids_pose.to(device),
        "bbox_ids_condition": bbox_ids_condition.to(device),
        "bbox_ids_real": bbox_ids_real.to(device)
    }

def prepare_t2p_train(t5: HFEmbedder,
                       clip: HFEmbedder,
                       bs_pose,
                       c_pose,
                       h_pose,
                       w_pose,
                       device,
                       prompt: str | list[str],
                       prompt_vec: Tensor = None,
                       prompt_t5: Tensor = None,
                       pose_id_value: float = 0.0,
                       add_h: bool = True) -> dict[str, Tensor]:
    if isinstance(prompt, str):
        prompt = [prompt]


    # -------- Positional encodings for pose and image --------
    img_ids_pose_list = []

    for i in range(bs_pose):
        h_p = int(h_pose[i])
        w_p = int(w_pose[i])


        # Pose positional ids
        pose_ids = torch.full((h_p // 2, w_p // 2, 3), pose_id_value)
        pose_ids[..., 1] += torch.arange(h_p // 2)[:, None]
        pose_ids[..., 2] += torch.arange(w_p // 2)[None, :]
        pose_ids = pose_ids.view(-1, 3)  # (h_p//2 * w_p//2, 3) to sequence
        img_ids_pose_list.append(pose_ids)


    # Stack everything
    img_ids_pose = torch.stack(img_ids_pose_list, dim=0).to(device)  # [bs, H_patches, 3]


    #### pre-save prompt_t5 #####
    if prompt_t5 is not None:
        txt = prompt_t5
    else:
        txt = t5(prompt)
    #### pre-save prompt_t5 #####
    if txt.shape[0] == 1 and bs_pose > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs_pose)
    txt_ids = torch.zeros(bs_pose, txt.shape[1], 3) # eg. [bs, 512, 3], every 3 will be all 0s

    #### pre-save prompt_vec #####
    if prompt_vec is not None:
        vec = prompt_vec
    else:
        vec = clip(prompt)
    #### pre-save prompt_vec #####
    if vec.shape[0] == 1 and bs_pose > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs_pose)

    return {
        "txt": txt.to(device),
        "txt_ids": txt_ids.to(device),
        "vec": vec.to(device),
        "img_ids_pose": img_ids_pose.to(device),
    }



def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # ip-adapter parameters
    image_proj: Tensor=None,
    ip_scale: Tensor | float = 1.0,
    use_attn_mask: bool = False,
    pose_only: bool = False,
    single_stream: bool = False,
    vec_pose: Tensor | None = None,
    attention_mask: Tensor | None = None,
    doubleblock_apply_mask_from: int = 0,
    apply_mask_from_denoise_step: int = 0,
):
    i = 0
    if i < apply_mask_from_denoise_step:
        attention_mask_to_apply = None
    else:
        if attention_mask is not None:
            attention_mask_to_apply = attention_mask.clone()
        else:
            attention_mask_to_apply = None
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in tqdm(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            y_pose=vec_pose,
            timesteps=t_vec,
            timesteps_pose=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
            use_attn_mask=use_attn_mask,
            pose_only=pose_only,
            single_stream=single_stream,
            single_stream_attention_mask=attention_mask_to_apply,
            doubleblock_apply_mask_from=doubleblock_apply_mask_from,
        )
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img

def denoise_controlnet(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None,
    neg_image_proj: Tensor=None,
    ip_scale: Tensor | float = 1,
    neg_ip_scale: Tensor | float = 1,
):
    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        if i >= timestep_to_start_cfg:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond,
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale,
            )
            pred = neg_pred + true_gs * (pred - neg_pred)

        img = img + (t_prev - t_curr) * pred

        i += 1
    return img

def create_black_image_latents(ae, height: int, width: int, device: torch.device, dtype: torch.dtype = torch.bfloat16):
    """
    Create black image latents by encoding a black image through VAE.

    This function is used to create the initial condition for iterative generation.
    The black image represents "no information" and serves as a starting point.

    Args:
        ae: VAE model for encoding
        height: Image height in pixels
        width: Image width in pixels
        device: Device to place the output tensor on
        dtype: Data type for the output tensor

    Returns:
        Tensor of shape (num_patches, patch_dim) representing black image latents
    """
    # Create black image tensor on the same device as VAE encoder
    vae_device = next(ae.encoder.parameters()).device
    black_img = torch.zeros(1, 3, height, width, device=vae_device, dtype=torch.float32)

    with torch.no_grad():
        # Encode through VAE
        latents = ae.encode(black_img)

        # Patchify for transformer input
        latents_patchified = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    return latents_patchified.squeeze(0).to(dtype)  # Remove batch dimension


def denoise_ar(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None,
    neg_image_proj: Tensor=None,
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0,
    use_attn_mask: bool = False,
    vec_pose: Tensor | None = None,
    # iterative-generation-specific parameters
    image_sequence_len: int = 1024,
    cond_sequence_len: int = 1024,
    pose_sequence_len: int = 1024,
):
    """
    Iterative denoising function that handles [condition, pose, real] sequence.

    Input sequence structure:
    - img: [condition_latents, pose_noise, real_noise] concatenated
    - condition_latents: Either black latents (round 1) or previous pose output (round 2+)
    - pose_noise: Random noise for pose generation
    - real_noise: Random noise for real image generation

    Output:
    - Denoised sequence: [condition_output, pose_output, real_output]
    - condition_output: Should be same as input condition (no noise added)
    - pose_output: Generated pose latents (used as condition for next round)
    - real_output: Generated real image latents
    """
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # Create separate timesteps for different sequence parts
        t_pose = t_vec.clone()      # Pose gets noise schedule
        t_real = t_vec.clone()      # Real gets noise schedule
        t_cond = torch.full_like(t_vec, timesteps[-1])  # Condition always at timestep 999 (no noise)

        # Forward pass through model
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            y_pose=vec_pose,
            timesteps=t_real,           # For real image generation
            timesteps_pose=t_pose,      # For pose generation
            timesteps_cond=t_cond,      # For condition (always no noise)
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
            use_attn_mask=use_attn_mask,
            image_sequence_len=image_sequence_len,
            cond_sequence_len=cond_sequence_len,
        )

        # Update sequence using Euler method
        img = img + (t_prev - t_curr) * pred
        i += 1

    return img

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    # print("unpack-----")
    # print("x.shape", x.shape)
    # print("height", height)
    # print("width", width)
    # print("unpack-----")
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
