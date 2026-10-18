from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams):
        super().__init__()

        self.mockpe = None # will be set in v0_ar_triplelora_unlimited_canvas_w_indicator_w_pretend_pe_learnable.py

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor, # cated
        img_ids: Tensor, # cated
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        y_pose: Tensor | None = None,
        block_controlnet_hidden_states=None,
        guidance: Tensor | None = None,
        image_proj: Tensor | None = None, 
        ip_scale: Tensor | float = 1.0,
        ## separated vectors -- may be None, if none, use same modulation
        timesteps_pose: Tensor | None = None,
        timesteps_cond: Tensor | None = None, # for ar always timesteps[999]
        timesteps_pre_real: Tensor | None = None, # for ar always timesteps[999] # only use for version2
        image_sequence_len: int = 1024,
        cond_sequence_len: int = 1024,
        ## random_masked
        random_masked: bool = False,
        force_attn_mask: bool = False,
        ## inference 单独写
        sample_mode: bool = False,
        ## for t2p
        pose_only: bool = False,
        single_stream: bool = False,
        ## parse use_attn_mask here
        use_attn_mask: bool = False,
        # for batch-wise use_attn_mask
        real_image_active: Tensor | None = None,
        ar_attn_mask: Tensor | None = None,
        ### embed id ###########################
        embed_id: bool = False,
        bbox_ids: Tensor | None = None,
        ### embed id ###########################
        single_stream_attention_mask: Tensor | None = None,
        doubleblock_apply_mask_from: int = 0,
        ####&&&&&& add for learnable pe &&&&&&####
        pe_learnable: bool = False,
        batch_shift: Tensor | None = None,
        batch_resize_scale: Tensor | None = None,
        batch_multi_person_last_indicator: Tensor | None = None,
        learned_shift_scale_ratio: float = None,
        ori_h: Tensor | None = None,
        ori_w: Tensor | None = None,
        ####&&&&&& add for learnable pe &&&&&&####
        return_embeddings: bool = False,
        version1: bool = False,
    ) -> Tensor:
        if embed_id:
            assert bbox_ids is not None, "bbox_ids must be provided when embed_id is True"
            assert bbox_ids.shape[1] == img_ids.shape[1], "bbox_ids.shape[1] must be equal to img_ids.shape[1]"
            
        # print("DEBUG -----------------------------------")
        # print("img.shape", img.shape)
        # print("img_ids.shape", img_ids.shape)
        # print("txt.shape", txt.shape)
        # print("txt_ids.shape", txt_ids.shape)
        # print("timesteps.shape", timesteps.shape)
        # print("DEBUG -----------------------------------")
        if pose_only or single_stream:
            timesteps_pose = None

        for block in self.double_blocks:
            block.processor.use_attn_mask = use_attn_mask
            block.processor.real_image_active = real_image_active
        for block in self.single_blocks:
            block.processor.use_attn_mask = use_attn_mask
            block.processor.real_image_active = real_image_active

        if random_masked:
            for block in self.double_blocks:
                block.processor.use_attn_mask = force_attn_mask
            for block in self.single_blocks:
                block.processor.use_attn_mask = force_attn_mask
        if not sample_mode: # training and current sampling   
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            # running on sequences img
            if pose_only:
                img = self.img_in.forward_pose_only(img)
            else:
                if single_stream:
                    img = self.img_in(img)
                else: # if img_in is class ImgInSplit
                    if version1:
                        img = self.img_in.forward_version1(img, img_seq_len=image_sequence_len)
                    else:
                        img = self.img_in(img, img_seq_len=image_sequence_len)

            # print("DEBUG -----------------------------------")
            # print("after img_in img.shape", img.shape)
            # print("DEBUG -----------------------------------")
            vec = self.time_in(timestep_embedding(timesteps, 256))
            vec_pose = None
            vec_cond = None
            vec_pre_real = None
            if timesteps_pose is not None:
                vec_pose = self.time_in(timestep_embedding(timesteps_pose, 256))
                if timesteps_cond is not None:
                    vec_cond = self.time_in(timestep_embedding(timesteps_cond, 256))
                if timesteps_pre_real is not None:
                    vec_pre_real = self.time_in(timestep_embedding(timesteps_pre_real, 256))
            if self.params.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
                if timesteps_pose is not None:
                    vec_pose = vec_pose + self.guidance_in(timestep_embedding(guidance, 256))
                    if timesteps_cond is not None:
                        vec_cond = vec_cond + self.guidance_in(timestep_embedding(guidance, 256))
                    if timesteps_pre_real is not None:
                        vec_pre_real = vec_pre_real + self.guidance_in(timestep_embedding(guidance, 256))
            vec = vec + self.vector_in(y)
            if timesteps_pre_real is not None:
                vec_pre_real = vec_pre_real + self.vector_in(y)
            if timesteps_pose is not None:
                if y_pose is not None:
                    vec_pose = vec_pose + self.vector_in(y_pose)
                    if timesteps_cond is not None:
                        vec_cond = vec_cond + self.vector_in(y_pose)
                else:
                    vec_pose = vec_pose + self.vector_in(y)
                    if timesteps_cond is not None:
                        vec_cond = vec_cond + self.vector_in(y)
            txt = self.txt_in(txt)

            ids = torch.cat((txt_ids, img_ids), dim=1)

            if pe_learnable:
                pe = ids # plain ids [B, L, 3] L = 512 + 4096 + 4096 + 4096
            else: # by default
                pe = self.pe_embedder(ids)
            
            ##### add bbox_id_embedder ############################################################
            if embed_id:
                B, L = bbox_ids.shape
                valid_id_mask = bbox_ids != -1
                valid_ids = bbox_ids[valid_id_mask]
                valid_embeddings = self.bbox_id_embedder(valid_ids) # added in the script
                bbox_id_embeddings = torch.zeros(
                    B, L, self.params.axes_dim[0], device=img.device, dtype=img.dtype
                )
                bbox_id_embeddings[valid_id_mask] = valid_embeddings
            else:
                bbox_id_embeddings = None
             ##### add bbox_id_embedder ############################################################



            if block_controlnet_hidden_states is not None:
                controlnet_depth = len(block_controlnet_hidden_states)

            # print("DEBUG -----------------------------------")
            # print("image_sequence_len", image_sequence_len)
            # print("cond_sequence_len", cond_sequence_len)
            # print("DEBUG -----------------------------------")

            for index_block, block in enumerate(self.double_blocks):
                if index_block < doubleblock_apply_mask_from:
                    double_block_single_stream_attention_mask = None
                else:
                    if single_stream_attention_mask is not None:
                        double_block_single_stream_attention_mask = single_stream_attention_mask.clone()
                    else:
                        double_block_single_stream_attention_mask = None
                
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    image_proj=image_proj,
                    ip_scale=ip_scale, 
                    vec_pose=vec_pose,
                    vec_cond=vec_cond,
                    vec_pre_real=vec_pre_real,
                    image_sequence_len=image_sequence_len,
                    cond_sequence_len=cond_sequence_len,
                    ar_attn_mask=ar_attn_mask,
                    bbox_id_embeddings=bbox_id_embeddings,
                    single_stream_attention_mask=double_block_single_stream_attention_mask,
                    layer_index=index_block,
                    mockpe=self.mockpe,
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    ####&&&&&& add for learnable pe &&&&&&####
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ori_h=ori_h,
                    ori_w=ori_w,
                    version1=version1
                )
                # controlnet residual
                if block_controlnet_hidden_states is not None:
                    img = img + block_controlnet_hidden_states[index_block % 2]

            # print("DEBUG -----------------------------------")
            # print("after double_blocks img.shape", img.shape)
            # print("DEBUG -----------------------------------")

            img = torch.cat((txt, img), 1)
            for block in self.single_blocks:
                img = block(img, vec=vec, pe=pe, vec_pose=vec_pose, image_sequence_len=image_sequence_len, vec_cond=vec_cond, vec_pre_real=vec_pre_real, cond_sequence_len=cond_sequence_len, ar_attn_mask=ar_attn_mask, bbox_id_embeddings=bbox_id_embeddings, single_stream_attention_mask=single_stream_attention_mask, mockpe=self.mockpe, 
                    ####&&&&&& add for learnable pe &&&&&&####
                    pe_learnable=pe_learnable,
                    batch_shift=batch_shift,
                    batch_resize_scale=batch_resize_scale,
                    batch_multi_person_last_indicator=batch_multi_person_last_indicator,
                    learned_shift_scale_ratio=learned_shift_scale_ratio,
                    ####&&&&&& add for learnable pe &&&&&&####
                    ori_h=ori_h,
                    ori_w=ori_w,
                    version1=version1
                )
                #check_tensor("img", img)
            if return_embeddings:
                img = self.final_layer(img, vec, return_embeddings=True)
                return img, pe
            
            img = img[:, txt.shape[1] :, ...]

            if timesteps_pose is not None:
                img = self.final_layer(img, vec, vec_pose, img_seq_len=image_sequence_len) 
            else:
                if pose_only: # for t2p
                    img = self.final_layer.forward_pose_only(img, vec)
                elif single_stream: # for single stream sampling
                    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
                else: # for dual sampling
                    if version1:
                        img = self.final_layer.forward_version1(img, vec, vec, img_seq_len=image_sequence_len)
                    else:
                        img = self.final_layer(img, vec, vec, img_seq_len=image_sequence_len)  # (N, T, patch_size ** 2 * out_channels)
            return img


def check_tensor(name, x):
    return
