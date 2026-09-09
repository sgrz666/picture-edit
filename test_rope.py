from transformers import Qwen2_5_VLForConditionalGeneration
import torch

m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/vlm", 
    torch_dtype=torch.bfloat16
).to("cuda")

# Test forward with get_rope_index
input_ids = torch.tensor([[100, 200, 300]], device="cuda")
attention_mask = torch.tensor([[1, 1, 1]], device="cuda", dtype=torch.bool)
pos_ids, _ = m.model.get_rope_index(input_ids, image_grid_thw=None, video_grid_thw=None, second_per_grid_ts=None, attention_mask=attention_mask)
print("pos_ids shape:", pos_ids.shape)

embeds = m.language_model.get_input_embeddings()(input_ids)
out = m.language_model(inputs_embeds=embeds, attention_mask=attention_mask, position_ids=pos_ids, output_hidden_states=True)
print("out last_hidden_state shape:", out.last_hidden_state.shape)
print("hidden_states min/max/mean/std:", out.last_hidden_state.min().item(), out.last_hidden_state.max().item(), out.last_hidden_state.mean().item(), out.last_hidden_state.std().item())