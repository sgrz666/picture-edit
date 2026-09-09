from transformers import Qwen2_5_VLForConditionalGeneration
m = Qwen2_5_VLForConditionalGeneration.from_pretrained("/home/shangguanrz/project/pic-edit/models/DeepGen-1.0-diffusers/vlm", torch_dtype="auto")
print("has language_model?", hasattr(m, "language_model"))
print("has model?", hasattr(m, "model"))
print("type of model:", type(m.model))
if hasattr(m, "language_model"):
    print("type of language_model:", type(m.language_model))
if hasattr(m.model, "layers"):
    print("m.model num layers:", len(m.model.layers))