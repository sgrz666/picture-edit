import os
import gc
import glob
import time
import json
import torch
import inspect
import traceback
from PIL import Image
from diffusers import DiffusionPipeline

MODEL_PATH = "/root/autodl-tmp/projects/picture-edit/models/DeepGen-1.0-diffusers"
OUTDIR = "/root/autodl-tmp/projects/picture-edit/smoke_test_outputs"
os.makedirs(OUTDIR, exist_ok=True)

def print_sep(title):
    print("\n" + "=" * 20 + f" {title} " + "=" * 20)

def mem_gb(x):
    return round(x / 1024**3, 3)

def cuda_mem_report(tag):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        alloc = mem_gb(torch.cuda.memory_allocated())
        reserved = mem_gb(torch.cuda.memory_reserved())
        peak = mem_gb(torch.cuda.max_memory_allocated())
        print(f"[{tag}] CUDA allocated={alloc} GB, reserved={reserved} GB, peak={peak} GB")

def save_result_image(result, save_path):
    imgs = None
    if hasattr(result, "images"):
        imgs = result.images
    elif isinstance(result, dict) and "images" in result:
        imgs = result["images"]

    if imgs and len(imgs) > 0:
        img = imgs[0]
        if not isinstance(img, Image.Image):
            raise TypeError(f"Output image is not PIL.Image, got {type(img)}")
        img.save(save_path)
        print(f"Saved image to: {save_path}")
        return True
    else:
        print("No images found in pipeline output.")
        print("Output type:", type(result))
        print("Output repr:", repr(result)[:1000])
        return False

def find_test_image():
    candidates = [
        "/root/autodl-tmp/projects/picture-edit/manual_test_apple.png",
        "/root/autodl-tmp/projects/picture-edit/person1.jpg",
        "/root/autodl-tmp/projects/picture-edit/dual_person.jpg",
        "/root/autodl-tmp/projects/picture-edit/native_input_src.png",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    exts = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    found = []
    for ext in exts:
        found.extend(glob.glob(f"/root/autodl-tmp/projects/picture-edit/{ext}"))
        found.extend(glob.glob(f"/root/autodl-tmp/projects/picture-edit/demooutput/{ext}"))
        found.extend(glob.glob(f"/root/autodl-tmp/projects/picture-edit/results/**/*.png", recursive=True))
        found.extend(glob.glob(f"/root/autodl-tmp/projects/picture-edit/results/**/*.jpg", recursive=True))

    for p in found:
        if os.path.isfile(p):
            return p
    return None

def try_generate(pipe):
    print_sep("T2I SMOKE TEST")
    prompt = "A young woman standing outdoors, natural lighting, realistic photo"

    attempts = [
        {
            "name": "t2i_try_1",
            "kwargs": dict(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=4.0,
                height=512,
                width=512,
                generator=torch.Generator(device="cpu").manual_seed(42),
            ),
        },
        {
            "name": "t2i_try_2",
            "kwargs": dict(
                prompt=prompt,
                num_inference_steps=4,
                guidance_scale=4.0,
                generator=torch.Generator(device="cpu").manual_seed(42),
            ),
        },
        {
            "name": "t2i_try_3",
            "kwargs": dict(
                prompt=prompt,
                generator=torch.Generator(device="cpu").manual_seed(42),
            ),
        },
    ]

    for i, item in enumerate(attempts, 1):
        print(f"\n--- Attempt {i}: {item['name']} ---")
        print("Kwargs:", item["kwargs"])
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            result = pipe(**item["kwargs"])
            dt = time.time() - t0

            save_path = os.path.join(OUTDIR, f"{item['name']}.png")
            ok = save_result_image(result, save_path)
            cuda_mem_report(item["name"])
            print(f"{item['name']} runtime: {dt:.2f}s")
            if ok:
                return True, save_path
        except Exception as e:
            print(f"{item['name']} failed:")
            traceback.print_exc()

    return False, None

def try_edit(pipe):
    print_sep("IMAGE EDIT SMOKE TEST")
    test_image_path = find_test_image()
    if test_image_path is None:
        print("No candidate input image found for editing test.")
        return False, None

    print("Using input image:", test_image_path)
    image = Image.open(test_image_path).convert("RGB").resize((512, 512))

    edit_instruction = "Make the person smile and raise one hand. Preserve identity and overall composition as much as possible."

    attempts = [
        {
            "name": "edit_try_1",
            "kwargs": dict(
                prompt=edit_instruction,
                image=image,
                num_inference_steps=4,
                guidance_scale=4.0,
                generator=torch.Generator(device="cpu").manual_seed(123),
            ),
        },
        {
            "name": "edit_try_2",
            "kwargs": dict(
                instruction=edit_instruction,
                image=image,
                num_inference_steps=4,
                guidance_scale=4.0,
                generator=torch.Generator(device="cpu").manual_seed(123),
            ),
        },
        {
            "name": "edit_try_3",
            "kwargs": dict(
                prompt=edit_instruction,
                input_image=image,
                num_inference_steps=4,
                guidance_scale=4.0,
                generator=torch.Generator(device="cpu").manual_seed(123),
            ),
        },
        {
            "name": "edit_try_4",
            "kwargs": dict(
                prompt=edit_instruction,
                init_image=image,
                num_inference_steps=4,
                guidance_scale=4.0,
                generator=torch.Generator(device="cpu").manual_seed(123),
            ),
        },
        {
            "name": "edit_try_5",
            "kwargs": dict(
                prompt=edit_instruction,
                image=image,
                generator=torch.Generator(device="cpu").manual_seed(123),
            ),
        },
    ]

    for i, item in enumerate(attempts, 1):
        print(f"\n--- Attempt {i}: {item['name']} ---")
        print("Kwargs keys:", list(item["kwargs"].keys()))
        try:
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.reset_peak_memory_stats()

            t0 = time.time()
            result = pipe(**item["kwargs"])
            dt = time.time() - t0

            save_path = os.path.join(OUTDIR, f"{item['name']}.png")
            ok = save_result_image(result, save_path)
            cuda_mem_report(item["name"])
            print(f"{item['name']} runtime: {dt:.2f}s")
            if ok:
                return True, save_path
        except Exception as e:
            print(f"{item['name']} failed:")
            traceback.print_exc()

    return False, None

def main():
    print_sep("ENV INFO")
    print("torch version:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    print_sep("LOAD PIPELINE")
    print("Loading from:", MODEL_PATH)
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    print("Pipeline type:", type(pipe))

    try:
        sig = inspect.signature(pipe.__call__)
        print("Pipeline __call__ signature:", sig)
    except Exception as e:
        print("Could not inspect signature:", e)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        cuda_mem_report("after pipe.to(cuda)")

    # 尽量减少干扰
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)

    # 先做 T2I
    t2i_ok, t2i_path = try_generate(pipe)

    # 再做 Editing
    edit_ok, edit_path = try_edit(pipe)

    print_sep("SUMMARY")
    summary = {
        "t2i_ok": t2i_ok,
        "t2i_path": t2i_path,
        "edit_ok": edit_ok,
        "edit_path": edit_path,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
