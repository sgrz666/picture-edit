"""Native Qwen-Image-2.1 image/latent I/O for the full V6 migration.

Copied from the existing Qwen runtime only; no simplified control modules are used."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image


MODEL_REVISION = "790c92633540aa0cb11d9abf19eb46d861714758"
PROMPT = "Change the person pose to match the target pose. Preserve identity, clothing, and background."
RESOLUTION = 512


def dataset_image_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert the dataset's normalized RGB tensor into an opaque RGB image."""
    return to_pil_image(((image.detach().float().cpu() + 1) * 0.5).clamp(0, 1))


class Qwen21IPERCache:
    def __init__(self, root: str | Path, *, prompt: str = PROMPT, resolution: int | str = RESOLUTION,
                 preprocessing: str = 'resize_v1') -> None:
        if resolution not in (512, 1024, '512', '1024', 'native'):
            raise ValueError("resolution must be 512, 1024, or 'native'")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.prompt = prompt
        self.resolution = resolution
        self.preprocessing = preprocessing
        if preprocessing != 'resize_v1':
            import json
            policy = {'preprocessing': preprocessing, 'resolution': resolution,
                      'model_revision': MODEL_REVISION, 'prompt': prompt}
            path = self.root / 'cache_policy.json'
            if path.exists() and json.loads(path.read_text()) != policy:
                raise ValueError('Cache preprocessing policy differs; use a new cache directory')
            if not path.exists():
                path.write_text(json.dumps(policy,indent=2)+'\n')

    def _path(self, image: Image.Image, kind: str, *, prompt: str | None = None) -> Path:
        resolved_prompt = self.prompt if prompt is None or kind == "target" else prompt
        digest = hashlib.sha256()
        digest.update((MODEL_REVISION + resolved_prompt + kind + str(self.resolution)).encode())
        if self.preprocessing != 'resize_v1':
            digest.update(self.preprocessing.encode())
        digest.update(image.convert("RGB").tobytes())
        return self.root / f"{kind}_{digest.hexdigest()[:24]}.pt"

    @staticmethod
    def _store(path: Path, value: object) -> None:
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        torch.save(value, temporary)
        temporary.replace(path)

    def _vae_input(self, pipe, image: Image.Image, device: torch.device) -> torch.Tensor:
        rgba = image.convert("RGBA")
        width = image.width if self.resolution == 'native' else int(self.resolution)
        height = image.height if self.resolution == 'native' else int(self.resolution)
        return pipe.image_processor.preprocess(
            rgba, width=width, height=height
        ).unsqueeze(2).to(device=device, dtype=pipe.vae.dtype)

    def source(self, pipe, image: Image.Image, *, prompt: str | None = None) -> dict[str, torch.Tensor]:
        resolved_prompt = self.prompt if prompt is None else prompt
        path = self._path(image, "source", prompt=resolved_prompt)
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        device = next(pipe.text_encoder.parameters()).device
        if device.type == "cpu":
            raise RuntimeError(f"missing source cache {path.name}; text encoder must be on GPU for precompute")
        with torch.inference_mode():
            rgba = image.convert("RGBA")
            width = image.width if self.resolution == 'native' else int(self.resolution)
            height = image.height if self.resolution == 'native' else int(self.resolution)
            vlm_image = pipe.image_processor.resize(rgba, width=width, height=height)
            prompt_embeds, prompt_mask, image_mask = pipe.encode_prompt(
                resolved_prompt, image=[vlm_image], device=device
            )
            vae_image = self._vae_input(pipe, rgba, device)
            latent = pipe._encode_vae_image(vae_image, torch.Generator(device=device).manual_seed(0))
            latent = latent.flatten(3).squeeze(2).transpose(1, 2)
            if prompt_mask is None:
                prompt_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.bool)
            entry = {
                "prompt_embeds": prompt_embeds.cpu().contiguous(),
                "prompt_mask": prompt_mask.cpu().bool().contiguous(),
                "image_mask": image_mask.cpu().bool().contiguous(),
                "reference_latent": latent.cpu().contiguous(),
                "token_hw": (height // 16, width // 16),
            }
        self._store(path, entry)
        return entry

    def target(self, pipe, image: Image.Image) -> torch.Tensor:
        path = self._path(image, "target")
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        device = next(pipe.vae.parameters()).device
        if device.type == "cpu":
            raise RuntimeError(f"missing target cache {path.name}; VAE must be on GPU for precompute")
        with torch.inference_mode():
            latent = pipe._encode_vae_image(
                self._vae_input(pipe, image, device), torch.Generator(device=device).manual_seed(0)
            )
            latent = latent.flatten(3).squeeze(2).transpose(1, 2).cpu().contiguous()
        self._store(path, latent)
        return latent

    def pair(
        self, pipe, batch: dict, device: torch.device, *, prompts: Sequence[str] | None = None
    ) -> dict[str, torch.Tensor]:
        batch_size = batch["src_image"].shape[0]
        if prompts is None:
            conditions = [self.source(pipe, dataset_image_to_pil(batch["src_image"][index])) for index in range(batch_size)]
        else:
            if len(prompts) != batch_size:
                raise ValueError(f"prompts length {len(prompts)} does not match batch size {batch_size}")
            conditions = [
                self.source(pipe, dataset_image_to_pil(batch["src_image"][index]), prompt=prompts[index])
                for index in range(batch_size)
            ]
        targets = [self.target(pipe, dataset_image_to_pil(batch["tgt_image"][index])) for index in range(batch_size)]
        target_latent = torch.cat(targets, dim=0)
        tokens = target_latent.shape[1]
        if self.resolution == 'native':
            tgt_h, tgt_w = batch["tgt_image"].shape[2] // 16, batch["tgt_image"].shape[3] // 16
            src_h, src_w = batch["src_image"].shape[2] // 16, batch["src_image"].shape[3] // 16
            if tokens != tgt_h * tgt_w or tokens % 4:
                raise ValueError(f"unexpected Qwen21 target token count: {tokens} (expected {tgt_h}x{tgt_w}={tgt_h * tgt_w})")
        else:
            res_int = int(self.resolution)
            tgt_h = tgt_w = src_h = src_w = res_int // 16
            if tokens != (res_int // 16) ** 2 or tokens % 4:
                raise ValueError(f"unexpected Qwen21 target token count: {tokens}")
        max_prompt = max(condition["prompt_embeds"].shape[1] for condition in conditions)
        prompt_embeds = torch.cat([
            F.pad(condition["prompt_embeds"], (0, 0, 0, max_prompt - condition["prompt_embeds"].shape[1]))
            for condition in conditions
        ], dim=0)
        prompt_mask = torch.cat([
            F.pad(condition["prompt_mask"], (0, max_prompt - condition["prompt_mask"].shape[1]), value=False)
            for condition in conditions
        ], dim=0)
        image_mask = torch.cat([
            torch.cat((
                F.pad(condition["image_mask"], (0, max_prompt - condition["image_mask"].shape[1]), value=False),
                torch.ones(1, tokens // 4, dtype=torch.bool),
            ), dim=1) for condition in conditions
        ], dim=0)
        return {
            "source_latent": torch.cat([condition["reference_latent"] for condition in conditions], dim=0).to(device),
            "target_latent": target_latent.to(device),
            "prompt_embeds": prompt_embeds.to(device),
            "prompt_mask": prompt_mask.to(device),
            "img_mask": image_mask.to(device),
            "resolution": self.resolution,
            "target_token_hw": (tgt_h, tgt_w),
            "source_token_hw": (src_h, src_w),
        }


def transformer_inputs(pair: dict[str, torch.Tensor], noisy_target: torch.Tensor, sigma: torch.Tensor):
    if pair.get('target_token_hw') is not None:
        th, tw = pair['target_token_hw']
        sh, sw = pair.get('source_token_hw', (th, tw))
    else:
        grid = int(pair['resolution']) // 16
        th = tw = sh = sw = grid
    if noisy_target.shape[1] != th * tw:
        raise ValueError('target latent token count does not match resolution')
    return {
        "hidden_states": torch.cat((pair["source_latent"], noisy_target), dim=1),
        "timestep": sigma,
        "encoder_hidden_states": pair["prompt_embeds"],
        "encoder_hidden_states_mask": pair["prompt_mask"],
        "img_shapes": [[(1, sh, sw), (1, th, tw)] for _ in range(noisy_target.shape[0])],
        "img_mask": pair["img_mask"],
        "return_dict": False,
    }
