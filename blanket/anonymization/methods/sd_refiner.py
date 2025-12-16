import torch
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from typing import Optional


class SDRefiner:
    """Stable Diffusion XL refiner for high-quality output."""

    def __init__(
        self,
        enabled: bool,
        refiner_model_id: str,
        device: str,
        switch_at: float,
        use_fp32: bool = True,
    ):
        self.enabled = enabled
        self.refiner_model_id = refiner_model_id
        self.device = device
        self.switch_at = switch_at
        self.use_fp32 = use_fp32
        self.refiner_pipe = None

    def load(self):
        if not self.enabled:
            return

        if self.refiner_pipe is not None:
            return

        self.refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.refiner_model_id,
            torch_dtype=torch.float32 if self.use_fp32 else torch.float16,
        )

        #  cpu offload for CUDA to save memory
        if self.device == "cuda":
            self.refiner_pipe.enable_sequential_cpu_offload()
            self.refiner_pipe.enable_attention_slicing(1)
            self.refiner_pipe.enable_vae_slicing()
        else:
            self.refiner_pipe = self.refiner_pipe.to(self.device)

    def unload(self):
        if self.refiner_pipe is not None:
            del self.refiner_pipe
            self.refiner_pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def refine(
        self,
        base_output: Image.Image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        mask: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        Refine the base output image.

        If mask is provided, only refines the masked region and composites it back.
        This preserves areas outside the mask (e.g., background, hair, clothing).
        """
        if not self.enabled:
            return base_output

        if self.refiner_pipe is None:
            self.load()

        refiner_steps = max(1, int(num_inference_steps * (1.0 - self.switch_at)))

        refined_output = self.refiner_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_output,
            num_inference_steps=refiner_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        if mask is not None:
            mask_array = np.array(mask.convert('L')).astype(np.float32) / 255.0
            mask_array = np.expand_dims(mask_array, axis=-1)
            base_array = np.array(base_output).astype(np.float32)
            refined_array = np.array(refined_output).astype(np.float32)
            composited = refined_array * mask_array + base_array * (1.0 - mask_array)
            composited = composited.astype(np.uint8)

            return Image.fromarray(composited)

        return refined_output