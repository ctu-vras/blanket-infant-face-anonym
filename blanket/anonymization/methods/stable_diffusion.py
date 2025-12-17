import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import yaml

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    AutoencoderKL
)

from blanket.anonymization.methods.controlnet_preprocessors import (
    CannyPreprocessor,
    OpenPosePreprocessor
)
from blanket.anonymization.methods.sd_refiner import SDRefiner


class StableDiffusionAnonymizer:
    """Generate synthetic face identities using SDXL with ControlNet."""

    def __init__(self, config_path=None, device=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "module_parameters" / "stable_diffusion_parameters.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        self.model_id = self.config.get('model_id', 'diffusers/stable-diffusion-xl-1.0-inpainting-0.1')
        self.prompt = self.config.get('prompt', 'high quality photo of a baby face')
        self.negative_prompt = self.config.get('negative_prompt', '')
        self.steps = self.config.get('steps', 50)
        self.cfg_scale = self.config.get('cfg_scale', 4.5)
        self.denoising_strength = self.config.get('denoising_strength', 0.7)
        self.seed = self.config.get('seed', 1)
        self.mask_blur = self.config.get('mask_blur', 4)

        self.use_controlnet = self.config.get('use_controlnet', True)
        self.controlnet_configs = self.config.get('controlnet_models', [])

        self.use_refiner = self.config.get('use_refiner', True)
        self.refiner_switch_at = self.config.get('refiner_switch_at', 0.4)
        self.refiner_model = self.config.get('refiner_model', 'stabilityai/stable-diffusion-xl-refiner-1.0')

        self._pipeline = None
        self._preprocessors = {}
        self._refiner = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return

        controlnets = []
        if self.use_controlnet and len(self.controlnet_configs) > 0:
            for ctrl_config in self.controlnet_configs:
                model_id = ctrl_config['model']
                controlnet = ControlNetModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                controlnets.append(controlnet)
        # perhaps redundant, added to correct colors
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

        if len(controlnets) > 0:
            self._pipeline = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnets,
                vae=vae,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        else:
            self._pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
                self.model_id,
                vae=vae,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )

        #  sequential cpu offload for maximum memory efficiency on CUDA
        if self.device == "cuda":
            self._pipeline.enable_sequential_cpu_offload()
            self._pipeline.enable_attention_slicing(1)
            self._pipeline.enable_vae_slicing()
        else:
            self._pipeline = self._pipeline.to(self.device)

        if self.use_refiner:
            self._refiner = SDRefiner(
                enabled=True,
                refiner_model_id=self.refiner_model,
                device=self.device,
                switch_at=self.refiner_switch_at,
                use_fp32=(self.device != "cuda")
            )
            self._refiner.load()

    def _load_preprocessors(self):
        if len(self._preprocessors) > 0:
            return

        if not self.use_controlnet:
            return

        for ctrl_config in self.controlnet_configs:
            ctrl_type = ctrl_config['type']
            if ctrl_type == 'canny':
                low = ctrl_config.get('canny_low_threshold', 100)
                high = ctrl_config.get('canny_high_threshold', 200)
                self._preprocessors['canny'] = CannyPreprocessor(low, high)
            elif ctrl_type == 'openpose':
                self._preprocessors['openpose'] = OpenPosePreprocessor()

    def _create_face_mask(self, image, bbox, landmarks=None):
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        if landmarks is not None and len(landmarks) > 0:
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 255)
        else:
            x1, y1, x2, y2 = map(int, bbox)
            padding = int((x2 - x1) * 0.1)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            mask[y1:y2, x1:x2] = 255

        if self.mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (self.mask_blur * 2 + 1, self.mask_blur * 2 + 1), 0)

        return Image.fromarray(mask)

    def generate(self, image, face_bbox, face_landmarks=None, output_size=(896, 896), save_mask_path=None):
        self._load_pipeline()
        self._load_preprocessors()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image = pil_image.resize(output_size, Image.LANCZOS)

        orig_h, orig_w = image.shape[:2]
        scale_x = output_size[0] / orig_w
        scale_y = output_size[1] / orig_h
        scaled_bbox = [
            face_bbox[0] * scale_x,
            face_bbox[1] * scale_y,
            face_bbox[2] * scale_x,
            face_bbox[3] * scale_y
        ]

        scaled_landmarks = None
        if face_landmarks is not None:
            scaled_landmarks = face_landmarks.astype(np.float64)
            scaled_landmarks[:, 0] *= scale_x
            scaled_landmarks[:, 1] *= scale_y

        mask = self._create_face_mask(np.array(pil_image), scaled_bbox, scaled_landmarks)
        if mask.size != pil_image.size:
            mask = mask.resize(pil_image.size, Image.LANCZOS)

        if save_mask_path is not None:
            mask.save(save_mask_path)
            print(f"Saved inpainting mask to: {save_mask_path}")

        control_images = []
        controlnet_scales = []

        if self.use_controlnet and len(self.controlnet_configs) > 0:
            for ctrl_config in self.controlnet_configs:
                ctrl_type = ctrl_config['type']
                preprocessor = self._preprocessors.get(ctrl_type)

                if preprocessor is not None:
                    control_image = preprocessor(pil_image)
                    if control_image.size != pil_image.size:
                        control_image = control_image.resize(pil_image.size, Image.LANCZOS)

                    if save_mask_path is not None:
                        debug_path = Path(save_mask_path).parent / f"controlnet_{ctrl_type}.png"
                        control_image.save(str(debug_path))

                    control_images.append(control_image)
                    controlnet_scales.append(ctrl_config.get('weight', 1.0))

        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        base_steps = int(self.steps * self.refiner_switch_at) if self.use_refiner else self.steps

        if len(control_images) > 0:
            output = self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=pil_image,
                mask_image=mask,
                control_image=control_images,
                controlnet_conditioning_scale=controlnet_scales,
                num_inference_steps=base_steps,
                strength=self.denoising_strength,
                guidance_scale=self.cfg_scale,
                generator=generator,
            ).images[0]
        else:
            output = self._pipeline(
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                image=pil_image,
                mask_image=mask,
                num_inference_steps=base_steps,
                strength=self.denoising_strength,
                guidance_scale=self.cfg_scale,
                generator=generator,
            ).images[0]

        if self.use_refiner and self._refiner is not None:
            output = self._refiner.refine(
                base_output=output,
                prompt=self.prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.cfg_scale,
                mask=mask,
            )

        return output

    def unload(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None

        if self._refiner is not None:
            self._refiner.unload()
            self._refiner = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()