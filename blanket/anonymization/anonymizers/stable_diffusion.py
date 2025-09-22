from typing import List, Optional
from pathlib import Path
import os
import numpy as np
import cv2

import blanket.anonymization.anonymizers.stable_diffusion_api as api
from blanket.core.objects.primitives import ImagePrimitive, VideoPrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.anonymization.base_anonymizer import BaseAnonymizer


class StableDiffusionAnonymizer(BaseAnonymizer):
    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            # conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
        """
        Placeholder for Stable Diffusion anonymization method.
        Args:
            input_image (ImagePrimitive): Image that is to be anonymized.
            face_detections (List[FaceDetection]): List of FaceDetection objects.
        Raises:
            NotImplementedError: Always, as not implemented yet.
        """

        raise NotImplementedError("StableDiffusion image anonymization not implemented yet.")

        # # skipping current image because no faces were detected or using fallback detections from the conditioning
        # if len(detections) == 0:
        #     if image.conditioning.fallback_detections:  # conditioning fallback_detections is a list of nonzero length
        #         detections = image.conditioning.fallback_detections
        #     else:
        #         if self.settings.blacken_without_detections:
        #             image.anonymized_image_bgr = image.blackened_image
        #         else:
        #             image.anonymized_image_bgr = image.original_image_bgr
        #         return

        image_byte64 = api.encode_bgr_to_base64(input_image.image_bgr)

        for face_index, face_detection in enumerate(face_detections):
            image_byte64 = self._anonymize_face(input_image, image_byte64, face_detection)

        anonymized_image = ImagePrimitive(api.decode_base64_to_bgr(image_byte64), input_image.clockwise_rotation_index)
        return anonymized_image

    def _anonymize_face(self, original_image: ImagePrimitive, image_byte64: str, face_detection: FaceDetection):
        mask = face_detection.create_mask(original_image.shape)
        mask_byte64 = api.encode_bgr_to_base64(mask)

        padding = self._compute_padding(face_detection)

        payload = self._generate_payload(image_byte64, mask_byte64, padding)

        api_response = api.call_api(self._webui_server_url, "sdapi/v1/img2img", **payload)

        api_response_images = api_response.get("images")

        # TODO - try generating multiple images and picking the best one based on metrics
        image_byte64 = api_response_images[0]

        return image_byte64

    def _compute_padding(self, face_detection: FaceDetection) -> int:
        padding_method_name = self.settings.extra_parameters.get("anonymization_padding_method")

        if padding_method_name == "ratio":
            padding = round(self.settings.extra_parameters.get("anonymization_padding_ratio") * max(face_detection.width, face_detection.height))
        elif padding_method_name:
            padding = self.settings.extra_parameters.get("anonymization_padding_constant")
        else:
            raise ValueError(f"Unexpected value of anonymization_padding_method \"{padding_method_name}\"")

        return padding

    def _generate_payload(self, image_byte64, mask_byte64, padding=96):
        payload = self.settings.extra_parameters.get("payload")

        payload["init_images"] = [image_byte64]
        payload["mask"] = mask_byte64
        payload["inpaint_full_res_padding"] = padding

        return payload

    def _webui_server_url(self):
        ipv4_address = self.settings.extra_parameters.get("ipv4_address")
        port = self.settings.extra_parameters.get("port")

        return f"http://{ipv4_address}:{port}"

