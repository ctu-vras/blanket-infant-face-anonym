from typing import List, Optional
import cv2

from blanket.core.objects.primitives import ImagePrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.anonymization.base_anonymizer import BaseAnonymizer

class PixelationAnonymizer:
    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            # conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
        """
        Applies pixelation to each detected face region in the image.
        Args:
            input_image (ImagePrimitive): Image that is to be anonymized.
            face_detections (List[FaceDetection]): List of FaceDetection objects.
        Returns:
            ImagePrimitive: Anonymized image.
        """
        # pixel_size (int): Size of the pixel blocks.
        pixel_size = 16

        anonymized_image = input_image.image_bgr.copy()
        for face_detection in face_detections:
            l, t, r, b = face_detection.left, face_detection.top, face_detection.right, face_detection.bottom
            face_roi = anonymized_image[t:b, l:r]
            if face_roi.size == 0:
                continue
            h, w = face_roi.shape[:2]
            # Downscale and then upscale to create pixelation
            temp = cv2.resize(
                face_roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR
            )
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            anonymized_image[t:b, l:r] = pixelated
        return ImagePrimitive(anonymized_image, clockwise_rotation_index=input_image.clockwise_rotation_index)
