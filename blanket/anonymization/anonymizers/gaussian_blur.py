from typing import List, Optional
import cv2

from blanket.core.objects.primitives import ImagePrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.anonymization.base_anonymizer import BaseAnonymizer


class GaussianBlurAnonymizer(BaseAnonymizer):
    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            # conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
        """
        Applies Gaussian blur to each detected face region in the image.
        Args:
            input_image (ImagePrimitive): Image that is to be anonymized.
            face_detections (List[FaceDetection]): List of FaceDetection objects.
        Returns:
            ImagePrimitive: Anonymized image.
        """
        # ksize (tuple): Kernel size for Gaussian blur.
        # sigma (int): Sigma for Gaussian blur.
        kernel_size = (31, 31)
        sigma = 0

        anonymized_image = input_image.image_bgr.copy()
        for face_detection in face_detections:
            l, t, r, b = face_detection.left, face_detection.top, face_detection.right, face_detection.bottom
            face_roi = anonymized_image[t:b, l:r]
            if face_roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(face_roi, kernel_size, sigma)
            anonymized_image[t:b, l:r] = blurred
        return ImagePrimitive(anonymized_image, clockwise_rotation_index=input_image.clockwise_rotation_index)

