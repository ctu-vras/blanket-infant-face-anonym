from typing import List, Optional
import cv2

from blanket.core.objects.primitives import ImagePrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.anonymization.base_anonymizer import BaseAnonymizer


class BlackBoxAnonymizer(BaseAnonymizer):
    # requires_face_detections = True
    # requires_facial_landmarks_detections = False

    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            # conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
        """
        Draws a black rectangle over each detected face in the image.
        Args:
            input_image (ImagePrimitive): Image that is to be anonymized.
            face_detections (List[FaceDetection]): List of FaceDetection objects.
        Returns:
            ImagePrimitive: Anonymized image.
        """
        anonymized_image = input_image.image_bgr.copy()
        for face_detection in face_detections:
            l, t, r, b = face_detection.left, face_detection.top, face_detection.right, face_detection.bottom
            cv2.rectangle(anonymized_image, (l, t), (r, b), (0, 0, 0), thickness=-1)
        return ImagePrimitive(anonymized_image, clockwise_rotation_index=input_image.clockwise_rotation_index)
