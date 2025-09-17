from typing import List, Optional
import cv2

from blanket.core.objects.primitives import ImagePrimitive, VideoPrimitive
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.anonymization.base_anonymizer import BaseAnonymizer


class BlackBoxAnonymizer(BaseAnonymizer):
    requires_face_detections = True
    requires_facial_landmarks_detections = False

    def anonymize_image(
            self,
            input_image: ImagePrimitive,
            face_detections: Optional[List[FaceDetection]] = None,
            facial_landmarks_detection: Optional[List[FacialLandmarksDetection]] = None,
            conditioning: Optional[AnonymizationConditioning] = None
    ) -> ImagePrimitive:
    # def anonymize_image(self, input_image: ImagePrimitive) -> ImagePrimitive:
        """
        Draws a black rectangle over each detected face in the image.
        Args:
            input_image (ImagePrimitive): Image that is to be anonymized.
            detections (list): List of FaceDetection objects.
        Returns:
            np.ndarray: Anonymized image.
        """
        anonymized = image.copy()
        for det in detections:
            l, t, r, b = det.left, det.top, det.right, det.bottom
            cv2.rectangle(anonymized, (l, t), (r, b), (0, 0, 0), thickness=-1)
        return anonymized
