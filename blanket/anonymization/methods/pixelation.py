import cv2
import numpy as np


class PixelationAnonymizer:
    def anonymize(self, image, detections, pixel_size=16):
        """
        Applies pixelation to each detected face region in the image.
        Args:
                image: np.ndarray (BGR image)
                detections: list of FaceDetection (with left_top_right_bottom property)
                pixel_size: int, size of the pixel blocks
        Returns:
                np.ndarray: anonymized image
        """
        anonymized = image.copy()
        for det in detections:
            l, t, r, b = det.left, det.top, det.right, det.bottom
            face_roi = anonymized[t:b, l:r]
            if face_roi.size == 0:
                continue
            h, w = face_roi.shape[:2]
            # Downscale and then upscale to create pixelation
            temp = cv2.resize(
                face_roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR
            )
            pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            anonymized[t:b, l:r] = pixelated
        return anonymized
