import cv2


class BlackBoxAnonymizer:
    def anonymize(self, image, detections):
        """
        Draws a black rectangle over each detected face in the image.
        Args:
                image: np.ndarray (BGR image)
                detections: list of FaceDetection (with left_top_right_bottom property)
        Returns:
                np.ndarray: anonymized image
        """
        anonymized = image.copy()
        for det in detections:
            l, t, r, b = det.left, det.top, det.right, det.bottom
            cv2.rectangle(anonymized, (l, t), (r, b), (0, 0, 0), thickness=-1)
        return anonymized
