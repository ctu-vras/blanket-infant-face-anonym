import cv2


class GaussianBlurAnonymizer:
    def anonymize(self, image, detections, ksize=(31, 31), sigma=0):
        """
        Applies Gaussian blur to each detected face region in the image.
        Args:
            image (np.ndarray): BGR image.
            detections (list): List of FaceDetection objects.
            ksize (tuple): Kernel size for Gaussian blur.
            sigma (int): Sigma for Gaussian blur.
        Returns:
            np.ndarray: Anonymized image.
        """
        anonymized = image.copy()
        for det in detections:
            l, t, r, b = det.left, det.top, det.right, det.bottom
            face_roi = anonymized[t:b, l:r]
            if face_roi.size == 0:
                continue
            blurred = cv2.GaussianBlur(face_roi, ksize, sigma)
            anonymized[t:b, l:r] = blurred
        return anonymized
