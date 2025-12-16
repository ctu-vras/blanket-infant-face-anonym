import cv2
import numpy as np
from PIL import Image


class CannyPreprocessor:
    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image: Image.Image) -> Image.Image:
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)


class OpenPosePreprocessor:
    def __init__(self):
        self.processor = None
        from controlnet_aux import OpenposeDetector
        self.processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.processor is None:
            return image

        return self.processor(image)