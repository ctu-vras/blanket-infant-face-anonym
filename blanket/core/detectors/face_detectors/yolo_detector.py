from __future__ import annotations
import numpy as np

import ultralytics

from blanket.core.objects.detections import FaceDetection
from blanket.settings.individual_modules_settings.face_detector_settings import FaceDetectorSettings
from blanket.core.detectors.base_detectors import BaseFaceDetector


class YOLOFaceDetector(BaseFaceDetector):
    def __init__(self, settings: FaceDetectorSettings):
        super().__init__(settings)

        self._detection_model = ultralytics.YOLO(settings.model_path)

    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        pred = self._detection_model(image_bgr,
                                     conf=self.settings.minimum_confidence,
                                     imgsz=self.settings.extra_parameters["detection_image_size"],
                                     verbose=False)[0]

        xywhs = ultralytics.utils.ops.xyxy2ltwh(pred.boxes.xyxy)
        confidences = pred.boxes.conf

        return [FaceDetection.from_left_top_width_height(ltwh.cpu().numpy(), conf.cpu().item())
                for ltwh, conf in zip(xywhs, confidences)]
