from __future__ import annotations

import cv2
import numpy as np

from blanket.core.detectors.base_detectors import BaseFacialLandmarksDetector
from blanket.core.geometry import SO3
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.settings.individual_modules_settings.facial_landmarks_detector_settings import (
    FacialLandmarksDetectorSettings,
)

# from SPIGA.spiga.inference.config import ModelConfig
# from SPIGA.spiga.inference.framework import SPIGAFramework


class SPIGAFacialLandmarksDetector(BaseFacialLandmarksDetector):
    def __init__(self, settings: FacialLandmarksDetectorSettings):
        super().__init__(settings)

        self._processor = SPIGAFramework(self.settings.model_name)

    def detect(self, image_bgr: np.ndarray, face_detection: FaceDetection) -> FacialLandmarksDetection:
        features = self._processor.inference(image_bgr, [face_detection.left_top_width_height])
        landmarks = np.array(features["landmarks"][0])
        headpose_ea_deg = np.array(features["headpose"][0])[:3]
        orientation_ea_deg = np.array([-(headpose_ea_deg[1]), headpose_ea_deg[0], -(headpose_ea_deg[2])])
        orientation_ea_rad = orientation_ea_deg / 180 * np.pi
        orientation_so3 = SO3.from_euler_angles(orientation_ea_rad, "yzx")
        # order based on https://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
        # which is used in the original SPIGA function (maybe could use that one instead)

        return FacialLandmarksDetection(landmarks.astype(int), orientation=orientation_so3)
