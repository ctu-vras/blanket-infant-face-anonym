from __future__ import annotations

import cv2
import numpy as np

from blanket.core.detectors.base_detectors import BaseFacialLandmarksDetector
from blanket.core.geometry import SO3
from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.settings.individual_modules_settings.facial_landmarks_detector_settings import (
    FacialLandmarksDetectorSettings,
)

from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework


class SPIGAFacialLandmarksDetector(BaseFacialLandmarksDetector):
    def __init__(self, settings: FacialLandmarksDetectorSettings):
        """
        Initialize SPIGA facial landmarks detector with given settings.
        Args:
            settings (FacialLandmarksDetectorSettings): Settings for SPIGA detector.
        """
        super().__init__(settings)

        import torch

        model_cfg = ModelConfig(self.settings.model_name)

        # hack for devices without cuda
        self.is_cpu_only = not torch.cuda.is_available()
        if self.is_cpu_only:
            self._original_tensor_cuda = torch.Tensor.cuda
            self._original_module_cuda = torch.nn.Module.cuda
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            torch.nn.Module.cuda = lambda self, *args, **kwargs: self

        self._processor = SPIGAFramework(model_cfg, gpus=[0])

    def detect(self, image_bgr: np.ndarray, face_detection: FaceDetection) -> FacialLandmarksDetection:
        """
        Detect facial landmarks for a given face using SPIGA model.
        Args:
            image_bgr (np.ndarray): Image in BGR format.
            face_detection (FaceDetection): Detected face bounding box.
        Returns:
            FacialLandmarksDetection: Detected facial landmarks.
        """
        features = self._processor.inference(image_bgr, [face_detection.left_top_width_height])
        landmarks = np.array(features["landmarks"][0])
        headpose_ea_deg = np.array(features["headpose"][0])[:3]
        orientation_ea_deg = np.array([-(headpose_ea_deg[1]), headpose_ea_deg[0], -(headpose_ea_deg[2])])
        orientation_ea_rad = orientation_ea_deg / 180 * np.pi
        orientation_so3 = SO3.from_euler_angles(orientation_ea_rad, "yzx")
        # order based on https://euclideanspace.com/maths/geometry/rotations/conversions/eulerToMatrix/index.htm
        # which is used in the original SPIGA function (maybe could use that one instead)

        return FacialLandmarksDetection(landmarks.astype(int), orientation=orientation_so3)
