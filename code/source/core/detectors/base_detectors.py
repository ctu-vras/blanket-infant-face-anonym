from __future__ import annotations
from abc import ABC, abstractmethod
# from dataclasses import dataclass
import numpy as np

from code.source.core.objects.detections import FaceDetection, FacialLandmarksDetection
from code.source.settings.individual_modules_settings.face_detector_settings import FaceDetectorSettings
from code.source.settings.individual_modules_settings.facial_landmarks_detector_settings import (
    FacialLandmarksDetectorSettings)


class BaseFaceDetector(ABC):
    def __init__(self, settings: FaceDetectorSettings):
        self.settings = settings

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        pass


class BaseFacialLandmarksDetector(ABC):
    def __init__(self, settings: FacialLandmarksDetectorSettings):
        self.settings = settings

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, face_detection: FaceDetection) -> FacialLandmarksDetection:
        pass

    # TODO - add support for simultaneous detection of facial landmarks on multiple face detections
