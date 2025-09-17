from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from blanket.core.objects.detections import FaceDetection, FacialLandmarksDetection
from blanket.settings.individual_module_settings.face_detector_settings import FaceDetectorSettings
from blanket.settings.individual_module_settings.facial_landmarks_detector_settings import (
    FacialLandmarksDetectorSettings,
)


class BaseFaceDetector(ABC):
    def __init__(self, settings: FaceDetectorSettings):
        """
        Initialize base face detector.
        Args:
            settings (FaceDetectorSettings): Settings for face detector.
        """
        self.settings = settings

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[FaceDetection]:
        """
        Detect faces in an image (to be implemented by subclasses).
        Args:
            image_bgr (np.ndarray): Image in BGR format.
        Returns:
            list[FaceDetection]: List of detected faces.
        """
        pass


class BaseFacialLandmarksDetector(ABC):
    def __init__(self, settings: FacialLandmarksDetectorSettings):
        """
        Initialize base facial landmarks detector.
        Args:
            settings (FacialLandmarksDetectorSettings): Settings for detector.
        """
        self.settings = settings

    @abstractmethod
    def detect(self, image_bgr: np.ndarray, face_detection: FaceDetection) -> FacialLandmarksDetection:
        """
        Detect facial landmarks for a given face (to be implemented by subclasses).
        Args:
            image_bgr (np.ndarray): Image in BGR format.
            face_detection (FaceDetection): Detected face bounding box.
        Returns:
            FacialLandmarksDetection: Detected facial landmarks.
        """
        pass

    # TODO - add support for simultaneous detection of facial landmarks on multiple face detections
